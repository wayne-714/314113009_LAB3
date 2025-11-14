import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from dataset import CXRDataset, get_valid_transforms, get_tta_transforms


def predict_with_tta(model, test_loader, tta_transforms, device, config):
    """使用 TTA 進行預測"""
    model.eval()
    
    all_predictions = []
    all_filenames = []
    
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc='Predicting with TTA'):
            batch_preds = []
            
            # 對每種 TTA 轉換進行預測
            for tta_transform in tta_transforms:
                images_tta = images.to(device)
                outputs = model(images_tta)
                probs = torch.softmax(outputs, dim=1)
                batch_preds.append(probs.cpu().numpy())
            
            # 平均所有 TTA 預測
            avg_preds = np.mean(batch_preds, axis=0)
            all_predictions.append(avg_preds)
            all_filenames.extend(filenames)
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    return all_predictions, all_filenames


def predict(model, test_loader, device):
    """標準預測"""
    model.eval()
    
    all_predictions = []
    all_filenames = []
    
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_predictions.append(probs.cpu().numpy())
            all_filenames.extend(filenames)
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    return all_predictions, all_filenames


def create_submission(config, model_path=None):
    """創建提交檔案"""
    from model import create_model
    
    # 載入模型
    model = create_model(config)
    if model_path:
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 載入模型: {model_path}")
        print(f"  Epoch: {checkpoint['epoch']}, F1: {checkpoint['best_f1']:.4f}")
    model.to(config.DEVICE)
    model.eval()
    
    # 讀取或建立測試資料 CSV
    if os.path.exists(config.TEST_SAMPLE_CSV):
        print(f"✓ 找到測試集 CSV: {config.TEST_SAMPLE_CSV}")
        test_df = pd.read_csv(config.TEST_SAMPLE_CSV)
    else:
        print(f"⚠️  未找到 {config.TEST_SAMPLE_CSV}，自動建立...")
        
        # 從目錄建立測試集 CSV
        if not os.path.exists(config.TEST_IMG_DIR):
            raise FileNotFoundError(f"測試集目錄不存在: {config.TEST_IMG_DIR}")
        
        test_images = []
        for f in os.listdir(config.TEST_IMG_DIR):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(f)
        
        test_images.sort()
        
        test_df = pd.DataFrame({
            'new_filename': test_images,
            'normal': 0,
            'bacteria': 0,
            'virus': 0,
            'COVID-19': 0
        })
        
        # 儲存以備後用
        test_df.to_csv(config.TEST_SAMPLE_CSV, index=False)
        print(f"✓ 建立測試集 CSV: {config.TEST_SAMPLE_CSV} ({len(test_df)} 張)")
    
    print(f"測試集圖片數量: {len(test_df)}")
    
    # 創建測試集
    test_dataset = CXRDataset(
        test_df,
        config.TEST_IMG_DIR,
        transform=get_valid_transforms(config.IMG_SIZE),
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Windows 建議用 0
        pin_memory=True
    )
    
    # 預測
    if config.USE_TTA:
        print("🔄 使用 TTA 預測...")
        tta_transforms = get_tta_transforms(config.IMG_SIZE)
        predictions, filenames = predict_with_tta(
            model, test_loader, tta_transforms, config.DEVICE, config
        )
    else:
        print("🔄 標準預測...")
        predictions, filenames = predict(model, test_loader, config.DEVICE)
    
    # 轉換為 one-hot 格式
    pred_labels = np.argmax(predictions, axis=1)
    one_hot = np.zeros((len(pred_labels), config.NUM_CLASSES), dtype=int)
    one_hot[np.arange(len(pred_labels)), pred_labels] = 1
    
    # 創建提交 DataFrame
    submission_df = pd.DataFrame({
        'new_filename': filenames,
        'normal': one_hot[:, 0],
        'bacteria': one_hot[:, 1],
        'virus': one_hot[:, 2],
        'COVID-19': one_hot[:, 3]
    })
    
    # 確保順序與原始測試檔案一致
    submission_df = test_df[['new_filename']].merge(
        submission_df, on='new_filename', how='left'
    )
    
    # 處理可能的 NaN（如果有圖片沒有被預測到）
    submission_df.fillna(0, inplace=True)
    submission_df[['normal', 'bacteria', 'virus', 'COVID-19']] = \
        submission_df[['normal', 'bacteria', 'virus', 'COVID-19']].astype(int)
    
    # 儲存 CSV
    submission_df.to_csv(config.OUTPUT_CSV, index=False)
    print(f"\n✓ 提交檔案已儲存: {config.OUTPUT_CSV}")
    print(f"   Shape: {submission_df.shape}")
    
    # 顯示預測分佈
    print(f"\n📊 預測結果分佈:")
    for col in config.CLASS_NAMES:
        count = submission_df[col].sum()
        print(f"  {col:12s}: {count:4d} ({count/len(submission_df)*100:5.1f}%)")
    
    print(f"\n前 10 筆預測:")
    print(submission_df.head(10))
    
    # 驗證格式
    print(f"\n🔍 驗證提交格式...")
    assert submission_df.shape[1] == 5, f"應該有 5 欄，實際: {submission_df.shape[1]}"
    assert all(submission_df[config.CLASS_NAMES].sum(axis=1) == 1), "每列應該只有一個 1"
    assert submission_df['new_filename'].duplicated().sum() == 0, "檔名不應重複"
    print("✓ 格式驗證通過！")
    
    return submission_df