import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import shutil
from tqdm import tqdm


class CXRDataset(Dataset):
    """胸腔 X 光資料集"""
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'new_filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        # 讀取圖像
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"無法讀取圖像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 直方圖均衡化（CLAHE）
        image = self.apply_clahe(image)
        
        # 應用轉換
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if self.is_test:
            return image, img_name
        else:
            # 讀取標籤
            label = self.df.loc[idx, ['normal', 'bacteria', 'virus', 'COVID-19']].values.astype(np.float32)
            label = torch.tensor(label, dtype=torch.float32)
            # 轉換為 class index
            label_idx = torch.argmax(label).long()
            return image, label_idx
    
    @staticmethod
    def apply_clahe(image):
        """應用 CLAHE 提高對比度"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return image


def get_train_transforms(img_size=384, advanced=True):
    """訓練時的數據增強"""
    if advanced:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


def get_valid_transforms(img_size=384):
    """驗證/測試時的轉換"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size=384):
    """TTA 的多種轉換"""
    return [
        # Original
        A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Horizontal Flip
        A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Slight rotation
        A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=10, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Brightness adjust
        A.Compose([
            A.Resize(img_size, img_size),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]


def merge_train_val_images(config):
    """
    合併驗證集圖片到訓練集目錄
    這樣可以使用所有資料進行訓練
    """
    print("\n📁 正在合併訓練集與驗證集圖片...")
    
    # 確認目錄存在
    if not os.path.exists(config.VAL_IMG_DIR):
        print(f"⚠️  驗證集目錄不存在: {config.VAL_IMG_DIR}")
        return False
    
    if not os.path.exists(config.TRAIN_IMG_DIR):
        os.makedirs(config.TRAIN_IMG_DIR)
    
    # 取得驗證集的所有圖片
    val_images = [f for f in os.listdir(config.VAL_IMG_DIR) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(val_images) == 0:
        print(f"⚠️  驗證集目錄中沒有圖片")
        return False
    
    copied_count = 0
    for img_name in tqdm(val_images, desc="複製圖片"):
        src = os.path.join(config.VAL_IMG_DIR, img_name)
        dst = os.path.join(config.TRAIN_IMG_DIR, img_name)
        
        # 如果目標不存在才複製
        if not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
                copied_count += 1
            except Exception as e:
                print(f"⚠️  複製 {img_name} 失敗: {e}")
    
    print(f"✓ 完成！複製了 {copied_count} 張圖片到訓練目錄")
    return True


def create_dataloaders(config, combine_train_val=True):
    """創建 DataLoader（GPU 優化版）"""
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    
    print(f"\n📊 資料集資訊:")
    print(f"訓練集: {len(train_df)} 張")
    print(f"驗證集: {len(val_df)} 張")
    
    # 檢查類別分佈
    print("\n類別分佈 (訓練集):")
    for col in config.CLASS_NAMES:
        count = train_df[col].sum()
        print(f"  {col:12s}: {count:4d} ({count/len(train_df)*100:5.1f}%)")
    
    if combine_train_val:
        # 合併圖片檔案
        merge_train_val_images(config)
        
        # 合併 DataFrame
        full_train_df = pd.concat([train_df, val_df], ignore_index=True)
        
        print(f"\n✓ 合併後總樣本數: {len(full_train_df)}")
        
        # 創建完整訓練集
        train_dataset = CXRDataset(
            full_train_df,
            config.TRAIN_IMG_DIR,
            transform=get_train_transforms(config.IMG_SIZE, config.USE_ADVANCED_AUG)
        )
        val_dataset = None
        val_loader = None
    else:
        # 分開使用訓練集與驗證集
        train_dataset = CXRDataset(
            train_df,
            config.TRAIN_IMG_DIR,
            transform=get_train_transforms(config.IMG_SIZE, config.USE_ADVANCED_AUG)
        )
        
        val_dataset = CXRDataset(
            val_df,
            config.VAL_IMG_DIR,
            transform=get_valid_transforms(config.IMG_SIZE)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY if hasattr(config, 'PIN_MEMORY') else False,
            persistent_workers=True if config.NUM_WORKERS > 0 else False
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY if hasattr(config, 'PIN_MEMORY') else False,
        drop_last=True,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    
    return train_loader, val_loader