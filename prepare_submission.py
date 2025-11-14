import pandas as pd
import os
from config import Config
from inference import create_submission

def validate_submission(csv_path):
    """驗證提交檔案是否符合規則"""
    print("\n" + "="*60)
    print("驗證 Kaggle 提交檔案")
    print("="*60)
    
    # 讀取 CSV
    df = pd.read_csv(csv_path)
    
    # 檢查 1: 欄位名稱
    required_cols = ['new_filename', 'normal', 'bacteria', 'virus', 'COVID-19']
    assert list(df.columns) == required_cols, f"欄位錯誤！應為: {required_cols}"
    print("✓ 欄位名稱正確")
    
    # 檢查 2: 每列總和為 1
    row_sums = df[['normal', 'bacteria', 'virus', 'COVID-19']].sum(axis=1)
    assert all(row_sums == 1), "每列應該只有一個 1"
    print("✓ One-hot 格式正確")
    
    # 檢查 3: 只有 0 和 1
    for col in ['normal', 'bacteria', 'virus', 'COVID-19']:
        assert df[col].isin([0, 1]).all(), f"{col} 欄應只包含 0 或 1"
    print("✓ 數值正確 (只有 0 和 1)")
    
    # 檢查 4: 無重複檔名
    assert df['new_filename'].duplicated().sum() == 0, "檔名有重複"
    print("✓ 無重複檔名")
    
    # 檢查 5: 檔案數量
    print(f"\n📊 統計資訊:")
    print(f"  總樣本數: {len(df)}")
    print(f"  預測分佈:")
    for col in ['normal', 'bacteria', 'virus', 'COVID-19']:
        count = df[col].sum()
        print(f"    {col:12s}: {count:4d} ({count/len(df)*100:5.1f}%)")
    
    print("\n" + "="*60)
    print("✅ 提交檔案驗證通過！可以上傳至 Kaggle")
    print("="*60)
    
    return True


def main():
    """完整流程"""
    config = Config()
    
    # 1. 建立提交檔案
    print("🚀 開始建立提交檔案...")
    submission_df = create_submission(
        config,
        model_path=f"{config.CHECKPOINT_DIR}/best_model.pth"
    )
    
    # 2. 驗證
    validate_submission(config.OUTPUT_CSV)
    
    # 3. 提供上傳說明
    print(f"\n📤 上傳步驟:")
    print(f"1. 前往 Kaggle 競賽頁面")
    print(f"2. 點選 'Submit Predictions'")
    print(f"3. 上傳檔案: {config.OUTPUT_CSV}")
    print(f"4. 填寫提交說明 (例如: EfficientNet-B3 with TTA, F1={0.9834:.4f})")
    print(f"5. 點選 'Submit'")
    print(f"\n⚠️  注意: 每天只能提交 10 次，請謹慎使用！")
    

if __name__ == '__main__':
    main()