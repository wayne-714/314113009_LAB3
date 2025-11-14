import pandas as pd
import os
from config import Config
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset():
    """分析資料集"""
    config = Config()
    
    print("="*60)
    print("資料集分析")
    print("="*60)
    
    # 讀取 CSV
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    
    print(f"\n📊 基本統計:")
    print(f"訓練集樣本數: {len(train_df)}")
    print(f"驗證集樣本數: {len(val_df)}")
    print(f"總樣本數: {len(train_df) + len(val_df)}")
    
    # 類別分佈
    print(f"\n📈 訓練集類別分佈:")
    for col in config.CLASS_NAMES:
        count = train_df[col].sum()
        print(f"  {col:12s}: {count:4d} ({count/len(train_df)*100:5.1f}%)")
    
    print(f"\n📈 驗證集類別分佈:")
    for col in config.CLASS_NAMES:
        count = val_df[col].sum()
        print(f"  {col:12s}: {count:4d} ({count/len(val_df)*100:5.1f}%)")
    
    # 檢查圖片是否存在
    print(f"\n🔍 檢查圖片檔案...")
    
    train_missing = []
    for img_name in train_df['new_filename']:
        img_path = os.path.join(config.TRAIN_IMG_DIR, img_name)
        if not os.path.exists(img_path):
            train_missing.append(img_name)
    
    val_missing = []
    for img_name in val_df['new_filename']:
        img_path = os.path.join(config.VAL_IMG_DIR, img_name)
        if not os.path.exists(img_path):
            val_missing.append(img_name)
    
    if train_missing:
        print(f"⚠️  訓練集缺少 {len(train_missing)} 張圖片")
        print(f"   範例: {train_missing[:3]}")
    else:
        print(f"✓ 訓練集所有圖片都存在")
    
    if val_missing:
        print(f"⚠️  驗證集缺少 {len(val_missing)} 張圖片")
        print(f"   範例: {val_missing[:3]}")
    else:
        print(f"✓ 驗證集所有圖片都存在")
    
    # 檢查標籤格式
    print(f"\n🔍 檢查標籤格式...")
    for idx, row in train_df.head(5).iterrows():
        labels = row[config.CLASS_NAMES].values
        label_sum = labels.sum()
        if label_sum != 1:
            print(f"⚠️  第 {idx} 行標籤總和不等於 1: {labels}")
    
    print("\n✓ 資料檢查完成！")
    
    # 繪製類別分佈圖
    plot_class_distribution(train_df, val_df, config)


def plot_class_distribution(train_df, val_df, config):
    """繪製類別分佈圖"""
    train_counts = [train_df[col].sum() for col in config.CLASS_NAMES]
    val_counts = [val_df[col].sum() for col in config.CLASS_NAMES]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 訓練集
    ax1.bar(config.CLASS_NAMES, train_counts, color='steelblue')
    ax1.set_title('Training Set Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Class')
    for i, v in enumerate(train_counts):
        ax1.text(i, v + 10, str(v), ha='center', fontweight='bold')
    
    # 驗證集
    ax2.bar(config.CLASS_NAMES, val_counts, color='coral')
    ax2.set_title('Validation Set Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Class')
    for i, v in enumerate(val_counts):
        ax2.text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 類別分佈圖已儲存: class_distribution.png")
    plt.close()


if __name__ == '__main__':
    analyze_dataset()