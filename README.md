# 314113009_LAB3
Multi-class Classification
胸腔 X 光影像多分類專案 - 使用深度學習進行 Normal、Bacteria、Virus、COVID-19 四類分類

---
## 專案簡介

本專案旨在透過深度學習技術，對胸腔 X 光影像進行自動分類，協助醫療人員快速辨識肺部疾病類型。專案使用 EfficientNet-B3 作為骨幹網路，並結合先進的影像處理與資料增強技術，達到高準確度的分類效果。

### 主要特色

- ✅ 使用預訓練的 EfficientNet-B3 模型
- ✅ 採用 CLAHE (對比度限制自適應直方圖均衡化) 進行影像前處理
- ✅ 實作多樣化資料增強策略
- ✅ 支援 GPU 加速與混合精度訓練
- ✅ 提供 Test Time Augmentation (TTA) 提升預測穩定性
- ✅ 自動化早停機制避免過擬合

---

## 資料集說明

### 資料來源

本專案使用胸腔 X 光影像資料集，包含四種分類：

| 類別 | 說明 |
|------|------|
| **Normal** | 正常肺部 |
| **Bacteria** | 細菌性肺炎 |
| **Virus** | 病毒性肺炎 |
| **COVID-19** | 新冠肺炎 |

### 資料格式

CSV 檔案格式：
```csv
new_filename,normal,bacteria,virus,COVID-19
0001.jpeg,0,1,0,0
0002.jpeg,1,0,0,0
```

- `new_filename`: 影像檔名
- 其餘欄位為 One-hot 編碼，每列僅有一個 `1`

---

## 環境需求

### 硬體需求

- **GPU**: NVIDIA GPU (建議 8GB+ VRAM)
  - 測試環境: RTX 2080 Ti (11GB)
- **RAM**: 16GB+
- **儲存空間**: 5GB+

### 軟體需求

- **作業系統**: Windows 10/11, Linux, macOS
- **Python**: 3.8+
- **CUDA**: 11.8 或 12.1 (如使用 GPU)

---

## 安裝步驟

### 1. 建立虛擬環境

```bash
# 使用 conda 
conda create -n cxr-classification python=3.9
conda activate cxr-classification
```

### 2. 安裝 PyTorch (GPU 版本)

**CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 安裝其他套件

```bash
pip install -r requirements.txt
```

### 4. 驗證安裝

```bash
python check_gpu.py
```

預期輸出：
```
🎮 CUDA 可用: True
🎮 GPU 數量: 1
🎮 GPU 0: NVIDIA GeForce RTX 2080 Ti
✓ GPU 運算測試成功！
```

---

## 專案結構

```
cxr-classification/
├── README.md                    # 專案說明文件
├── requirements.txt             # Python 套件需求
├── config.py                    # 配置檔案
├── main.py                      # 主程式 - 訓練流程
├── dataset.py                   # 資料載入與增強
├── model.py                     # 模型定義
├── train.py                     # 訓練邏輯
├── inference.py                 # 推論與預測
├── utils.py                     # 工具函數
├── check_gpu.py                 # GPU 環境檢查
├── check_data.py                # 資料集檢查
├── create_test_csv.py           # 建立測試集 CSV
├── prepare_submission.py        # 生成提交檔案
│
├── train_images/                # 訓練影像目錄
├── val_images/                  # 驗證影像目錄
├── test_images/                 # 測試影像目錄
├── train_data.csv               # 訓練標籤
├── val_data.csv                 # 驗證標籤
├── test_data_sample.csv         # 測試集檔案列表
│
├── checkpoints/                 # 模型檢查點
│   └── best_model.pth          # 最佳模型權重
└── submission.csv               # Kaggle 提交檔案
```

---
## 使用方法

### 1. 資料準備

確保資料結構如下：
```
project/
├── train_images/
│   ├── 0001.jpeg
│   ├── 0002.jpeg
│   └── ...
├── val_images/
│   ├── 1001.jpeg
│   └── ...
├── test_images/
│   ├── 2001.jpeg
│   └── ...
├── train_data.csv
├── val_data.csv
└── test_data_sample.csv
```

### 2. 檢查資料

```bash
# 檢查 GPU 環境
python check_gpu.py

# 檢查資料集完整性
python check_data.py
```

### 3. 訓練模型

```bash
python main.py
```

訓練過程中會顯示：
```
==================================================
Epoch 1/50
==================================================
Training: 100%|████████| 393/393 [01:02<00:00, 6.30it/s, loss=0.421]
Train Loss: 0.4210 | Train Macro-F1: 0.8523
✓ Best model saved with F1: 0.8523
```

### 4. 生成提交檔案

訓練完成後會自動生成 `submission.csv`，或手動執行：

```bash
python prepare_submission.py
```
## 作者

**張涵崴 (wayne-714)**  
- 專案: Multi-Class Classification
- 日期: 2025-11-11
