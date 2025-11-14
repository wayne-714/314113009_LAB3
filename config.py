import torch
import os

class Config:
    # ==================== GPU 設定 ====================
    # 自動選擇最佳設備
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        # 如果有多張 GPU，可以指定使用哪一張
        GPU_ID = 0  # 使用第 0 張 GPU，如果有多張可以改成 1, 2...
        torch.cuda.set_device(GPU_ID)
        print(f"🎮 使用 GPU: {torch.cuda.get_device_name(GPU_ID)}")
    else:
        DEVICE = torch.device('cpu')
        print("⚠️  未偵測到 GPU，使用 CPU 訓練")
    
    # GPU 優化設定
    PIN_MEMORY = True if torch.cuda.is_available() else False
    USE_AMP = True if torch.cuda.is_available() else False  # 混合精度訓練
    
    # ==================== 路徑設定 ====================
    TRAIN_IMG_DIR = 'train_images'
    VAL_IMG_DIR = 'val_images'
    TEST_IMG_DIR = 'test_images'
    TRAIN_CSV = 'train_data.csv'
    VAL_CSV = 'val_data.csv'
    TEST_SAMPLE_CSV = 'test_data_sample.csv'
    OUTPUT_CSV = 'submission.csv'
    CHECKPOINT_DIR = 'checkpoints'
    
    # ==================== 模型設定 ====================
    MODEL_NAME = 'efficientnet_b3'  # 可選: resnet50, efficientnet_b0~b7
    NUM_CLASSES = 4
    IMG_SIZE = 384  # EfficientNet-B3 建議 384
    
    # ==================== 訓練設定 ====================
    # 根據 GPU 記憶體調整批次大小
    if torch.cuda.is_available():
        # 檢查 GPU 記憶體
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_memory >= 24:  # 24GB+ (e.g., RTX 4090, A100)
            BATCH_SIZE = 32
            NUM_WORKERS = 8
        elif total_memory >= 16:  # 16GB+ (e.g., RTX 4080, V100)
            BATCH_SIZE = 24
            NUM_WORKERS = 6
        elif total_memory >= 12:  # 12GB+ (e.g., RTX 3080, RTX 4070)
            BATCH_SIZE = 16
            NUM_WORKERS = 4
        elif total_memory >= 8:   # 8GB+ (e.g., RTX 3070, RTX 4060)
            BATCH_SIZE = 12
            NUM_WORKERS = 4
        else:  # < 8GB (e.g., RTX 3060)
            BATCH_SIZE = 8
            NUM_WORKERS = 2
        print(f"🎮 GPU 記憶體: {total_memory:.1f}GB, 批次大小: {BATCH_SIZE}")
    else:
        BATCH_SIZE = 4
        NUM_WORKERS = 2
    
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # ==================== 優化器設定 ====================
    OPTIMIZER = 'AdamW'
    SCHEDULER = 'CosineAnnealingLR'
    T_MAX = NUM_EPOCHS
    MIN_LR = 1e-6
    
    # ==================== Early Stopping ====================
    PATIENCE = 10
    
    # ==================== 損失函數 ====================
    USE_FOCAL_LOSS = False
    FOCAL_ALPHA = [1.0, 1.0, 1.0, 1.0]
    FOCAL_GAMMA = 2.0
    
    # ==================== 資料增強 ====================
    USE_ADVANCED_AUG = True
    
    # ==================== TTA ====================
    USE_TTA = True
    TTA_TRANSFORMS = 4
    
    # ==================== 其他設定 ====================
    SEED = 42
    CLASS_NAMES = ['normal', 'bacteria', 'virus', 'COVID-19']
    
    # Gradient accumulation (如果記憶體不足，增加這個值)
    ACCUMULATION_STEPS = 1
    
    # 混合精度訓練
    USE_AMP = True if torch.cuda.is_available() else False
    
    @staticmethod
    def create_dirs():
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)