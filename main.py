import torch
import torch.optim as optim
from config import Config
from dataset import create_dataloaders
from model import create_model, create_criterion
from train import train
from inference import create_submission
from utils import set_seed
import os

def main():
    # 設定隨機種子
    set_seed(Config.SEED)
    
    # 創建必要目錄
    Config.create_dirs()
    
    print("="*60)
    print("CXR Multi-Class Classification - Training Pipeline")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Image Size: {Config.IMG_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print("="*60)
    
    # 創建 DataLoader
    print("\n📁 Loading data...")
    train_loader, val_loader = create_dataloaders(
        Config, 
        combine_train_val=True  # 設為 False 如果要使用驗證集
    )
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"✓ Val samples: {len(val_loader.dataset)}")
    
    # 創建模型
    print("\n🔨 Building model...")
    model = create_model(Config)
    print(f"✓ Model created: {Config.MODEL_NAME}")
    
    # 創建損失函數
    criterion = create_criterion(Config)
    print(f"✓ Loss function: {'Focal Loss' if Config.USE_FOCAL_LOSS else 'CrossEntropy'}")
    
    # 創建優化器
    if Config.OPTIMIZER == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=Config.WEIGHT_DECAY
        )
    print(f"✓ Optimizer: {Config.OPTIMIZER}")
    
    # 創建學習率調度器
    if Config.SCHEDULER == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=Config.T_MAX,
            eta_min=Config.MIN_LR
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
    print(f"✓ Scheduler: {Config.SCHEDULER}")
    
    # 訓練
    print("\n🚀 Starting training...")
    best_f1 = train(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler, Config
    )
    
    # 生成提交檔案
    print("\n📊 Generating submission file...")
    submission_df = create_submission(
        Config,
        model_path=f"{Config.CHECKPOINT_DIR}/best_model.pth"
    )
    
    print("\n🎉 All done!")
    print(f"Best Macro-F1 Score: {best_f1:.4f}")
    print(f"Submission file: {Config.OUTPUT_CSV}")

if __name__ == '__main__':
    main()