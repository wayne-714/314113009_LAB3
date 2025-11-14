import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from utils import AverageMeter, calculate_metrics, EarlyStopping, save_checkpoint

def train_one_epoch(model, train_loader, criterion, optimizer, device, config, scaler=None):
    """訓練一個 epoch"""
    model.train()
    losses = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Mixed precision training
        if config.USE_AMP and scaler:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / config.ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / config.ACCUMULATION_STEPS
            loss.backward()
            
            if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # 記錄
        losses.update(loss.item() * config.ACCUMULATION_STEPS, images.size(0))
        
        # 預測
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新進度條
        pbar.set_postfix({'loss': losses.avg})
    
    # 計算 F1
    train_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return losses.avg, train_f1


def validate(model, val_loader, criterion, device):
    """驗證"""
    model.eval()
    losses = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            losses.update(loss.item(), images.size(0))
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': losses.avg})
    
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return losses.avg, val_f1, all_preds, all_labels


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    """完整訓練流程"""
    best_f1 = 0.0
    early_stopping = EarlyStopping(patience=config.PATIENCE, mode='max')
    scaler = GradScaler() if config.USE_AMP else None
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"{'='*50}")
        
        # 訓練
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, 
            config.DEVICE, config, scaler
        )
        
        print(f"Train Loss: {train_loss:.4f} | Train Macro-F1: {train_f1:.4f}")
        
        # 驗證（如果有驗證集）
        if val_loader is not None:
            val_loss, val_f1, val_preds, val_labels = validate(
                model, val_loader, criterion, config.DEVICE
            )
            print(f"Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}")
            current_f1 = val_f1
        else:
            current_f1 = train_f1
        
        # 更新學習率
        if config.SCHEDULER == 'ReduceLROnPlateau':
            scheduler.step(current_f1)
        else:
            scheduler.step()
        
        # 儲存最佳模型
        if current_f1 > best_f1:
            best_f1 = current_f1
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_f1,
                f"{config.CHECKPOINT_DIR}/best_model.pth"
            )
            print(f"✓ Best model saved with F1: {best_f1:.4f}")
        
        # Early stopping
        if early_stopping(current_f1):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\n{'='*50}")
    print(f"Training completed! Best Macro-F1: {best_f1:.4f}")
    print(f"{'='*50}")
    
    return best_f1