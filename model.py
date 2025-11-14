import torch
import torch.nn as nn
import timm

class CXRClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b3', num_classes=4, pretrained=True):
        super(CXRClassifier, self).__init__()
        
        # 使用 timm 載入預訓練模型
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分類頭
            global_pool=''
        )
        
        # 獲取特徵維度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            if len(features.shape) == 4:
                num_features = features.shape[1]
            else:
                num_features = features.shape[-1]
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分類頭
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        
        # 如果是 4D tensor (batch, channels, h, w)
        if len(features.shape) == 4:
            features = self.global_pool(features)
            features = features.view(features.size(0), -1)
        
        output = self.classifier(features)
        return output


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_model(config):
    """創建模型"""
    model = CXRClassifier(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=True
    )
    return model.to(config.DEVICE)


def create_criterion(config):
    """創建損失函數"""
    if config.USE_FOCAL_LOSS:
        alpha = torch.tensor(config.FOCAL_ALPHA).to(config.DEVICE)
        criterion = FocalLoss(alpha=alpha, gamma=config.FOCAL_GAMMA)
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion