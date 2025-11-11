import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from basicsr.data.paired_image_dataset import Dataset_S_IQA, Dataset_SuperRestoration_param
from torch.amp import GradScaler, autocast
import time
import numpy as np
from tqdm import tqdm
import random
import swanlab
import torchvision.models as models
from S_IQA_model import DenoiseRatePredictor
# 设置随机种子确保可复现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
checkpoint_dir = 'checkpoints-iqa-bc'
os.makedirs(checkpoint_dir, exist_ok=True)

# 配置参数
opt = {
    'dataroot_gt': '/mnt/mnt/data2/YTH/Image_Sonar_Generate/flsea-vi/canyons/flatiron/SonarIQA/clearSonar',
    'dataroot_lq': '/mnt/mnt/data2/YTH/Image_Sonar_Generate/flsea-vi/canyons/flatiron/SonarIQA/noiseSonar',
    'dataroot_param': '/mnt/mnt/data2/YTH/Image_Sonar_Generate/flsea-vi/canyons/flatiron/SonarIQA/params',
    'io_backend': {'type': 'disk'},  # IO后端类型
    'filename_tmpl': '{}',  # 文件名模板
    'gt_size': 512,  # 高质量图像裁剪尺寸
    'geometric_augs': True,  # 是否使用几何增强
    'scale': 1,  # 缩放比例
    'phase': 'train',  # 阶段：训练
}

# 实例化数据集
dataset = Dataset_S_IQA(opt)
print("Dataset length:", len(dataset))

# 分割数据集为训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)



# 初始化模型
model = DenoiseRatePredictor().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 设置混合精度训练
scaler = GradScaler()

# SwanLab初始化
swanlab.init(
    project="DenoiseRatePrediction",
    config={
        "learning_rate": 0.001,
        "batch_size": 4,
        "epochs": 50,
        "optimizer": "Adam",
        "loss_function": "MSE",
        "model_architecture": "DenoiseRatePredictor",
        "dataset_size": len(dataset),
        "train_size": train_size,
        "val_size": val_size
    }
)

# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, accumulation_steps=32):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    total_batches = len(dataloader)
    
    progress_bar = tqdm(enumerate(dataloader), total=total_batches, desc=f'Epoch {epoch+1}/{50} [Train]')
    
    for i, batch in progress_bar:
        lq_images = batch['lq']['img'].to(device)
        gt_images = batch['gt']['hq'].to(device)
        denoise_rate = batch['lq']['score'].to(device, dtype=torch.float)
        
        # 调整目标形状为 [batch_size, 1]
        denoise_rate = denoise_rate.view(-1, 1)
        
        # 前向传播
        with autocast(device):
            outputs = model(lq_images, gt_images)
            loss = criterion(outputs, denoise_rate)
            loss = loss / accumulation_steps  # 归一化损失
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 计算MAE
        mae = torch.abs(outputs - denoise_rate).mean().item()
        
        running_loss += loss.item() * accumulation_steps  # 还原损失
        running_mae += mae
        
        # 每accumulation_steps次迭代更新一次参数
        if (i + 1) % accumulation_steps == 0 or (i + 1) == total_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item() * accumulation_steps, 'mae': mae})
        
        # 记录每accumulation_steps个batch的指标
        if (i + 1) % accumulation_steps == 0:
            swanlab.log({
                "train_batch_loss": loss.item() * accumulation_steps,
                "train_batch_mae": mae,
                "train_step": epoch * total_batches + i
            })
    
    # 计算平均损失和MAE
    avg_loss = running_loss / total_batches
    avg_mae = running_mae / total_batches
    
    # 记录每个epoch的指标
    swanlab.log({
        "train_epoch_loss": avg_loss,
        "train_epoch_mae": avg_mae,
        "epoch": epoch
    })
    
    return avg_loss, avg_mae
# 验证函数
def validate_epoch(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    total_batches = len(dataloader)
    
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=total_batches, desc=f'Epoch {epoch+1}/{50} [Val]')
        for i, batch in progress_bar:
            lq_images = batch['lq']['img'].to(device)
            gt_images = batch['gt']['hq'].to(device)
            denoise_rate = batch['lq']['score'].to(device, dtype=torch.float)
            
            # 调整目标形状为 [batch_size, 1]
            denoise_rate = denoise_rate.view(-1, 1)
            
            # 前向传播
            outputs = model(lq_images, gt_images)
            loss = criterion(outputs, denoise_rate)
            
            # 计算MAE
            mae = torch.abs(outputs - denoise_rate).mean().item()
            
            running_loss += loss.item()
            running_mae += mae
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item(), 'mae': mae})
    
    # 计算平均损失和MAE
    avg_loss = running_loss / total_batches
    avg_mae = running_mae / total_batches
    
    # 记录每个epoch的验证指标
    swanlab.log({
        "val_epoch_loss": avg_loss,
        "val_epoch_mae": avg_mae,
        "epoch": epoch
    })
    
    # 更新学习率
    scheduler.step(avg_loss)
    
    return avg_loss, avg_mae

# 训练循环
best_val_loss = float('inf')
epochs = 50

for epoch in range(epochs):
    # 训练一个epoch
    train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
    
    # 验证一个epoch
    val_loss, val_mae = validate_epoch(model, val_loader, criterion, device, epoch)
    
    # 打印训练进度
    print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
    # 阶段性保存模型
    if (epoch + 1) % 1 == 0:
        print(f'Saving model at epoch {epoch + 1}...')
        model_save_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_save_path)
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_path = os.path.join(checkpoint_dir, f'best_denoise_rate_model.pth')
        torch.save(model.state_dict(), best_path)
        print(f'Best Model saved at epoch {epoch+1} with val loss: {val_loss:.4f}')

    model_save_path = os.path.join(checkpoint_dir, f'model_latest.pth')
    torch.save(model.state_dict(), model_save_path)
# 完成训练
swanlab.finish()
print("Training completed!")    