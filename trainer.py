import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

class GANLoss(nn.Module):
    """
    GAN损失函数，支持原始GAN损失和LSGAN损失
    """
    def __init__(self, gan_mode='hinge', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.loss = None
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')
    
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction, target_is_real):
        if self.gan_mode == 'hinge':
            if target_is_real:
                return -torch.mean(torch.min(torch.zeros_like(prediction), -1 + prediction))
            else:
                return -torch.mean(torch.min(torch.zeros_like(prediction), -1 - prediction))
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)


class FeatureMatchingLoss(nn.Module):
    """
    特征匹配损失，用于稳定GAN训练
    """
    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()
        self.criterion = nn.L1Loss()
    
    def __call__(self, fake_features, real_features):
        loss = 0.0
        num_D = len(fake_features)
        
        for i in range(num_D):
            for j in range(len(fake_features[i])):
                loss += self.criterion(fake_features[i][j], real_features[i][j].detach())
        
        return loss / num_D


class PerceptualLoss(nn.Module):
    """
    感知损失，使用预训练的VGG网络提取特征
    """
    def __init__(self, layers=[2, 7, 12, 21, 30]):
        super(PerceptualLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.layers = layers
        
        # 加载预训练的VGG19
        vgg = models.vgg19(pretrained=True).features
        self.model = nn.ModuleList()
        
        # 提取指定层
        i = 0
        for layer in range(max(self.layers) + 1):
            self.model.append(vgg[layer])
            
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features
    
    def __call__(self, fake_img, real_img):
        # 归一化到VGG的输入范围
        fake_img = (fake_img + 1) / 2
        real_img = (real_img + 1) / 2
        
        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(fake_img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(fake_img.device)
        
        fake_img = (fake_img - mean) / std
        real_img = (real_img - mean) / std
        
        # 提取特征
        fake_features = self.forward(fake_img)
        real_features = self.forward(real_img)
        
        # 计算损失
        loss = 0.0
        for i in range(len(fake_features)):
            loss += self.criterion(fake_features[i], real_features[i].detach())
        
        return loss


class LowLightColorizationTrainer:
    """
    低光彩色化模型的训练器
    """
    def __init__(self, model, train_loader, test_loader, device, 
                 lr=0.0002, beta1=0.5, beta2=0.999, lambda_L1=100.0, 
                 lambda_FM=10.0, lambda_VGG=10.0, z_dim=8):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.z_dim = z_dim
        
        # 将模型移动到指定设备
        self.model.to(device)
        
        # 优化器
        self.optimizer_G = optim.Adam(self.model.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_D = optim.Adam(self.model.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        
        # 损失函数
        self.criterionGAN = GANLoss(gan_mode='hinge').to(device)
        self.criterionL1 = nn.L1Loss()
        self.criterionFM = FeatureMatchingLoss()
        self.criterionVGG = PerceptualLoss()
        
        # 损失权重
        self.lambda_L1 = lambda_L1
        self.lambda_FM = lambda_FM
        self.lambda_VGG = lambda_VGG
        
        # 记录训练过程
        self.train_history = {
            'G_losses': [],
            'D_losses': [],
            'L1_losses': [],
            'FM_losses': [],
            'VGG_losses': []
        }
        
    def train_step(self, data):
        # 准备输入数据
        mono = data['mono'].to(self.device)
        real_color = data['color'].to(self.device)
        semantic = data['semantic'].to(self.device)
        
        # 生成随机风格向量
        z = torch.randn(mono.size(0), self.z_dim, device=self.device)
        
        # 前向传播
        fake_color, attention_maps = self.model.generator(mono, semantic, z)
        
        # 更新判别器
        self.optimizer_D.zero_grad()
        
        # 真实图像的判别结果
        pred_real, real_features = [], []
        for d_out in self.model.discriminator(real_color, semantic):
            pred, feat = d_out
            pred_real.append(pred)
            real_features.append(feat)
        
        # 生成图像的判别结果
        pred_fake, fake_features = [], []
        for d_out in self.model.discriminator(fake_color.detach(), semantic):
            pred, feat = d_out
            pred_fake.append(pred)
            fake_features.append(feat)
        
        # 判别器损失
        D_loss_real = 0
        D_loss_fake = 0
        for pred in pred_real:
            D_loss_real += self.criterionGAN(pred, True)
        for pred in pred_fake:
            D_loss_fake += self.criterionGAN(pred, False)
        
        D_loss = (D_loss_real + D_loss_fake) * 0.5
        D_loss.backward()
        self.optimizer_D.step()
        
        # 更新生成器
        self.optimizer_G.zero_grad()
        
        # 重新计算生成图像的判别结果（因为判别器已更新）
        pred_fake, fake_features = [], []
        for d_out in self.model.discriminator(fake_color, semantic):
            pred, feat = d_out
            pred_fake.append(pred)
            fake_features.append(feat)
        
        # 生成器对抗损失
        G_loss_GAN = 0
        for pred in pred_fake:
            G_loss_GAN += self.criterionGAN(pred, True)
        
        # L1损失
        G_loss_L1 = self.criterionL1(fake_color, real_color) * self.lambda_L1
        
        # 特征匹配损失
        G_loss_FM = self.criterionFM(fake_features, real_features) * self.lambda_FM
        
        # 感知损失
        G_loss_VGG = self.criterionVGG(fake_color, real_color) * self.lambda_VGG
        
        # 总生成器损失
        G_loss = G_loss_GAN + G_loss_L1 + G_loss_FM + G_loss_VGG
        G_loss.backward()
        self.optimizer_G.step()
        
        # 记录损失
        self.train_history['G_losses'].append(G_loss.item())
        self.train_history['D_losses'].append(D_loss.item())
        self.train_history['L1_losses'].append(G_loss_L1.item())
        self.train_history['FM_losses'].append(G_loss_FM.item())
        self.train_history['VGG_losses'].append(G_loss_VGG.item())
        
        return {
            'G_loss': G_loss.item(),
            'D_loss': D_loss.item(),
            'L1_loss': G_loss_L1.item(),
            'FM_loss': G_loss_FM.item(),
            'VGG_loss': G_loss_VGG.item()
        }
    
    def validate(self):
        self.model.eval()
        val_losses = {
            'G_loss': 0.0,
            'L1_loss': 0.0,
            'VGG_loss': 0.0
        }
        
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                # 准备输入数据
                mono = data['mono'].to(self.device)
                real_color = data['color'].to(self.device)
                semantic = data['semantic'].to(self.device)
                
                # 生成随机风格向量
                z = torch.randn(mono.size(0), self.z_dim, device=self.device)
                
                # 前向传播
                fake_color, _ = self.model.generator(mono, semantic, z)
                
                # 计算损失
                L1_loss = self.criterionL1(fake_color, real_color) * self.lambda_L1
                VGG_loss = self.criterionVGG(fake_color, real_color) * self.lambda_VGG
                
                val_losses['L1_loss'] += L1_loss.item()
                val_losses['VGG_loss'] += VGG_loss.item()
                
                # 只处理少量批次
                if i >= 10:
                    break
        
        # 计算平均损失
        for k in val_losses:
            val_losses[k] /= min(len(self.test_loader), 10)
        
        self.model.train()
        return val_losses
    
    def train(self, num_epochs, save_dir, save_interval=5, sample_interval=1):
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_losses = {
                'G_loss': 0.0,
                'D_loss': 0.0,
                'L1_loss': 0.0,
                'FM_loss': 0.0,
                'VGG_loss': 0.0
            }
            
            for i, data in enumerate(self.train_loader):
                step_losses = self.train_step(data)
                
                # 累加损失
                for k in epoch_losses:
                    epoch_losses[k] += step_losses[k]
                
                # 打印进度
                if i % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(self.train_loader)}] "
                          f"G_loss: {step_losses['G_loss']:.4f}, D_loss: {step_losses['D_loss']:.4f}, "
                          f"L1_loss: {step_losses['L1_loss']:.4f}")
            
            # 计算平均损失
            for k in epoch_losses:
                epoch_losses[k] /= len(self.train_loader)
            
            # 验证
            val_losses = self.validate()
            
            # 打印epoch结果
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s | "
                  f"Train G_loss: {epoch_losses['G_loss']:.4f}, D_loss: {epoch_losses['D_loss']:.4f} | "
                  f"Val L1_loss: {val_losses['L1_loss']:.4f}, VGG_loss: {val_losses['VGG_loss']:.4f}")
            
            # 保存样本
            if (epoch + 1) % sample_interval == 0:
                self.save_samples(epoch + 1, save_dir)
            
            # 保存模型
            if (epoch + 1) % save_interval == 0:
                self.save_model(epoch + 1, os.path.join(save_dir, 'checkpoints'))
        
        # 保存最终模型
        self.save_model('final', os.path.join(save_dir, 'checkpoints'))
        
        # 绘制训练曲线
        self.plot_losses(save_dir)
    
    def save_samples(self, epoch, save_dir):
        """保存生成的样本图像"""
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                mono = data['mono'].to(self.device)
                real_color = data['color'].to(self.device)
                semantic = data['semantic'].to(self.device)
                
                # 使用固定的风格向量生成样本
                z = torch.randn(mono.size(0), self.z_dim, device=self.device)
                fake_color, attention_maps = self.model.generator(mono, semantic, z)
                
                # 创建网格图像
                img_grid = make_grid(torch.cat([
                    mono.repeat(1, 3, 1, 1),  # 将单通道转为三通道以便显示
                    fake_color,
                    real_color
                ], dim=0), nrow=mono.size(0))
                
                # 保存图像
                save_image(img_grid, os.path.join(save_dir, 'samples', f'epoch_{epoch}_batch_{i}.png'), normalize=True)
                
                # 保存注意力图
                if attention_maps:
                    for j, att_map in enumerate(attention_maps):
                        att_grid = make_grid(att_map, nrow=int(np.sqrt(att_map.size(0))))
                        save_image(att_grid, os.path.join(save_dir, 'samples', f'epoch_{epoch}_batch_{i}_attention_{j}.png'), normalize=True)
                
                # 只保存少量批次
                if i >= 2:
                    break
        
        self.model.train()
    
    def save_model(self, epoch, save_dir):
        """保存模型检查点"""
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'train_history': self.train_history
        }, os.path.join(save_dir, f'model_epoch_{epoch}.pth'))
    
    def load_model(self, checkpoint_path):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path)
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.train_history = checkpoint['train_history']
        return checkpoint['epoch']
    
    def plot_losses(self, save_dir):
        """绘制训练损失曲线"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.train_history['G_losses'], label='Generator')
        plt.plot(self.train_history['D_losses'], label='Discriminator')
        plt.title('GAN Losses')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(self.train_history['L1_losses'], label='L1')
        plt.title('L1 Loss')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(self.train_history['FM_losses'], label='Feature Matching')
        plt.title('Feature Matching Loss')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(self.train_history['VGG_losses'], label='VGG Perceptual')
        plt.title('VGG Perceptual Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_losses.png'))
        plt.close()


def visualize_attention(attention_map, save_path):
    """
    可视化注意力图
    
    参数:
        attention_map: 注意力图张量
        save_path: 保存路径
    """
    # 确保注意力图是CPU张量
    attention_map = attention_map.cpu().detach()
    
    # 如果是批次数据，只取第一个样本
    if len(attention_map.shape) == 4:
        attention_map = attention_map[0]
    
    # 如果有多个通道，取平均值
    if len(attention_map.shape) > 2:
        attention_map = attention_map.mean(dim=0)
    
    # 归一化到[0,1]
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # 转换为numpy数组
    attention_map = attention_map.numpy()
    
    # 绘制热力图
    plt.figure(figsize=(8, 8))
    plt.imshow(attention_map, cmap='jet')
    plt.colorbar()
    plt.title('Attention Map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_with_different_styles(model, mono, semantic, num_styles=5, device='cuda'):
    """
    使用不同的风格向量生成多样化的彩色图像
    
    参数:
        model: 低光彩色化模型
        mono: 低光单色图像
        semantic: 语义分割图
        num_styles: 生成的风格数量
        device: 设备
    
    返回:
        生成的彩色图像列表
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in range(num_styles):
            # 生成随机风格向量
            z = torch.randn(1, model.generator.z_dim, device=device)
            
            # 生成彩色图像
            fake_color, _ = model.generator(mono, semantic, z)
            results.append(fake_color)
    
    model.train()
    return results
