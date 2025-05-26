import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

class SPADE(nn.Module):
    """
    语义自适应去归一化模块 (Spatially-Adaptive Denormalization)
    用于将语义信息注入到生成器的特征图中
    """
    def __init__(self, norm_nc, semantic_nc):
        super().__init__()
        
        # 参数化学习的缩放和偏移
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        
        # 语义特征处理网络
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(semantic_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        
    def forward(self, x, segmap):
        # 首先应用参数无关的归一化
        normalized = self.param_free_norm(x)
        
        # 调整分割图大小以匹配特征图
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        
        # 获取调制参数
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        
        # 应用调制
        out = normalized * (1 + gamma) + beta
        
        return out


class CBAM(nn.Module):
    """
    卷积块注意力模块 (Convolutional Block Attention Module)
    提供通道注意力和空间注意力，便于可视化
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att
        
        # 空间注意力
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        
        # 应用空间注意力
        output = x_channel * spatial_att
        
        return output, spatial_att  # 返回输出和空间注意力图，便于可视化


class SPADEResBlock(nn.Module):
    """
    SPADE残差块，用于生成器中
    """
    def __init__(self, in_channels, out_channels, semantic_nc, use_attention=False):
        super(SPADEResBlock, self).__init__()
        
        self.learned_shortcut = (in_channels != out_channels)
        middle_channels = min(in_channels, out_channels)
        
        # 主路径
        self.conv_0 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)
        
        # 快捷路径
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # SPADE模块
        self.norm_0 = SPADE(in_channels, semantic_nc)
        self.norm_1 = SPADE(middle_channels, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(in_channels, semantic_nc)
        
        # 注意力模块
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(out_channels)
        
    def forward(self, x, segmap):
        # 快捷路径
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, segmap))
        else:
            x_s = x
        
        # 主路径
        dx = self.conv_0(F.leaky_relu(self.norm_0(x, segmap), 0.2))
        dx = self.conv_1(F.leaky_relu(self.norm_1(dx, segmap), 0.2))
        
        # 残差连接
        out = x_s + dx
        
        # 应用注意力（如果启用）
        attention_map = None
        if self.use_attention:
            out, attention_map = self.attention(out)
        
        return out, attention_map


class StyleEncoder(nn.Module):
    """
    风格编码器，用于将随机噪声映射为风格向量
    """
    def __init__(self, z_dim=8, style_dim=512):
        super(StyleEncoder, self).__init__()
        
        self.mapping = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, style_dim),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, z):
        return self.mapping(z)


class SPADEGenerator(nn.Module):
    """
    基于SPADE的生成器，结合风格控制和注意力机制
    """
    def __init__(self, input_nc=1, output_nc=3, semantic_nc=20, ngf=64, z_dim=8, use_attention=True):
        super(SPADEGenerator, self).__init__()
        
        self.z_dim = z_dim
        self.use_attention = use_attention
        
        # 风格编码器
        self.style_encoder = StyleEncoder(z_dim=z_dim)
        
        # 初始卷积层
        self.conv_in = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1)
        
        # 下采样编码器
        self.down1 = nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1)
        
        # 瓶颈层
        self.bottleneck = nn.Conv2d(ngf*8, ngf*8, kernel_size=3, padding=1)
        
        # SPADE残差块
        self.spade_res1 = SPADEResBlock(ngf*8, ngf*8, semantic_nc, use_attention=use_attention)
        self.spade_res2 = SPADEResBlock(ngf*8, ngf*8, semantic_nc, use_attention=use_attention)
        self.spade_res3 = SPADEResBlock(ngf*8, ngf*8, semantic_nc, use_attention=use_attention)
        
        # 上采样解码器
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf*8, ngf*4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(ngf*4),
            nn.LeakyReLU(0.2)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf*4, ngf*2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.LeakyReLU(0.2)
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf*2, ngf, kernel_size=3, padding=1),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2)
        )
        
        # 输出层
        self.conv_out = nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x, segmap, z=None):
        # 如果没有提供风格向量，则随机生成
        if z is None:
            z = torch.randn(x.size(0), self.z_dim, device=x.device)
        
        # 获取风格编码
        style = self.style_encoder(z)
        
        # 初始特征提取
        x = self.conv_in(x)
        
        # 下采样编码
        x1 = F.leaky_relu(self.down1(x), 0.2)
        x2 = F.leaky_relu(self.down2(x1), 0.2)
        x3 = F.leaky_relu(self.down3(x2), 0.2)
        
        # 瓶颈层
        x = F.leaky_relu(self.bottleneck(x3), 0.2)
        
        # 应用SPADE残差块
        attention_maps = []
        
        x, att_map1 = self.spade_res1(x, segmap)
        if att_map1 is not None:
            attention_maps.append(att_map1)
        
        x, att_map2 = self.spade_res2(x, segmap)
        if att_map2 is not None:
            attention_maps.append(att_map2)
        
        x, att_map3 = self.spade_res3(x, segmap)
        if att_map3 is not None:
            attention_maps.append(att_map3)
        
        # 上采样解码
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        
        # 输出
        output = self.conv_out(x)
        
        return output, attention_maps


class MultiscaleDiscriminator(nn.Module):
    """
    多尺度判别器，在不同分辨率上操作
    """
    def __init__(self, input_nc, semantic_nc, ndf=64, n_layers=3, num_D=3):
        super(MultiscaleDiscriminator, self).__init__()
        
        self.num_D = num_D
        self.n_layers = n_layers
        
        # 创建多个判别器，每个在不同尺度上操作
        self.discriminators = nn.ModuleList()
        for i in range(num_D):
            self.discriminators.append(
                NLayerDiscriminator(input_nc + semantic_nc, ndf, n_layers)
            )
        
        # 下采样层，用于生成不同尺度的输入
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)
    
    def forward(self, input_img, segmap):
        # 将语义图调整为与输入图像相同的大小
        segmap = F.interpolate(segmap, size=input_img.size()[2:], mode='nearest')
        
        # 拼接输入图像和语义图
        x = torch.cat([input_img, segmap], dim=1)
        
        # 在不同尺度上应用判别器
        result = []
        for i in range(self.num_D):
            # 获取当前尺度的判别器输出
            out = self.discriminators[i](x)
            result.append(out)
            
            # 下采样输入，为下一个尺度准备
            if i != self.num_D - 1:
                x = self.downsample(x)
        
        return result


class NLayerDiscriminator(nn.Module):
    """
    N层PatchGAN判别器
    """
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        
        # 初始卷积层
        sequence = [
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True)
        ]
        
        # 中间层
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                                      kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, True)
            ]
        
        # 最后两层
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                                  kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, True)
        ]
        
        # 输出层
        sequence += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        ]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        # 返回所有中间特征，用于特征匹配损失
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i % 3 == 0 and i > 0:  # 每3层保存一次特征
                features.append(x)
        
        return x, features


class LowLightColorizationModel(nn.Module):
    """
    完整的低光彩色化模型，包括生成器和判别器
    """
    def __init__(self, input_nc=1, output_nc=3, semantic_nc=20, ngf=64, ndf=64, z_dim=8):
        super(LowLightColorizationModel, self).__init__()
        
        # 生成器
        self.generator = SPADEGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            semantic_nc=semantic_nc,
            ngf=ngf,
            z_dim=z_dim,
            use_attention=True
        )
        
        # 多尺度判别器
        self.discriminator = MultiscaleDiscriminator(
            input_nc=output_nc,
            semantic_nc=semantic_nc,
            ndf=ndf,
            n_layers=3,
            num_D=3
        )
    
    def forward(self, mono, semantic, z=None):
        # 生成彩色图像
        fake_color, attention_maps = self.generator(mono, semantic, z)
        
        return fake_color, attention_maps
