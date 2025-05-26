import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class LowLightColorDataset(Dataset):
    """
    数据集类，用于加载低光单色图像、彩色真值和语义分割图
    """
    def __init__(self, root_dir, mode='train', img_size=256, augment=True, num_semantic_classes=20):
        """
        初始化数据集
        
        参数:
            root_dir (str): 数据集根目录，应包含三个子目录：
                           - monochrome: 低光单色图像
                           - color: 彩色真值图像
                           - semantic: 语义分割图
            mode (str): 'train' 或 'test'
            img_size (int): 图像大小
            augment (bool): 是否进行数据增强
            num_semantic_classes (int): 语义分割类别数量
        """
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.augment = augment and mode == 'train'
        self.num_semantic_classes = num_semantic_classes
        
        # 图像路径
        self.mono_dir = os.path.join(root_dir, 'monochrome')
        self.color_dir = os.path.join(root_dir, 'color')
        self.semantic_dir = os.path.join(root_dir, 'semantic')
        
        # 获取图像文件名列表
        self.image_files = [f for f in os.listdir(self.mono_dir) 
                           if os.path.isfile(os.path.join(self.mono_dir, f)) and 
                           os.path.isfile(os.path.join(self.color_dir, f)) and
                           os.path.isfile(os.path.join(self.semantic_dir, f.replace('.jpg', '.png').replace('.jpeg', '.png')))]
        
        # 数据转换
        self.transform_common = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        self.transform_mono = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(1),  # 确保是单通道
            transforms.ToTensor(),
        ])
        
        self.normalize = transforms.Normalize((0.5,), (0.5,))
        self.normalize_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # 加载低光单色图像
        mono_path = os.path.join(self.mono_dir, img_name)
        mono_img = Image.open(mono_path).convert('L')  # 确保是灰度图
        
        # 加载彩色真值图像
        color_path = os.path.join(self.color_dir, img_name)
        color_img = Image.open(color_path).convert('RGB')
        
        # 加载语义分割图
        semantic_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        semantic_path = os.path.join(self.semantic_dir, semantic_name)
        semantic_img = Image.open(semantic_path).convert('L')  # 语义图是单通道的
        
        # 数据增强
        if self.augment:
            # 随机裁剪
            i, j, h, w = transforms.RandomCrop.get_params(mono_img, output_size=(self.img_size, self.img_size))
            mono_img = transforms.functional.crop(mono_img, i, j, h, w)
            color_img = transforms.functional.crop(color_img, i, j, h, w)
            semantic_img = transforms.functional.crop(semantic_img, i, j, h, w)
            
            # 随机水平翻转
            if random.random() > 0.5:
                mono_img = transforms.functional.hflip(mono_img)
                color_img = transforms.functional.hflip(color_img)
                semantic_img = transforms.functional.hflip(semantic_img)
        
        # 转换为张量
        mono_tensor = self.transform_mono(mono_img)
        color_tensor = self.transform_common(color_img)
        semantic_tensor = self.transform_common(semantic_img)
        
        # 归一化
        mono_tensor = self.normalize(mono_tensor)
        color_tensor = self.normalize_rgb(color_tensor)
        
        # 将语义图转换为one-hot编码
        semantic_np = np.array(semantic_tensor[0] * (self.num_semantic_classes - 1), dtype=np.uint8)
        semantic_one_hot = np.zeros((self.num_semantic_classes, self.img_size, self.img_size), dtype=np.float32)
        for c in range(self.num_semantic_classes):
            semantic_one_hot[c][semantic_np == c] = 1.0
        
        semantic_tensor = torch.from_numpy(semantic_one_hot)
        
        return {
            'mono': mono_tensor,
            'color': color_tensor,
            'semantic': semantic_tensor,
            'path': img_name
        }


def get_dataloader(root_dir, batch_size=8, num_workers=4, img_size=256, num_semantic_classes=20):
    """
    创建训练和测试数据加载器
    
    参数:
        root_dir (str): 数据集根目录
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
        img_size (int): 图像大小
        num_semantic_classes (int): 语义分割类别数量
    
    返回:
        train_loader, test_loader: 训练和测试数据加载器
    """
    train_dataset = LowLightColorDataset(
        root_dir=root_dir,
        mode='train',
        img_size=img_size,
        augment=True,
        num_semantic_classes=num_semantic_classes
    )
    
    test_dataset = LowLightColorDataset(
        root_dir=root_dir,
        mode='test',
        img_size=img_size,
        augment=False,
        num_semantic_classes=num_semantic_classes
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


# 数据集预处理工具函数
def create_synthetic_lowlight_dataset(source_dir, target_dir, semantic_dir=None, gamma_range=(1.5, 3.5), noise_level=(0.05, 0.15)):
    """
    从正常光照图像创建合成的低光数据集
    
    参数:
        source_dir (str): 源图像目录（正常光照彩色图像）
        target_dir (str): 目标目录，将包含三个子目录：monochrome, color, semantic
        semantic_dir (str): 语义分割图目录，如果为None则不处理语义图
        gamma_range (tuple): gamma校正范围，用于降低亮度
        noise_level (tuple): 噪声水平范围
    """
    import cv2
    import shutil
    
    # 创建目标目录结构
    os.makedirs(os.path.join(target_dir, 'monochrome'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'color'), exist_ok=True)
    if semantic_dir:
        os.makedirs(os.path.join(target_dir, 'semantic'), exist_ok=True)
    
    # 获取源图像列表
    image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        # 读取原始彩色图像
        img_path = os.path.join(source_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
        
        # 转换为RGB（OpenCV默认是BGR）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 保存原始彩色图像作为真值
        color_path = os.path.join(target_dir, 'color', img_file)
        cv2.imwrite(color_path, img)
        
        # 创建低光图像
        gamma = np.random.uniform(gamma_range[0], gamma_range[1])
        noise_sigma = np.random.uniform(noise_level[0], noise_level[1])
        
        # Gamma校正降低亮度
        lowlight = np.power(img_rgb / 255.0, gamma) * 255.0
        lowlight = lowlight.astype(np.uint8)
        
        # 添加高斯噪声
        noise = np.random.normal(0, noise_sigma * 255, lowlight.shape).astype(np.int16)
        lowlight = np.clip(lowlight.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 转换为灰度图
        lowlight_gray = cv2.cvtColor(lowlight, cv2.COLOR_RGB2GRAY)
        
        # 保存低光单色图像
        mono_path = os.path.join(target_dir, 'monochrome', img_file)
        cv2.imwrite(mono_path, lowlight_gray)
        
        # 处理语义分割图（如果提供）
        if semantic_dir:
            semantic_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
            semantic_path = os.path.join(semantic_dir, semantic_file)
            if os.path.exists(semantic_path):
                # 复制语义图到目标目录
                shutil.copy(semantic_path, os.path.join(target_dir, 'semantic', semantic_file))
            else:
                print(f"找不到语义图: {semantic_path}")
    
    print(f"已创建合成低光数据集，共处理 {len(image_files)} 张图像")
