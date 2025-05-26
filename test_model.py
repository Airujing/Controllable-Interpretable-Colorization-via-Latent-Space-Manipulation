import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append('/home/ubuntu/lowlight_colorization')
from models import LowLightColorizationModel

def preprocess_image(image_path, img_size=256):
    """预处理输入图像"""
    # 加载图像并转换为灰度
    img = Image.open(image_path).convert('L')
    
    # 调整大小
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 转换为张量
    img_tensor = transform(img).unsqueeze(0)  # 添加批次维度
    return img_tensor

def create_dummy_semantic(batch_size=1, img_size=256, num_classes=20):
    """创建虚拟语义分割图用于测试"""
    # 创建一个简单的语义分割图，将图像分为几个区域
    semantic = np.zeros((batch_size, num_classes, img_size, img_size), dtype=np.float32)
    
    # 背景 (类别0)
    semantic[:, 0, :, :] = 1.0
    
    # 天空 (类别1)
    semantic[:, 0, :img_size//3, :] = 0.0
    semantic[:, 1, :img_size//3, :] = 1.0
    
    # 地面 (类别2)
    semantic[:, 0, 2*img_size//3:, :] = 0.0
    semantic[:, 2, 2*img_size//3:, :] = 1.0
    
    # 中间区域可以是建筑或其他物体 (类别3)
    center_start = img_size//3
    center_end = 2*img_size//3
    center_width = img_size//4
    center_start_x = (img_size - center_width) // 2
    center_end_x = center_start_x + center_width
    
    semantic[:, 0, center_start:center_end, center_start_x:center_end_x] = 0.0
    semantic[:, 3, center_start:center_end, center_start_x:center_end_x] = 1.0
    
    return torch.from_numpy(semantic)

def test_model():
    """测试模型性能"""
    # 创建保存目录
    os.makedirs('/home/ubuntu/lowlight_colorization/test_results', exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = LowLightColorizationModel(
        input_nc=1,
        output_nc=3,
        semantic_nc=20,
        z_dim=8
    )
    
    # 将模型移动到设备
    model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    # 创建测试输入
    test_image_path = '/home/ubuntu/test_image.jpg'  # 这里需要一个测试图像
    
    # 检查测试图像是否存在，如果不存在则创建一个
    if not os.path.exists(test_image_path):
        # 创建一个简单的灰度图像用于测试
        img = Image.new('L', (256, 256), color=128)
        img.save(test_image_path)
        print(f"创建测试图像: {test_image_path}")
    
    # 预处理图像
    mono_tensor = preprocess_image(test_image_path)
    mono_tensor = mono_tensor.to(device)
    
    # 创建虚拟语义分割图
    semantic_tensor = create_dummy_semantic(batch_size=1, img_size=256, num_classes=20)
    semantic_tensor = semantic_tensor.to(device)
    
    # 生成不同风格的彩色图像
    results = []
    attention_maps_list = []
    
    with torch.no_grad():
        for i in range(5):  # 生成5种不同风格
            # 生成随机风格向量
            z = torch.randn(1, 8, device=device)
            
            # 生成彩色图像
            fake_color, attention_maps = model.generator(mono_tensor, semantic_tensor, z)
            results.append(fake_color.cpu())
            
            if attention_maps:
                attention_maps_list.append([am.cpu() for am in attention_maps])
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 显示原始灰度图像
    plt.subplot(2, 3, 1)
    mono_np = mono_tensor[0, 0].cpu().numpy()
    mono_np = (mono_np * 0.5 + 0.5)  # 反归一化
    plt.imshow(mono_np, cmap='gray')
    plt.title('原始低光单色图像')
    plt.axis('off')
    
    # 显示生成的彩色图像
    for i, result in enumerate(results[:4]):  # 只显示前4个结果
        plt.subplot(2, 3, i+2)
        fake_color_np = result[0].numpy().transpose(1, 2, 0)
        fake_color_np = (fake_color_np * 0.5 + 0.5)  # 反归一化
        fake_color_np = np.clip(fake_color_np, 0, 1)
        plt.imshow(fake_color_np)
        plt.title(f'生成风格 {i+1}')
        plt.axis('off')
    
    # 显示语义分割图
    plt.subplot(2, 3, 6)
    semantic_np = semantic_tensor[0].cpu().numpy()
    semantic_rgb = np.zeros((256, 256, 3))
    
    # 为不同语义类别分配不同颜色
    colors = [
        [0, 0, 0],      # 背景 - 黑色
        [0, 0, 1],      # 天空 - 蓝色
        [0, 0.5, 0],    # 地面 - 绿色
        [0.7, 0.3, 0.3]  # 建筑 - 棕色
    ]
    
    for i in range(4):  # 只使用前4个类别
        mask = semantic_np[i] > 0.5
        for c in range(3):
            semantic_rgb[:, :, c] += mask * colors[i][c]
    
    plt.imshow(semantic_rgb)
    plt.title('语义分割图')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/lowlight_colorization/test_results/generated_samples.png')
    print("生成的样本已保存到 /home/ubuntu/lowlight_colorization/test_results/generated_samples.png")
    
    # 可视化注意力图
    if attention_maps_list:
        plt.figure(figsize=(15, 5))
        for i, att_maps in enumerate(attention_maps_list[0]):  # 使用第一个样本的注意力图
            if i >= 3:  # 只显示前3个注意力图
                break
            
            plt.subplot(1, 3, i+1)
            att_map = att_maps[0, 0].numpy()  # 取第一个批次，第一个通道
            plt.imshow(att_map, cmap='jet')
            plt.title(f'注意力图 {i+1}')
            plt.colorbar()
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/lowlight_colorization/test_results/attention_maps.png')
        print("注意力图已保存到 /home/ubuntu/lowlight_colorization/test_results/attention_maps.png")
    
    print("模型测试完成")

if __name__ == "__main__":
    test_model()
