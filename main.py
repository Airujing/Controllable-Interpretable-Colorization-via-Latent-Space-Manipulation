import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

from dataset import LowLightColorDataset, get_dataloader, create_synthetic_lowlight_dataset
from models import LowLightColorizationModel
from trainer import LowLightColorizationTrainer, visualize_attention, generate_with_different_styles

def main():
    parser = argparse.ArgumentParser(description='低光彩色化模型训练与测试')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'demo'],
                        help='运行模式: train, test, demo')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据集根目录')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='结果保存目录')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='学习率')
    parser.add_argument('--img_size', type=int, default=256,
                        help='图像大小')
    parser.add_argument('--num_semantic_classes', type=int, default=20,
                        help='语义分割类别数量')
    parser.add_argument('--z_dim', type=int, default=8,
                        help='风格向量维度')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='加载检查点路径')
    parser.add_argument('--create_dataset', action='store_true',
                        help='是否创建合成低光数据集')
    parser.add_argument('--source_dir', type=str, default=None,
                        help='合成数据集的源图像目录')
    parser.add_argument('--semantic_dir', type=str, default=None,
                        help='合成数据集的语义分割图目录')
    
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建合成数据集（如果需要）
    if args.create_dataset and args.source_dir:
        print("创建合成低光数据集...")
        create_synthetic_lowlight_dataset(
            source_dir=args.source_dir,
            target_dir=args.data_dir,
            semantic_dir=args.semantic_dir,
            gamma_range=(1.5, 3.5),
            noise_level=(0.05, 0.15)
        )
    
    # 获取数据加载器
    train_loader, test_loader = get_dataloader(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_semantic_classes=args.num_semantic_classes
    )
    
    # 创建模型
    model = LowLightColorizationModel(
        input_nc=1,
        output_nc=3,
        semantic_nc=args.num_semantic_classes,
        z_dim=args.z_dim
    )
    
    # 创建训练器
    trainer = LowLightColorizationTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        lr=args.lr,
        z_dim=args.z_dim
    )
    
    # 加载检查点（如果有）
    start_epoch = 0
    if args.checkpoint:
        print(f"加载检查点: {args.checkpoint}")
        start_epoch = trainer.load_model(args.checkpoint)
        print(f"从epoch {start_epoch}继续训练")
    
    # 根据模式执行操作
    if args.mode == 'train':
        print("开始训练...")
        trainer.train(
            num_epochs=args.epochs,
            save_dir=args.save_dir,
            save_interval=5,
            sample_interval=1
        )
    
    elif args.mode == 'test':
        print("开始测试...")
        val_losses = trainer.validate()
        print(f"测试结果: L1_loss: {val_losses['L1_loss']:.4f}, VGG_loss: {val_losses['VGG_loss']:.4f}")
        
        # 生成测试样本
        os.makedirs(os.path.join(args.save_dir, 'test_samples'), exist_ok=True)
        trainer.save_samples('test', args.save_dir)
    
    elif args.mode == 'demo':
        print("运行演示...")
        model.eval()
        
        # 从测试集获取一些样本
        os.makedirs(os.path.join(args.save_dir, 'demo'), exist_ok=True)
        
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                if i >= 5:  # 只处理少量样本
                    break
                
                mono = data['mono'].to(device)
                real_color = data['color'].to(device)
                semantic = data['semantic'].to(device)
                
                # 生成不同风格的彩色图像
                fake_colors = generate_with_different_styles(
                    model=model,
                    mono=mono,
                    semantic=semantic,
                    num_styles=5,
                    device=device
                )
                
                # 保存结果
                for j, fake_color in enumerate(fake_colors):
                    # 创建网格图像
                    img_grid = make_grid(torch.cat([
                        mono[0:1].repeat(1, 3, 1, 1),  # 将单通道转为三通道以便显示
                        fake_color,
                        real_color[0:1]
                    ], dim=0), nrow=3)
                    
                    # 保存图像
                    save_image(img_grid, os.path.join(args.save_dir, 'demo', f'sample_{i}_style_{j}.png'), normalize=True)
                
                # 生成并保存注意力图
                _, attention_maps = model.generator(mono[0:1], semantic[0:1])
                if attention_maps:
                    for j, att_map in enumerate(attention_maps):
                        visualize_attention(
                            attention_map=att_map[0],
                            save_path=os.path.join(args.save_dir, 'demo', f'sample_{i}_attention_{j}.png')
                        )

if __name__ == '__main__':
    main()
