import torch
from torchvision import datasets, transforms
import os

# 检查保存路径是否存在，如果不存在则创建
save_dir = "/Users/shen/Documents/Work/Doctor/Course/AIproject2025/ShinyApps/FNN各层输出可视化/www/MNIST_samples"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created directory: {save_dir}")

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 下载MNIST训练集
print("Downloading MNIST dataset...")
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
print("Download complete.")

# 初始化一个字典来存储每类数字已保存的图片数量
counts = {i: 0 for i in range(10)}

# 遍历数据集并保存图片
print("Saving 10 images for each digit...")
for image, label in train_dataset:
    if counts[label] < 10:
        # 将PyTorch张量转换为PIL图像
        # 这里的image是(1, 28, 28)的张量，需要squeeze()去掉通道维度
        pil_image = transforms.ToPILImage()(image.squeeze(0))
        
        # 定义保存路径
        filename = f"{save_dir}/{label}_{counts[label]}.png"
        
        # 保存图片
        pil_image.save(filename)
        counts[label] += 1
        
        # 检查是否已保存完所有需要的图片
        if all(c == 10 for c in counts.values()):
            print("All 10 images for each digit have been saved.")
            break
print("Process finished.")
