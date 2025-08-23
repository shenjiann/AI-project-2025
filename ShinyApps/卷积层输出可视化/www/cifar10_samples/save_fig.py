import os
import random
from torchvision import datasets, transforms
from PIL import Image

# 固定随机种子，保证每次运行结果一致
random.seed(42)

# 下载并加载 CIFAR10 训练集
transform = transforms.ToTensor()
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# CIFAR10 的类别
classes = dataset.classes  # ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# 保存路径
save_root = "./Shiny应用/CIFAR/www/cifar10_samples"
os.makedirs(save_root, exist_ok=True)

# 每类选 10 张
num_per_class = 10
selected_indices = {cls: [] for cls in classes}

# 随机打乱索引（因为固定了seed，每次顺序一样）
indices = list(range(len(dataset)))
random.shuffle(indices)

# 遍历数据，按类别保存
for idx in indices:
    img, label = dataset[idx]
    class_name = classes[label]
    if len(selected_indices[class_name]) < num_per_class:
        # 转换为PIL保存
        pil_img = transforms.ToPILImage()(img)
        save_dir = os.path.join(save_root, class_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{class_name}_{len(selected_indices[class_name])}.png")
        pil_img.save(save_path)
        selected_indices[class_name].append(idx)

    # 如果所有类别都保存够了就退出
    if all(len(v) >= num_per_class for v in selected_indices.values()):
        break

print(f"已保存到 {save_root}/ 下，每类 {num_per_class} 张图片（固定种子）")