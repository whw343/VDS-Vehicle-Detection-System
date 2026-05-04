"""
车辆颜色分类模型训练脚本

数据：8种颜色仿真图片（各200张）
模型：自定义CNN（来自 color_classify.py）
输出：weights/color_model.pth

负责人：肖岱彤
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import random

sys.path.insert(0, os.path.dirname(__file__))
from color_classify import ColorClassifier, COLOR_LABELS

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.0005
SAMPLES_PER_CLASS = 300
NUM_CLASSES = 8
IMG_SIZE = 224

OUTPUT_PATH = 'weights/color_model.pth'
os.makedirs('weights', exist_ok=True)

# 8种颜色RGB定义
COLOR_RGB = {
    0: (30, 30, 30),     # black
    1: (30, 80, 180),    # blue
    2: (120, 60, 30),    # brown
    3: (40, 150, 60),    # green
    4: (200, 40, 40),    # red
    5: (180, 180, 190),  # silver
    6: (230, 230, 240),  # white
    7: (200, 200, 40),   # yellow
}


def generate_synthetic_image(label_id, img_size=IMG_SIZE):
    """
    生成仿真的车辆颜色图片，模拟真实光照和噪声
    """
    base_r, base_g, base_b = COLOR_RGB[label_id]

    # 较大范围的颜色变化（模拟不同光照条件）
    scale = random.uniform(0.7, 1.3)
    r = np.clip(int(base_r * scale) + random.randint(-15, 15), 5, 250)
    g = np.clip(int(base_g * scale) + random.randint(-15, 15), 5, 250)
    b = np.clip(int(base_b * scale) + random.randint(-15, 15), 5, 250)

    # 创建底色（带渐变）
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    gradient = np.linspace(0.8, 1.2, img_size).reshape(-1, 1)
    for c in range(3):
        val = [r, g, b][c]
        img[:, :, c] = np.clip(val * gradient * gradient.T, 0, 255)

    # 较强的高斯噪声
    noise = np.random.normal(0, 10, (img_size, img_size, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # 模拟多个反光区域
    for _ in range(random.randint(0, 3)):
        rx = random.randint(0, img_size - 30)
        ry = random.randint(0, img_size - 20)
        rw = random.randint(10, 50)
        rh = random.randint(5, 25)
        alpha = random.uniform(0.1, 0.4)
        roi = img[ry:ry+rh, rx:rx+rw].astype(float)
        img[ry:ry+rh, rx:rx+rw] = np.clip(roi * (1-alpha) + 255*alpha, 0, 255).astype(np.uint8)

    # 模拟暗角
    if random.random() > 0.5:
        xx, yy = np.meshgrid(np.linspace(0.7, 1.0, img_size), np.linspace(0.7, 1.0, img_size))
        vignette = np.stack([xx*yy]*3, axis=-1)
        img = np.clip(img * vignette, 0, 255).astype(np.uint8)

    return img


def generate_dataset():
    """生成完整训练+验证数据集"""
    print("[数据] 生成仿真颜色数据集...")

    X_train, y_train = [], []
    X_val, y_val = [], []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    for label_id in range(NUM_CLASSES):
        color_name = COLOR_LABELS[label_id]
        train_count = 0
        val_count = 0

        for i in range(SAMPLES_PER_CLASS):
            img = generate_synthetic_image(label_id)
            pil_img = Image.fromarray(img)
            tensor = transform(pil_img)

            # 80% 训练 / 20% 验证
            if i < SAMPLES_PER_CLASS * 0.8:
                X_train.append(tensor)
                y_train.append(label_id)
                train_count += 1
            else:
                X_val.append(tensor)
                y_val.append(label_id)
                val_count += 1

        print(f"  {color_name:8s}: train={train_count}, val={val_count}")

    X_train = torch.stack(X_train)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.stack(X_val)
    y_val = torch.tensor(y_val, dtype=torch.long)

    print(f"\n[数据] 总计: train={len(X_train)}, val={len(X_val)}")
    return X_train, y_train, X_val, y_val


def train():
    """训练颜色分类模型"""
    print(f"[训练] 设备: {DEVICE}")
    print(f"[训练] Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    # 生成数据
    X_train, y_train, X_val, y_val = generate_dataset()

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 模型
    model = ColorClassifier(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    best_val_acc = 0.0

    print("\n[训练] 开始训练...")
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_PATH)

        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Acc: {val_acc:.2%} {'*' if val_acc == best_val_acc else ''}")

    print(f"\n[训练] 完成! 最佳验证准确率: {best_val_acc:.2%}")
    print(f"[训练] 模型已保存: {OUTPUT_PATH}")
    return best_val_acc


def test_model():
    """测试训练好的模型"""
    from color_classify import VehicleColorClassifier
    import cv2

    classifier = VehicleColorClassifier(model_path=OUTPUT_PATH)
    if not classifier.load_model():
        print("[测试] 模型加载失败")
        return

    print("\n[测试] 模型测试...")
    correct = 0
    total = 0

    for label_id in range(NUM_CLASSES):
        color_name = COLOR_LABELS[label_id]
        class_correct = 0

        for _ in range(20):
            # 生成 RGB 图片，转 BGR（匹配 classify() 的预期输入）
            img_rgb = generate_synthetic_image(label_id)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            result = classifier.classify(img_bgr)
            if result['color'] == color_name:
                class_correct += 1
                correct += 1
            total += 1

        acc = class_correct / 20
        print(f"  {color_name:8s}: {class_correct}/20 ({acc:.0%})")

    print(f"\n[测试] 总准确率: {correct}/{total} ({correct/total:.1%})")


if __name__ == '__main__':
    # 训练
    acc = train()

    # 测试
    test_model()

    print(f"\n✅ 颜色分类模型训练完成!")
    print(f"   准确率: {acc:.2%}")
    print(f"   模型文件: {OUTPUT_PATH}")
