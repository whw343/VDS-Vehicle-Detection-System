"""
车辆类型分类模型训练脚本

数据：4种车型仿真图片（各300张，含车辆轮廓+颜色+背景）
模型：ResNet18（来自 type_classify.py）
输出：weights/type_model.pth

负责人：宋柄儒
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image, ImageDraw
import random

sys.path.insert(0, os.path.dirname(__file__))
from type_classify import TypeClassifier, TYPE_LABELS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.0005
SAMPLES_PER_CLASS = 300
NUM_CLASSES = 4
IMG_SIZE = 224

OUTPUT_PATH = 'weights/type_model.pth'
os.makedirs('weights', exist_ok=True)


def draw_car(draw, w, h):
    """绘制轿车形状（中等尺寸，流线型）"""
    body_top = h * 0.35
    body_bot = h * 0.75
    body_left = w * 0.08
    body_right = w * 0.92

    # 车身主体
    draw.rectangle([body_left, body_top, body_right, body_bot], fill=None)
    # 车顶（梯形，模拟轿车流线）
    roof_top = h * 0.12
    roof_left = w * 0.25
    roof_right = w * 0.75
    draw.polygon([
        (roof_left, body_top), (roof_right, body_top),
        (roof_right - w*0.05, roof_top), (roof_left + w*0.05, roof_top)
    ], fill=None)
    draw.rectangle([roof_left, roof_top, roof_right, body_top], fill=None)
    # 车窗
    win_top = roof_top + 3
    win_bot = body_top - 3
    draw.rectangle([roof_left + 5, win_top, roof_right - 5, win_bot],
                   fill=(180, 210, 240))

    # 车轮
    for _ in range(2):
        pass


def generate_vehicle_image(label_id, img_size=IMG_SIZE):
    """
    生成仿真车辆图片

    通过形状+颜色+细节区分4种车型：
    - car: 流线型轿车，接近方形比例
    - bus: 长条形，多车窗
    - truck: 高车身+货箱区
    - mini: 小型紧凑
    """
    # 背景色（模拟道路环境）
    bg_colors = [
        (180, 200, 220),  # 天空
        (140, 160, 180),  # 阴天
        (200, 210, 200),  # 城市
        (220, 200, 180),  # 黄昏
    ]
    bg = random.choice(bg_colors)
    img = Image.new('RGB', (img_size, img_size), bg)
    draw = ImageDraw.Draw(img)

    # 地面
    ground_y = int(img_size * 0.8)
    ground_colors = [(120, 120, 130), (80, 80, 90), (100, 110, 120)]
    gc = random.choice(ground_colors)
    draw.rectangle([0, ground_y, img_size, img_size], fill=gc)

    # 车辆主色调
    vehicle_colors = [
        (220, 40, 40), (40, 80, 200), (40, 40, 40),
        (230, 230, 240), (180, 180, 190), (200, 200, 40),
        (120, 60, 30), (40, 150, 60),
    ]
    vc = random.choice(vehicle_colors)

    if label_id == 0:  # car 轿车
        _draw_sedan(draw, vc, img_size, ground_y)
    elif label_id == 1:  # bus 客车
        _draw_bus(draw, vc, img_size, ground_y)
    elif label_id == 2:  # truck 卡车
        _draw_truck(draw, vc, img_size, ground_y)
    elif label_id == 3:  # mini 微型车
        _draw_mini(draw, vc, img_size, ground_y)

    # 添加噪声
    img_arr = np.array(img)
    noise = np.random.normal(0, 5, img_arr.shape).astype(np.int16)
    img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img_arr


def _draw_sedan(draw, vc, s, gy):
    """轿车：流线型，中等尺寸"""
    body_l = int(s * 0.10)
    body_r = int(s * 0.90)
    body_t = int(gy - s * 0.35)
    body_b = int(gy - s * 0.02)

    # 车身
    draw.rectangle([body_l, body_t, body_r, body_b], fill=vc)

    # 车顶流线
    roof_l = int(s * 0.28)
    roof_r = int(s * 0.72)
    roof_t = int(body_t - s * 0.15)
    draw.polygon([
        (roof_l, body_t), (roof_r, body_t),
        (roof_r - 15, roof_t), (roof_l + 15, roof_t)
    ], fill=vc)
    draw.rectangle([roof_l, roof_t, roof_r, body_t], fill=vc)

    # 车窗
    win_c = (180 + random.randint(-20, 20), 200 + random.randint(-20, 20),
             230 + random.randint(-20, 20))
    front_w = int(s * 0.20)
    mid_w = int(s * 0.18)
    draw.rectangle([roof_l + 8, roof_t + 6, roof_l + front_w, body_t - 2], fill=win_c)
    draw.rectangle([roof_l + front_w + 4, roof_t + 6, roof_l + front_w + mid_w + 4, body_t - 2], fill=win_c)

    # 车轮
    _draw_wheel(draw, int(body_l + s * 0.18), body_b, 18)
    _draw_wheel(draw, int(body_r - s * 0.22), body_b, 18)


def _draw_bus(draw, vc, s, gy):
    """客车：长条形，多车窗"""
    body_l = int(s * 0.03)
    body_r = int(s * 0.97)
    body_t = int(gy - s * 0.48)
    body_b = int(gy - s * 0.02)

    # 车身
    draw.rectangle([body_l, body_t, body_r, body_b], fill=vc)

    # 一排车窗
    win_c = (180, 200, 230)
    win_w = int(s * 0.08)
    win_h = int(s * 0.12)
    win_t = body_t + int(s * 0.05)
    for i in range(8):
        wx = body_l + int(s * 0.05) + i * (win_w + 5)
        if wx + win_w < body_r - 10:
            draw.rectangle([wx, win_t, wx + win_w, win_t + win_h], fill=win_c)

    # 前窗
    draw.rectangle([body_l + 5, win_t, body_l + win_w + 10, win_t + win_h + 10], fill=win_c)

    # 车轮
    _draw_wheel(draw, int(body_l + s * 0.10), body_b, 16)
    _draw_wheel(draw, int(body_r - s * 0.12), body_b, 16)


def _draw_truck(draw, vc, s, gy):
    """卡车：高车身+货箱"""
    # 驾驶室（前部，较高）
    cab_l = int(s * 0.05)
    cab_r = int(s * 0.38)
    cab_t = int(gy - s * 0.50)
    cab_b = int(gy - s * 0.02)

    cab_c = (vc[0], vc[1], vc[2])
    draw.rectangle([cab_l, cab_t, cab_r, cab_b], fill=cab_c)

    # 车窗
    win_c = (180, 200, 230)
    draw.rectangle([cab_l + 8, cab_t + 8, cab_r - 4, cab_t + int(s*0.12)], fill=win_c)

    # 货箱（后部，稍低）
    cargo_l = cab_r + 2
    cargo_r = int(s * 0.95)
    cargo_t = int(gy - s * 0.42)
    cargo_b = cab_b

    cargo_c = (
        min(vc[0] + 30, 250),
        min(vc[1] + 30, 250),
        min(vc[2] + 30, 250)
    )
    draw.rectangle([cargo_l, cargo_t, cargo_r, cargo_b], fill=cargo_c)

    # 货箱线条
    draw.line([(cargo_l, cargo_t), (cargo_r, cargo_t)], fill=(0, 0, 0), width=2)

    # 车轮
    _draw_wheel(draw, int(cab_l + s * 0.08), cab_b, 17)
    _draw_wheel(draw, int(cargo_r - s * 0.15), cargo_b, 17)


def _draw_mini(draw, vc, s, gy):
    """微型车：小型紧凑"""
    body_l = int(s * 0.18)
    body_r = int(s * 0.78)
    body_t = int(gy - s * 0.32)
    body_b = int(gy - s * 0.02)

    # 车身
    draw.rectangle([body_l, body_t, body_r, body_b], fill=vc)

    # 小顶棚
    roof_l = int(s * 0.32)
    roof_r = int(s * 0.64)
    roof_t = int(body_t - s * 0.10)
    draw.polygon([
        (roof_l, body_t), (roof_r, body_t),
        (roof_r - 8, roof_t), (roof_l + 8, roof_t)
    ], fill=vc)
    draw.rectangle([roof_l, roof_t, roof_r, body_t], fill=vc)

    # 小车窗
    win_c = (180, 200, 230)
    draw.rectangle([roof_l + 6, roof_t + 5, roof_r - 6, body_t - 2], fill=win_c)

    # 小车轮
    _draw_wheel(draw, int(body_l + s * 0.06), body_b, 12)
    _draw_wheel(draw, int(body_r - s * 0.10), body_b, 12)


def _draw_wheel(draw, cx, by, radius):
    """绘制车轮"""
    draw.ellipse([
        cx - radius, by - radius * 2,
        cx + radius, by
    ], fill=(30, 30, 30))
    draw.ellipse([
        cx - radius // 2, by - radius,
        cx + radius // 2, by - radius // 2
    ], fill=(100, 100, 100))


def generate_dataset():
    """生成训练+验证数据"""
    print("[数据] 生成仿真车型数据集...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    X_train, y_train = [], []
    X_val, y_val = [], []

    for label_id in range(NUM_CLASSES):
        type_name = TYPE_LABELS[label_id]
        tc = 0
        vc = 0

        for i in range(SAMPLES_PER_CLASS):
            img = generate_vehicle_image(label_id)
            pil_img = Image.fromarray(img)
            tensor = transform(pil_img)

            if i < int(SAMPLES_PER_CLASS * 0.8):
                X_train.append(tensor)
                y_train.append(label_id)
                tc += 1
            else:
                X_val.append(tensor)
                y_val.append(label_id)
                vc += 1

        print(f"  {type_name:6s}: train={tc}, val={vc}")

    X_train = torch.stack(X_train)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.stack(X_val)
    y_val = torch.tensor(y_val, dtype=torch.long)

    print(f"\n[数据] 总计: train={len(X_train)}, val={len(X_val)}")
    return X_train, y_train, X_val, y_val


def train():
    """训练车型分类模型"""
    print(f"[训练] 设备: {DEVICE}")
    print(f"[训练] 类别: {list(TYPE_LABELS.values())}")

    X_train, y_train, X_val, y_val = generate_dataset()

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=BATCH_SIZE)

    model = TypeClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    best_val_acc = 0.0

    print(f"\n[训练] 开始...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += outputs.argmax(1).eq(labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                val_correct += outputs.argmax(1).eq(labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_PATH)

        print(f"  Epoch {epoch+1:2d} | "
              f"Train: loss={train_loss/len(train_loader):.4f} acc={train_acc:.2%} | "
              f"Val: loss={val_loss/len(val_loader):.4f} acc={val_acc:.2%}"
              f"{' *' if val_acc >= best_val_acc else ''}")

    print(f"\n[训练] 完成! 最佳验证: {best_val_acc:.2%}")
    return best_val_acc


def test_model():
    """测试模型"""
    from type_classify import VehicleTypeClassifier

    classifier = VehicleTypeClassifier(model_path=OUTPUT_PATH)
    if not classifier.load_model():
        return

    print("\n[测试] 车型分类测试...")
    correct = 0
    total = 0

    for label_id in range(NUM_CLASSES):
        type_name = TYPE_LABELS[label_id]
        class_correct = 0

        for _ in range(25):
            img = generate_vehicle_image(label_id)
            result = classifier.classify(img)
            if result['type'] == type_name:
                class_correct += 1
                correct += 1
            total += 1

        print(f"  {type_name:6s}: {class_correct}/25 ({class_correct/25:.0%})")

    print(f"\n[测试] 总准确率: {correct}/{total} ({correct/total:.1%})")


if __name__ == '__main__':
    acc = train()
    test_model()
    print(f"\n✅ 车型分类模型训练完成! 准确率: {acc:.2%}")
