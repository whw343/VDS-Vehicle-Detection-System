"""
车辆颜色识别模块
从车辆图片中识别车身颜色

输入：车辆图片路径或numpy数组
输出：颜色类别标签

负责人：肖岱彤
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# 颜色标签映射
COLOR_LABELS = {
    0: 'black',    # 黑色
    1: 'blue',     # 蓝色
    2: 'brown',    # 棕色
    3: 'green',    # 绿色
    4: 'red',      # 红色
    5: 'silver',   # 银色/灰色
    6: 'white',    # 白色
    7: 'yellow',   # 黄色
}

# 中文颜色名
COLOR_CHINESE = {
    'black': '黑色',
    'blue': '蓝色',
    'brown': '棕色',
    'green': '绿色',
    'red': '红色',
    'silver': '银色',
    'white': '白色',
    'yellow': '黄色',
}


class ColorClassifier(nn.Module):
    """车辆颜色分类模型（ResNet18迁移学习）"""

    def __init__(self, num_classes=8):
        super(ColorClassifier, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class VehicleColorClassifier:
    """车辆颜色识别器"""

    def __init__(self, model_path='weights/color_model.pth', num_classes=8):
        self.model_path = model_path
        self.num_classes = num_classes
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 图片预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        """加载训练好的颜色分类模型"""
        if not os.path.exists(self.model_path):
            print(f"[颜色识别] 模型文件不存在: {self.model_path}")
            print("[颜色识别] 请先训练模型或使用预训练权重")
            return False

        try:
            self.model = ColorClassifier(num_classes=self.num_classes)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"[颜色识别] 模型加载成功: {self.model_path}")
            return True
        except Exception as e:
            print(f"[颜色识别] 模型加载失败: {e}")
            return False

    def preprocess(self, image):
        """预处理图片"""
        if isinstance(image, str):
            if not os.path.exists(image):
                return None
            image = cv2.imread(image)
            if image is None:
                return None

        # BGR转RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 转PIL Image
        image = Image.fromarray(image)

        # 应用变换
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def classify(self, image):
        """
        识别车辆颜色

        Args:
            image: 图片路径或numpy数组

        Returns:
            dict: {'color': 'red', 'color_cn': '红色', 'confidence': 0.85}
        """
        if self.model is None:
            return {'color': 'unknown', 'color_cn': '未知', 'confidence': 0.0}

        tensor = self.preprocess(image)
        if tensor is None:
            return {'color': 'unknown', 'color_cn': '未知', 'confidence': 0.0}

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            color_id = predicted.item()
            color_name = COLOR_LABELS.get(color_id, 'unknown')
            color_cn = COLOR_CHINESE.get(color_name, '未知')

            return {
                'color': color_name,
                'color_cn': color_cn,
                'confidence': confidence.item()
            }


def classify_color(image_path):
    """
    便捷函数：识别车辆颜色

    Args:
        image_path: 图片路径

    Returns:
        str: 颜色标签（中文）
    """
    classifier = VehicleColorClassifier()
    if classifier.load_model():
        result = classifier.classify(image_path)
        return result['color_cn']
    return "未知"


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'static/uploads/test.jpg'

    print(f"[测试] 识别颜色: {image_path}")
    color = classify_color(image_path)
    print(f"[测试] 识别结果: {color}")
