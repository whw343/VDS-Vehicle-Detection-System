"""
车辆类型分类模块
从车辆图片中识别车辆类型（轿车/客车/卡车/微型车）

输入：车辆图片路径或numpy数组
输出：类型标签 + 置信度

负责人：宋柄儒
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# 车辆类型标签
TYPE_LABELS = {
    0: 'car',      # 轿车
    1: 'bus',      # 客车
    2: 'truck',    # 卡车
    3: 'mini',     # 微型车
}

TYPE_CHINESE = {
    'car': '轿车',
    'bus': '客车',
    'truck': '卡车',
    'mini': '微型车',
}


class TypeClassifier(nn.Module):
    """车辆类型分类模型（ResNet18迁移学习）"""

    def __init__(self, num_classes=4):
        super(TypeClassifier, self).__init__()
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


class VehicleTypeClassifier:
    """车辆类型识别器"""

    def __init__(self, model_path='weights/type_model.pth', num_classes=4):
        self.model_path = model_path
        self.num_classes = num_classes
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        """加载训练好的类型分类模型"""
        if not os.path.exists(self.model_path):
            print(f"[类型识别] 模型文件不存在: {self.model_path}")
            print("[类型识别] 请先训练模型: python train_type_model.py")
            return False

        try:
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            if not any(k.startswith('backbone.') for k in state_dict.keys()):
                state_dict = {'backbone.' + k: v for k, v in state_dict.items()}
            self.model = TypeClassifier(num_classes=self.num_classes)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"[类型识别] 模型加载成功: {self.model_path}")
            return True
        except Exception as e:
            print(f"[类型识别] 模型加载失败: {e}")
            return False

    def preprocess(self, image):
        """预处理图片"""
        if isinstance(image, str):
            if not os.path.exists(image):
                return None
            image = cv2.imread(image)
            if image is None:
                return None

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def classify(self, image):
        """
        识别车辆类型

        Args:
            image: 图片路径或numpy数组（BGR格式）

        Returns:
            dict: {'type': 'car', 'type_cn': '轿车', 'confidence': 0.85}
        """
        if self.model is None:
            return {'type': 'unknown', 'type_cn': '未知', 'confidence': 0.0}

        tensor = self.preprocess(image)
        if tensor is None:
            return {'type': 'unknown', 'type_cn': '未知', 'confidence': 0.0}

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)

            top2_prob, top2_idx = torch.topk(probabilities, 2, dim=1)

            type_id = top2_idx[0][0].item()
            conf = top2_prob[0][0].item()

            type_name = TYPE_LABELS.get(type_id, 'unknown')
            type_cn = TYPE_CHINESE.get(type_name, '未知')

            return {
                'type': type_name,
                'type_cn': type_cn,
                'confidence': conf,
                'top2': [
                    {'type': TYPE_LABELS.get(top2_idx[0][i].item(), ''),
                     'confidence': top2_prob[0][i].item()}
                    for i in range(2)
                ]
            }


def classify_type(image_path):
    """便捷函数：识别车辆类型"""
    classifier = VehicleTypeClassifier()
    if classifier.load_model():
        result = classifier.classify(image_path)
        return result['type_cn']
    return "未知"


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'static/uploads/test.jpg'

    print(f"[测试] 识别类型: {image_path}")
    result = classify_type(image_path)
    print(f"[测试] 识别结果: {result}")
