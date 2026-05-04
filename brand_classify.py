"""
车辆品牌识别模块
从车辆图片中识别车辆品牌/型号

输入：车辆图片路径或numpy数组
输出：品牌分类标签

负责人：宋柄儒
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# 品牌标签（从训练数据XML标注中提取，或从weights/brand_labels.txt加载）
def _load_brand_labels(label_file='weights/brand_labels.txt'):
    """从品牌标签文件加载"""
    labels = []
    if os.path.exists(label_file):
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
    return labels

BRAND_LABELS = _load_brand_labels()


class BrandClassifier(nn.Module):
    """车辆品牌分类模型（基于ResNet迁移学习）"""

    def __init__(self, num_classes=50):
        super(BrandClassifier, self).__init__()
        # 使用预训练的ResNet18作为特征提取器
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class VehicleBrandClassifier:
    """车辆品牌识别器"""

    def __init__(self, model_path='weights/brand_model.pth', num_classes=None):
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
        """加载训练好的品牌分类模型"""
        if not os.path.exists(self.model_path):
            print(f"[品牌识别] 模型文件不存在: {self.model_path}")
            print("[品牌识别] 请先训练模型或使用预训练权重")
            return False

        try:
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            
            # 从模型权重推断类别数
            fc_key = None
            for k in state_dict.keys():
                if k.endswith('fc.3.weight') or k.endswith('fc.3.bias'):
                    fc_key = k
                    break
            if fc_key:
                self.num_classes = state_dict[fc_key].shape[0]
            elif self.num_classes is None:
                self.num_classes = len(BRAND_LABELS)

            # 兼容有无backbone.前缀
            if not any(k.startswith('backbone.') for k in state_dict.keys()):
                state_dict = {'backbone.' + k: v for k, v in state_dict.items()}
            
            self.model = BrandClassifier(num_classes=self.num_classes)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"[品牌识别] 模型加载成功: {self.model_path}")
            return True
        except Exception as e:
            print(f"[品牌识别] 模型加载失败: {e}")
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
        识别车辆品牌

        Args:
            image: 图片路径或numpy数组

        Returns:
            dict: {'brand': '大众家用车', 'confidence': 0.85, 'top3': [...]}
        """
        if self.model is None:
            return {'brand': '未知', 'confidence': 0.0, 'top3': []}

        tensor = self.preprocess(image)
        if tensor is None:
            return {'brand': '未知', 'confidence': 0.0, 'top3': []}

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)

            # Top3 结果
            top3_prob, top3_idx = torch.topk(probabilities, 3, dim=1)
            top3 = []
            for i in range(3):
                idx = top3_idx[0][i].item()
                prob = top3_prob[0][i].item()
                label = BRAND_LABELS[idx] if idx < len(BRAND_LABELS) else '未知'
                top3.append({'brand': label, 'confidence': prob})

            return {
                'brand': top3[0]['brand'],
                'confidence': top3[0]['confidence'],
                'top3': top3
            }


def classify_brand(image_path):
    """
    便捷函数：识别车辆品牌

    Args:
        image_path: 图片路径

    Returns:
        str: 品牌标签
    """
    classifier = VehicleBrandClassifier()
    if classifier.load_model():
        result = classifier.classify(image_path)
        return result['brand']
    return "未知"


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'static/uploads/test.jpg'

    print(f"[测试] 识别品牌: {image_path}")
    brand = classify_brand(image_path)
    print(f"[测试] 识别结果: {brand}")
