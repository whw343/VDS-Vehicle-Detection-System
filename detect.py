"""
车辆检测模块
使用 YOLOv5 从卡口图片中检测车辆区域

输入：卡口图片路径
输出：车辆边界框坐标列表 + 裁剪后的车辆图片

负责人：宋柄儒
"""

import os
import torch
import cv2
import numpy as np


class VehicleDetector:
    """YOLOv5 车辆检测器"""

    def __init__(self, model_path='weights/yolov5m.pt', confidence=0.5, device=None):
        """
        初始化检测器

        Args:
            model_path: YOLOv5 模型权重路径
            confidence: 置信度阈值
            device: 设备（'cpu' 或 'cuda'）
        """
        self.model_path = model_path
        self.confidence = confidence
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def load_model(self):
        """加载 YOLOv5 模型"""
        if not os.path.exists(self.model_path):
            print(f"[检测器] 模型文件不存在: {self.model_path}")
            print("[检测器] 请先下载模型权重：")
            print("  方式1: 运行 weights/download_weights.sh")
            print("  方式2: 从 https://github.com/ultralytics/yolov5/releases 下载")
            return False

        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                        path=self.model_path,
                                        force_reload=False)
            self.model.conf = self.confidence
            self.model.to(self.device)
            print(f"[检测器] 模型加载成功: {self.model_path}")
            print(f"[检测器] 使用设备: {self.device}")
            return True
        except Exception as e:
            print(f"[检测器] 模型加载失败: {e}")
            return False

    def detect(self, image_path):
        """
        检测图片中的车辆

        Args:
            image_path: 图片路径

        Returns:
            list[dict]: 每个检测结果包含：
                - bbox: [x1, y1, x2, y2] 边界框坐标
                - confidence: 置信度
                - class_name: 类别名称
                - crop: 裁剪后的车辆图片（numpy array）
        """
        if self.model is None:
            print("[检测器] 模型未加载，请先调用 load_model()")
            return []

        if not os.path.exists(image_path):
            print(f"[检测器] 图片不存在: {image_path}")
            return []

        img = cv2.imread(image_path)
        if img is None:
            print(f"[检测器] 图片读取失败: {image_path}")
            return []

        h, w = img.shape[:2]

        # 执行推理
        results = self.model(image_path)

        detections = []
        for det in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, conf, cls_id = det

            # 坐标转换为整数
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 裁剪车辆区域
            crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf),
                'class_name': self.model.names[int(cls_id)] if hasattr(self.model, 'names') else str(int(cls_id)),
                'crop': crop
            })

        print(f"[检测器] 检测到 {len(detections)} 辆车")
        return detections


def detect_vehicles(image_path, model_path='weights/yolov5m.pt'):
    """
    便捷函数：检测图片中的车辆

    Args:
        image_path: 图片路径
        model_path: 模型权重路径

    Returns:
        list[dict]: 检测结果列表
    """
    detector = VehicleDetector(model_path=model_path)
    if detector.load_model():
        return detector.detect(image_path)
    return []


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'static/uploads/test.jpg'

    print(f"[测试] 检测图片: {image_path}")
    results = detect_vehicles(image_path)

    for i, det in enumerate(results):
        print(f"  车辆 {i+1}: bbox={det['bbox']}, "
              f"conf={det['confidence']:.2f}, "
              f"class={det['class_name']}")
