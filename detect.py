"""
车辆检测模块
使用 YOLOv5 从卡口图片中检测车辆区域

输入：卡口图片路径
输出：车辆边界框坐标列表 + 裁剪后的车辆图片

负责人：宋柄儒
"""

import os
import cv2
import numpy as np


# YOLOv5 COCO数据集中的车辆相关类别ID
# car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]


class VehicleDetector:
    """YOLOv5 车辆检测器"""

    def __init__(self, model_path='weights/yolov5su.pt', confidence=0.45, device=None):
        """
        初始化检测器

        Args:
            model_path: YOLOv5 模型权重路径 (.pt)
            confidence: 置信度阈值
            device: 设备（'cpu' 或 'cuda'），None则自动选择
        """
        self.model_path = model_path
        self.confidence = confidence
        self.device = device or ('cuda' if self._cuda_available() else 'cpu')
        self.model = None
        self.class_names = {}

    @staticmethod
    def _cuda_available():
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def load_model(self):
        """加载 YOLOv5 模型（使用 ultralytics 包）"""
        if not os.path.exists(self.model_path):
            print(f"[检测器] 模型文件不存在: {self.model_path}")
            print("[检测器] 请确保 weights/yolov5su.pt 存在")
            print("[检测器] 可从 https://github.com/ultralytics/yolov5/releases 下载")
            return False

        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)

            # 获取类别名称
            self.class_names = self.model.names if hasattr(self.model, 'names') else {}

            print(f"[检测器] 模型加载成功: {self.model_path}")
            print(f"[检测器] 使用设备: {self.device}")
            print(f"[检测器] 类别数: {len(self.class_names)}")
            return True

        except Exception as e:
            print(f"[检测器] 模型加载失败: {e}")
            return False

    def detect(self, image_path, save_crop=False, save_dir='static/crops'):
        """
        检测图片中的车辆

        Args:
            image_path: 图片路径
            save_crop: 是否保存裁剪的车辆图片到磁盘
            save_dir: 裁剪图片保存目录

        Returns:
            list[dict]: 每个检测结果包含：
                - bbox: [x1, y1, x2, y2] 边界框坐标 (像素)
                - confidence: 置信度
                - class_id: 类别ID
                - class_name: 类别名称（car/bus/truck/motorcycle）
                - crop: 裁剪后的车辆图片（numpy array, BGR格式）
        """
        if self.model is None:
            print("[检测器] 模型未加载，请先调用 load_model()")
            return []

        if not os.path.exists(image_path):
            print(f"[检测器] 图片不存在: {image_path}")
            return []

        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"[检测器] 图片读取失败: {image_path}")
            return []

        h, w = img.shape[:2]

        # 执行推理
        results = self.model(
            image_path,
            conf=self.confidence,
            device=self.device if self.device != 'cpu' else None,
            verbose=False
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # 只保留车辆类别
                if cls_id not in VEHICLE_CLASSES:
                    continue

                # 边界裁剪保护
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # 裁剪车辆区域
                crop = img[y1:y2, x1:x2].copy()

                # 获取类别名
                cls_name = self.class_names.get(cls_id, str(cls_id))

                # 中文映射
                cls_name_cn = {
                    'car': '轿车',
                    'bus': '客车',
                    'truck': '卡车',
                    'motorcycle': '摩托车'
                }.get(cls_name, cls_name)

                det = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'class_name_cn': cls_name_cn,
                    'crop': crop
                }
                detections.append(det)

        # 按置信度降序排列
        detections.sort(key=lambda d: d['confidence'], reverse=True)

        # 可选：保存裁剪图
        if save_crop and detections:
            os.makedirs(save_dir, exist_ok=True)
            for i, det in enumerate(detections):
                crop_path = os.path.join(save_dir, f"crop_{i}.jpg")
                cv2.imwrite(crop_path, det['crop'])
                det['crop_path'] = crop_path

        print(f"[检测器] 检测到 {len(detections)} 辆车")
        return detections

    def detect_and_annotate(self, image_path, output_path=None):
        """
        检测并绘制标注框（用于调试/展示）

        Args:
            image_path: 输入图片路径
            output_path: 标注结果保存路径，None则自动生成

        Returns:
            tuple: (detections, annotated_image_path)
        """
        detections = self.detect(image_path)
        if not detections:
            return detections, None

        img = cv2.imread(image_path)

        # 颜色映射
        colors = {
            'car': (0, 255, 0),       # 绿色
            'bus': (0, 165, 255),      # 橙色
            'truck': (0, 0, 255),      # 红色
            'motorcycle': (255, 0, 0), # 蓝色
        }

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class_name']
            color = colors.get(cls_name, (255, 255, 255))

            # 绘制边框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{det['class_name_cn']} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_detected{ext}"

        cv2.imwrite(output_path, img)
        print(f"[检测器] 标注图已保存: {output_path}")
        return detections, output_path


def detect_vehicles(image_path, model_path='weights/yolov5su.pt', confidence=0.45):
    """
    便捷函数：检测图片中的车辆

    Args:
        image_path: 图片路径
        model_path: 模型权重路径
        confidence: 置信度阈值

    Returns:
        list[dict]: 检测结果列表
    """
    detector = VehicleDetector(model_path=model_path, confidence=confidence)
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
    print("=" * 60)

    detector = VehicleDetector()
    if not detector.load_model():
        sys.exit(1)

    detections, annotated_path = detector.detect_and_annotate(image_path)

    print(f"\n[结果] 共检测到 {len(detections)} 辆车")
    for i, det in enumerate(detections):
        print(f"\n  车辆 {i+1}:")
        print(f"    类型: {det['class_name_cn']} ({det['class_name']})")
        print(f"    置信度: {det['confidence']:.2%}")
        print(f"    边界框: {det['bbox']}")
        print(f"    裁剪尺寸: {det['crop'].shape}")

    if annotated_path:
        print(f"\n[输出] 标注图: {annotated_path}")
