"""
统一推理接口
串联所有AI模块，实现完整的车辆特征提取和套牌判定流程

流程：图片输入 → YOLOv5检测 → 裁剪车辆 → 特征提取 → 比对判定

负责人：宋柄儒
"""

import os
import cv2
import numpy as np

from detect import VehicleDetector
from plate_recognize import PlateRecognizer
from color_classify import VehicleColorClassifier
from brand_classify import VehicleBrandClassifier
from compare import compare_features


class VehicleAnalyzer:
    """车辆特征分析器 - 统一推理接口"""

    def __init__(self):
        self.detector = None
        self.plate_recognizer = None
        self.color_classifier = None
        self.brand_classifier = None
        self.vehicle_db = None

    def load_all_models(self, vehicle_db=None):
        """
        加载所有AI模型

        Args:
            vehicle_db: 车辆数据库 DataFrame

        Returns:
            bool: 是否全部加载成功
        """
        success = True

        # 1. 车辆检测器
        self.detector = VehicleDetector(model_path='weights/yolov5m.pt')
        if not self.detector.load_model():
            print("[推理] YOLOv5模型加载失败")
            success = False

        # 2. 车牌识别
        self.plate_recognizer = PlateRecognizer()
        if not self.plate_recognizer.load_model():
            print("[推理] OCR模型加载失败")
            success = False

        # 3. 颜色识别
        self.color_classifier = VehicleColorClassifier(model_path='weights/color_model.pth')
        if not self.color_classifier.load_model():
            print("[推理] 颜色模型加载失败（可选）")

        # 4. 品牌识别
        self.brand_classifier = VehicleBrandClassifier(model_path='weights/brand_model.pth')
        if not self.brand_classifier.load_model():
            print("[推理] 品牌模型加载失败（可选）")

        # 5. 数据库
        self.vehicle_db = vehicle_db

        return success

    def analyze(self, image_path):
        """
        完整分析流程

        Args:
            image_path: 卡口图片路径

        Returns:
            dict: {
                'image_path': 原始图片路径,
                'detections': [
                    {
                        'bbox': [x1, y1, x2, y2],
                        'plate_number': '鲁B62B23',
                        'brand': '现代',
                        'color': '蓝色',
                        'color_cn': '蓝色',
                        'comparison': {...}
                    }
                ],
                'summary': {
                    'total_vehicles': 2,
                    'suspicious_count': 0
                }
            }
        """
        result = {
            'image_path': image_path,
            'detections': [],
            'summary': {
                'total_vehicles': 0,
                'suspicious_count': 0
            }
        }

        if not os.path.exists(image_path):
            result['error'] = f'图片不存在: {image_path}'
            return result

        # Step 1: 车辆检测
        detections = self.detector.detect(image_path)
        result['summary']['total_vehicles'] = len(detections)

        for det in detections:
            vehicle_result = {
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'plate_number': '',
                'brand': '',
                'color': '',
                'color_cn': '',
                'comparison': None
            }

            crop = det['crop']
            if crop is None or crop.size == 0:
                continue

            # Step 2: 车牌识别
            vehicle_result['plate_number'] = self.plate_recognizer.recognize(crop)

            # Step 3: 颜色识别
            if self.color_classifier and self.color_classifier.model:
                color_result = self.color_classifier.classify(crop)
                vehicle_result['color'] = color_result['color']
                vehicle_result['color_cn'] = color_result['color_cn']

            # Step 4: 品牌识别
            if self.brand_classifier and self.brand_classifier.model:
                brand_result = self.brand_classifier.classify(crop)
                vehicle_result['brand'] = brand_result['brand']

            # Step 5: 数据库比对
            if self.vehicle_db is not None:
                comparison = compare_features(
                    vehicle_result['plate_number'],
                    vehicle_result['brand'],
                    vehicle_result['color_cn'],
                    self.vehicle_db
                )
                vehicle_result['comparison'] = comparison

                if comparison['status'] == 'suspicious':
                    result['summary']['suspicious_count'] += 1

            result['detections'].append(vehicle_result)

        return result

    def analyze_simple(self, image_path):
        """
        简化分析（只返回第一条结果）

        Args:
            image_path: 图片路径

        Returns:
            dict: 简化结果
        """
        full_result = self.analyze(image_path)

        if not full_result['detections']:
            return {
                'plate_number': '',
                'brand': '',
                'color': '',
                'judgment': '未检测到车辆',
                'status': 'error'
            }

        det = full_result['detections'][0]
        comparison = det.get('comparison', {})

        return {
            'plate_number': det['plate_number'],
            'brand': det['brand'],
            'color': det.get('color_cn', det.get('color', '')),
            'judgment': comparison.get('message', '待比对') if comparison else '待比对',
            'status': comparison.get('status', 'unknown') if comparison else 'unknown',
            'image_url': f'/static/uploads/{os.path.basename(image_path)}'
        }


# 全局分析器实例
analyzer = None


def get_analyzer():
    """获取全局分析器（懒加载）"""
    global analyzer
    if analyzer is None:
        analyzer = VehicleAnalyzer()
    return analyzer


if __name__ == '__main__':
    from db_loader import load_vehicle_database

    db = load_vehicle_database('vehicle-database.csv')

    analyzer = VehicleAnalyzer()
    analyzer.load_all_models(vehicle_db=db)

    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'static/uploads/test.jpg'

    print(f"[测试] 分析图片: {image_path}")
    result = analyzer.analyze(image_path)

    print(f"\n[结果] 检测到 {result['summary']['total_vehicles']} 辆车")
    for i, det in enumerate(result['detections']):
        print(f"\n  车辆 {i+1}:")
        print(f"    车牌号: {det['plate_number']}")
        print(f"    品牌: {det['brand']}")
        print(f"    颜色: {det.get('color_cn', '未知')}")
        if det['comparison']:
            print(f"    判定: {det['comparison']['message']}")
