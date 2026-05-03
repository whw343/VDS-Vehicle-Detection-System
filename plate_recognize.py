"""
车牌识别模块
从车辆图片中识别车牌号（OCR）

输入：车辆图片路径或numpy数组
输出：车牌号字符串

负责人：肖岱彤
"""

import os
import cv2
import numpy as np
import re


class PlateRecognizer:
    """车牌OCR识别器"""

    def __init__(self, use_gpu=False):
        """
        初始化识别器

        Args:
            use_gpu: 是否使用GPU加速
        """
        self.reader = None
        self.use_gpu = use_gpu

    def load_model(self):
        """加载OCR模型（EasyOCR）"""
        try:
            import easyocr
            self.reader = easyocr.Reader(
                ['ch_sim', 'en'],  # 中文简体 + 英文
                gpu=self.use_gpu
            )
            print("[车牌识别] EasyOCR模型加载成功")
            return True
        except ImportError:
            print("[车牌识别] 未安装EasyOCR，请运行: pip install easyocr")
            return False
        except Exception as e:
            print(f"[车牌识别] 模型加载失败: {e}")
            return False

    def preprocess_plate_region(self, image):
        """
        车牌区域预处理（提高OCR准确率）

        Args:
            image: 车辆图片（numpy array）

        Returns:
            numpy array: 预处理后的图片
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 自适应二值化
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return binary

    def extract_plate_text(self, image):
        """
        从图片中提取车牌号

        Args:
            image: 车辆图片（numpy array 或 文件路径）

        Returns:
            str: 识别出的车牌号，失败返回空字符串
        """
        if self.reader is None:
            print("[车牌识别] 模型未加载")
            return ""

        # 如果是路径则读取
        if isinstance(image, str):
            if not os.path.exists(image):
                return ""
            image = cv2.imread(image)

        if image is None or image.size == 0:
            return ""

        try:
            # OCR识别
            results = self.reader.readtext(image)

            # 过滤并合并结果
            plate_texts = []
            for (bbox, text, confidence) in results:
                # 清理文本
                cleaned = self._clean_plate_text(text)
                if cleaned and confidence > 0.3:
                    plate_texts.append(cleaned)

            # 选择最佳匹配
            if plate_texts:
                # 优先选择符合车牌格式的
                for text in plate_texts:
                    if self._is_valid_plate(text):
                        return text
                # 如果没有完全匹配，返回第一个
                return plate_texts[0]

            return ""

        except Exception as e:
            print(f"[车牌识别] 识别失败: {e}")
            return ""

    def _clean_plate_text(self, text):
        """清理OCR识别文本"""
        # 去除空格和特殊字符
        text = text.strip().replace(' ', '').replace('-', '')
        # 保留字母、数字和中文
        text = re.sub(r'[^\u4e00-\u9fa5A-Z0-9]', '', text)
        return text

    def _is_valid_plate(self, text):
        """判断是否符合车牌格式（鲁B62B23等）"""
        # 中国车牌格式：省份简称 + 字母 + 5位字母数字
        # 宽松匹配
        if len(text) >= 6 and len(text) <= 8:
            has_chinese = any('\u4e00' <= c <= '\u9fa5' for c in text)
            has_letter = any(c.isalpha() for c in text)
            if has_chinese and has_letter:
                return True
        return False

    def recognize(self, image):
        """
        识别车牌号（主接口）

        Args:
            image: 图片路径或numpy数组

        Returns:
            str: 车牌号
        """
        return self.extract_plate_text(image)


def recognize_plate(image_path):
    """
    便捷函数：识别车牌号

    Args:
        image_path: 图片路径

    Returns:
        str: 车牌号
    """
    recognizer = PlateRecognizer()
    if recognizer.load_model():
        return recognizer.recognize(image_path)
    return ""


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'static/uploads/test.jpg'

    print(f"[测试] 识别车牌: {image_path}")
    plate = recognize_plate(image_path)
    print(f"[测试] 识别结果: {plate}")
