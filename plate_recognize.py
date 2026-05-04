"""
车牌识别模块
从车辆图片中识别车牌号（OCR）

输入：车辆图片路径或numpy数组
输出：车牌号字符串 + 置信度

负责人：肖岱彤
"""

import os
import cv2
import numpy as np
import re


# 中国车牌省份简称
CHINESE_PROVINCES = set('京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁')


class PlateRecognizer:
    """
    车牌OCR识别器

    工作流程：
    1. 输入车辆图片（YOLOv5裁剪结果）
    2. 检测车牌区域（颜色过滤 + 轮廓检测）
    3. 裁剪车牌区域
    4. EasyOCR文字识别
    5. 格式校验 + 清洗输出
    """

    def __init__(self, use_gpu=False):
        self.reader = None
        self.use_gpu = use_gpu
        self._model_loaded = False

    def load_model(self):
        """加载OCR模型（EasyOCR）"""
        if self._model_loaded:
            return True

        try:
            import easyocr
            self.reader = easyocr.Reader(
                ['ch_sim', 'en'],
                gpu=self.use_gpu
            )
            self._model_loaded = True
            print("[车牌识别] EasyOCR模型加载成功")
            return True
        except ImportError:
            print("[车牌识别] 未安装EasyOCR，请运行: pip install easyocr")
            return False
        except Exception as e:
            print(f"[车牌识别] 模型加载失败: {e}")
            return False

    # ── 车牌区域检测 ──────────────────────────────

    def detect_plate_region(self, image):
        """
        在车辆图片中检测车牌区域

        策略：
        1. 蓝色车牌（蓝底白字）：HSV蓝色范围过滤
        2. 绿色车牌（新能源）：HSV绿色范围过滤
        3. 黄色车牌（大型车）：HSV黄色范围过滤
        4. 白色车牌（警车等）：边缘检测 + 矩形轮廓

        Args:
            image: BGR格式 numpy array

        Returns:
            numpy array or None: 裁剪的车牌区域图片（BGR）
        """
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]

        # 尝试多种颜色过滤
        candidates = []

        # 1. 蓝色车牌检测
        blue_mask = self._color_filter(image, 'blue')
        blue_crop = self._find_plate_from_mask(image, blue_mask, 'blue')
        if blue_crop is not None:
            candidates.append(('blue', blue_crop))

        # 2. 绿色车牌检测（新能源）
        green_mask = self._color_filter(image, 'green')
        green_crop = self._find_plate_from_mask(image, green_mask, 'green')
        if green_crop is not None:
            candidates.append(('green', green_crop))

        # 3. 黄色车牌检测
        yellow_mask = self._color_filter(image, 'yellow')
        yellow_crop = self._find_plate_from_mask(image, yellow_mask, 'yellow')
        if yellow_crop is not None:
            candidates.append(('yellow', yellow_crop))

        # 4. 白色车牌（边缘检测备选）
        white_crop = self._detect_white_plate(image)
        if white_crop is not None:
            candidates.append(('white', white_crop))

        if candidates:
            # 选面积最大的
            best = max(candidates, key=lambda x: x[1].shape[0] * x[1].shape[1])
            return best[1]

        # 如果都检测不到，返回原图
        return image

    def _color_filter(self, image, color_type):
        """
        基于HSV颜色空间的车牌颜色过滤

        Args:
            image: BGR图片
            color_type: 'blue' | 'green' | 'yellow'

        Returns:
            二值mask
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        ranges = {
            'blue':   (np.array([90, 50, 50]),   np.array([140, 255, 255])),
            'green':  (np.array([35, 40, 40]),   np.array([85, 255, 255])),
            'yellow': (np.array([15, 40, 40]),   np.array([40, 255, 255])),
        }

        lower, upper = ranges.get(color_type, ranges['blue'])
        mask = cv2.inRange(hsv, lower, upper)

        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def _find_plate_from_mask(self, image, mask, color_name):
        """
        从颜色mask中找到最大的矩形区域作为车牌候选

        Args:
            image: 原始BGR图片
            mask: 颜色过滤的二值mask
            color_name: 颜色名称（调试用）

        Returns:
            numpy array or None
        """
        h, w = image.shape[:2]
        min_area = max(w * h * 0.02, 500)  # 至少占图像2%面积

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_rects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, rw, rh = cv2.boundingRect(cnt)

            # 车牌宽高比约 3:1 ~ 5:1（宽 > 高）
            aspect_ratio = rw / rh if rh > 0 else 0
            if 1.8 <= aspect_ratio <= 8.0 and rh > 10:
                valid_rects.append((area, x, y, rw, rh))

        if not valid_rects:
            return None

        # 选面积最大的
        valid_rects.sort(reverse=True, key=lambda r: r[0])
        _, x, y, rw, rh = valid_rects[0]

        # 扩展边距
        pad = int(min(rw, rh) * 0.15)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + rw + pad)
        y2 = min(h, y + rh + pad)

        return image[y1:y2, x1:x2].copy()

    def _detect_white_plate(self, image):
        """白色车牌检测（基于边缘+Canny）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Canny边缘检测
        edges = cv2.Canny(gray, 50, 200)

        # 膨胀连接边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        h, w = image.shape[:2]
        min_area = max(w * h * 0.015, 300)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_rects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, rw, rh = cv2.boundingRect(cnt)
            aspect_ratio = rw / rh if rh > 0 else 0
            if 2.0 <= aspect_ratio <= 6.0 and rh > 8:
                valid_rects.append((area, x, y, rw, rh))

        if not valid_rects:
            return None

        valid_rects.sort(reverse=True, key=lambda r: r[0])
        _, x, y, rw, rh = valid_rects[0]

        pad = int(min(rw, rh) * 0.1)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + rw + pad)
        y2 = min(h, y + rh + pad)

        return image[y1:y2, x1:x2].copy()

    # ── 图片预处理 ─────────────────────────────────

    def preprocess_for_ocr(self, image):
        """
        多策略预处理，提高OCR识别率

        返回多个预处理版本，OCR对每个版本都识别，取最佳结果
        """
        results = []

        # 确保是BGR格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h, w = image.shape[:2]

        # 如果图片太小，放大
        if h < 50:
            scale = 100 / h
            image = cv2.resize(image, (int(w * scale), 100))

        # 版本1：原始图片
        results.append(('original', image))

        # 版本2：灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results.append(('gray', gray))

        # 版本3：CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results.append(('clahe', enhanced))

        # 版本4：二值化（OTSU）
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 判断是否需要反转（白底黑字 vs 黑底白字）
        white_pixels = np.sum(otsu == 255)
        black_pixels = np.sum(otsu == 0)
        if black_pixels > white_pixels:
            otsu = cv2.bitwise_not(otsu)

        results.append(('otsu', otsu))

        return results

    # ── OCR识别 ────────────────────────────────────

    def extract_plate_text(self, image):
        """
        从图片中提取车牌号（完整流程）

        Args:
            image: 车辆图片（numpy array 或 文件路径）

        Returns:
            str: 识别出的车牌号，失败返回空字符串
        """
        if self.reader is None:
            print("[车牌识别] 模型未加载")
            return ""

        # 路径输入 → 读取
        if isinstance(image, str):
            if not os.path.exists(image):
                return ""
            image = cv2.imread(image)

        if image is None or image.size == 0:
            return ""

        # Step 1: 检测车牌区域
        plate_region = self.detect_plate_region(image)
        if plate_region is None:
            plate_region = image

        # Step 2: 多策略预处理 + OCR
        all_texts = []
        preprocessed = self.preprocess_for_ocr(plate_region)

        for name, proc_img in preprocessed:
            try:
                results = self.reader.readtext(proc_img)
                for bbox, text, confidence in results:
                    cleaned = self._clean_plate_text(text)
                    if cleaned and confidence > 0.2:
                        all_texts.append((cleaned, confidence, name))
            except Exception as e:
                continue

        if not all_texts:
            return ""

        # Step 3: 选择最佳结果
        # 优先：符合车牌格式 + 高置信度
        all_texts.sort(key=lambda x: (self._plate_score(x[0]), x[1]), reverse=True)

        best_text = all_texts[0][0]
        best_conf = all_texts[0][1]

        if best_conf < 0.3 and not self._is_valid_plate(best_text):
            print(f"[车牌识别] 低置信度结果: {best_text} ({best_conf:.2%})")
            return ""

        print(f"[车牌识别] 识别结果: {best_text} ({best_conf:.2%})")
        return best_text

    def _plate_score(self, text):
        """车牌号质量评分（越高越好）"""
        score = 0

        # 标准格式匹配
        if self._is_valid_plate(text):
            score += 50

        # 长度合适（7位标准车牌）
        if len(text) == 7:
            score += 20
        elif len(text) == 8:  # 新能源
            score += 15

        # 以省份简称开头
        if text and text[0] in CHINESE_PROVINCES:
            score += 10

        # 第二位是大写字母
        if len(text) >= 2 and text[1].isalpha() and text[1].isascii():
            score += 10

        return score

    def _clean_plate_text(self, text):
        """清理OCR识别文本"""
        text = text.strip().replace(' ', '').replace('-', '').replace('.', '')
        text = text.replace('·', '').replace(':', '').replace('：', '')

        # 保留：中文 + 大写英文 + 数字
        cleaned = ''
        for ch in text:
            if ('\u4e00' <= ch <= '\u9fa5') or ch.isdigit() or (ch.isalpha() and ch.isascii()):
                cleaned += ch.upper()

        return cleaned

    def _is_valid_plate(self, text):
        """
        判断是否符合中国车牌格式

        支持：
        - 标准蓝牌：京A12345（7位）
        - 新能源绿牌：京AD12345（8位）
        - 黄牌（大型车）：京A12345（7位）
        """
        if not text or len(text) < 6 or len(text) > 8:
            return False

        # 必须以省份简称开头
        if text[0] not in CHINESE_PROVINCES:
            return False

        # 第二位必须是字母
        if len(text) >= 2:
            if not (text[1].isalpha() and text[1].isascii()):
                return False

        # 至少有3个数字
        digit_count = sum(1 for c in text if c.isdigit())
        if digit_count < 3:
            return False

        return True

    def recognize(self, image, return_details=False):
        """
        识别车牌号（主接口）

        Args:
            image: 图片路径或numpy数组
            return_details: 是否返回详细信息

        Returns:
            str 或 dict: 车牌号，或包含详细信息的dict
        """
        plate_number = self.extract_plate_text(image)

        if return_details:
            return {
                'plate_number': plate_number,
                'is_valid': self._is_valid_plate(plate_number),
                'length': len(plate_number) if plate_number else 0
            }

        return plate_number


# ── 便捷函数 ───────────────────────────────────

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


# ── 测试入口 ───────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'static/uploads/test.jpg'

    print(f"[测试] 识别车牌: {image_path}")
    print("=" * 60)

    recognizer = PlateRecognizer()
    if not recognizer.load_model():
        print("[错误] 模型加载失败")
        sys.exit(1)

    result = recognizer.recognize(image_path, return_details=True)
    print(f"\n识别结果: {result['plate_number']}")
    print(f"格式合法: {result['is_valid']}")
    print(f"字符长度: {result['length']}")
