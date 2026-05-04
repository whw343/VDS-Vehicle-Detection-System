"""
套牌车稽查系统 - Flask Web应用
基于车辆特征分析的套牌车稽查系统

运行方式：python app.py
访问地址：http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import uuid
from werkzeug.utils import secure_filename

# 导入自定义模块
from db_loader import load_vehicle_database, query_vehicle
from compare import compare_features
from detect import VehicleDetector
from plate_recognize import PlateRecognizer
from color_classify import VehicleColorClassifier
from brand_classify import VehicleBrandClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# 加载车辆数据库
vehicle_db = load_vehicle_database('vehicle-database.csv')

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局模型实例（懒加载）
_detector = None
_plate_recognizer = None
_color_classifier = None
_brand_classifier = None

def get_detector():
    """获取车辆检测器（懒加载）"""
    global _detector
    if _detector is None:
        _detector = VehicleDetector(model_path='weights/yolov5su.pt')
        _detector.load_model()
    return _detector

def get_plate_recognizer():
    """获取车牌识别器（懒加载）"""
    global _plate_recognizer
    if _plate_recognizer is None:
        _plate_recognizer = PlateRecognizer()
        _plate_recognizer.load_model()
    return _plate_recognizer

def get_color_classifier():
    """获取颜色分类器（懒加载）"""
    global _color_classifier
    if _color_classifier is None:
        _color_classifier = VehicleColorClassifier(model_path='weights/color_model.pth')
        _color_classifier.load_model()
    return _color_classifier

def get_brand_classifier():
    """获取品牌分类器（懒加载）"""
    global _brand_classifier
    if _brand_classifier is None:
        _brand_classifier = VehicleBrandClassifier(model_path='weights/brand_model.pth')
        _brand_classifier.load_model()
    return _brand_classifier


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """处理图片上传并进行分析"""
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式'}), 400

    # 保存上传的图片
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    try:
        # 1. YOLOv5车辆检测
        detector = get_detector()
        detections = detector.detect(filepath)

        if not detections:
            result = {
                'image_url': f'/static/uploads/{unique_filename}',
                'plate_number': '',
                'brand': '',
                'color': '',
                'type': '',
                'judgment': '未检测到车辆',
                'message': '图片中未检测到车辆'
            }
            return render_template('result.html', result=result)

        # 取置信度最高的车辆
        best = detections[0]
        crop = best['crop']

        # 2. 车牌OCR识别
        plate_recognizer = get_plate_recognizer()
        plate_number = plate_recognizer.recognize(crop)

        # 3. 车辆颜色识别
        color_cn = '未知'
        color_classifier = get_color_classifier()
        if color_classifier.model:
            color_result = color_classifier.classify(crop)
            color_cn = color_result.get('color_cn', '未知')

        # 4. 车辆品牌识别
        brand = '未知'
        brand_classifier = get_brand_classifier()
        if brand_classifier.model:
            brand_result = brand_classifier.classify(crop)
            brand = brand_result.get('brand', '未知')

        # 5. 数据库比对
        comparison = compare_features(plate_number, brand, color_cn, vehicle_db)

        result = {
            'image_url': f'/static/uploads/{unique_filename}',
            'plate_number': plate_number or '未识别',
            'brand': brand,
            'color': color_cn,
            'type': best.get('class_name_cn', '未知'),
            'confidence': f"{best['confidence']:.1%}" if best.get('confidence') else '',
            'judgment': comparison.get('message', '待比对') if comparison else '待比对',
            'message': f"检测到 {len(detections)} 辆车"
        }

        return render_template('result.html', result=result)

    except Exception as e:
        return jsonify({'error': f'分析失败: {str(e)}'}), 500


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API接口：图片分析"""
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': '无效文件'}), 400

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    try:
        # 1. YOLOv5车辆检测
        detector = get_detector()
        detections = detector.detect(filepath)

        if not detections:
            return jsonify({'status': 'no_vehicle', 'message': '未检测到车辆'})

        best = detections[0]
        crop = best['crop']

        # 2. 车牌 + 颜色 + 品牌
        plate_recognizer = get_plate_recognizer()
        plate_number = plate_recognizer.recognize(crop)

        color_classifier = get_color_classifier()
        color_cn = '未知'
        if color_classifier.model:
            color_cn = color_classifier.classify(crop).get('color_cn', '未知')

        brand = '未知'
        brand_classifier = get_brand_classifier()
        if brand_classifier.model:
            brand = brand_classifier.classify(crop).get('brand', '未知')

        comparison = compare_features(plate_number, brand, color_cn, vehicle_db)

        result = {
            'status': 'success',
            'image_url': f'/static/uploads/{unique_filename}',
            'analysis': {
                'plate_number': plate_number or '未识别',
                'brand': brand,
                'color': color_cn,
                'type': best.get('class_name_cn', '未知'),
                'confidence': f"{best['confidence']:.1%}" if best.get('confidence') else ''
            },
            'judgment': comparison.get('message', '待比对') if comparison else '待比对'
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/result')
def result():
    """结果展示页"""
    return render_template('result.html', result={})


if __name__ == '__main__':
    print("=" * 50)
    print("  套牌车稽查系统 V0.1")
    print("  访问地址: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
