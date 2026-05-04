"""
套牌车稽查系统 - Flask Web应用
基于车辆特征分析的套牌车稽查系统

运行方式：python app.py
访问地址：http://localhost:5000

负责人：满瀚宇（组长）
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import uuid
import logging
from datetime import datetime
from werkzeug.utils import secure_filename

# 导入自定义模块
from db_loader import load_vehicle_database, query_vehicle, get_database_stats
from compare import compare_features, summarize_result, get_judgment_emoji
from detect import VehicleDetector
from plate_recognize import PlateRecognizer
from color_classify import VehicleColorClassifier
from brand_classify import VehicleBrandClassifier

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# 加载车辆数据库
vehicle_db = load_vehicle_database('vehicle-database.csv')

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 系统启动时间
START_TIME = datetime.now()

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


def _run_ai_pipeline(filepath):
    """
    执行AI分析管线

    Returns:
        (dict, str or None): (分析结果, 裁剪图路径)
    """
    # 1. YOLOv5车辆检测
    detector = get_detector()
    detections = detector.detect(filepath)

    if not detections:
        return None, None

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
        try:
            color_result = color_classifier.classify(crop)
            color_cn = color_result.get('color_cn', '未知')
        except Exception as e:
            logger.warning(f"颜色识别失败: {e}")

    # 4. 车辆品牌识别
    brand = '未知'
    brand_classifier = get_brand_classifier()
    if brand_classifier.model:
        try:
            brand_result = brand_classifier.classify(crop)
            brand = brand_result.get('brand', '未知')
        except Exception as e:
            logger.warning(f"品牌识别失败: {e}")

    # 5. 数据库比对
    comparison = compare_features(plate_number, brand, color_cn, vehicle_db)

    result = {
        'plate_number': plate_number or '未识别',
        'brand': brand,
        'color': color_cn,
        'type': best.get('class_name_cn', '未知'),
        'confidence': f"{best['confidence']:.1%}" if best.get('confidence') else '',
        'vehicle_count': len(detections),
        'comparison': comparison,
        'judgment': comparison.get('message', '待比对'),
        'judgment_emoji': get_judgment_emoji(comparison.get('status', 'unknown')),
        'score': comparison.get('score', 0)
    }

    return result, crop


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
        return jsonify({'error': '不支持的文件格式，请上传 JPG/PNG/BMP'}), 400

    # 保存上传的图片
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    logger.info(f"收到上传: {filename} → {unique_filename}")

    try:
        ai_result, crop = _run_ai_pipeline(filepath)

        if ai_result is None:
            result = {
                'image_url': f'/static/uploads/{unique_filename}',
                'plate_number': '',
                'brand': '',
                'color': '',
                'type': '',
                'judgment': '未检测到车辆',
                'judgment_emoji': '❌',
                'message': '图片中未检测到车辆'
            }
        else:
            result = {
                'image_url': f'/static/uploads/{unique_filename}',
                **ai_result,
                'message': f"检测到 {ai_result['vehicle_count']} 辆车"
            }

        logger.info(f"分析完成: {result.get('plate_number', '无')} - {result.get('judgment', '')[:50]}")

        # 保存裁剪图
        if crop is not None and crop.size > 0:
            import cv2
            crop_path = os.path.join('static/uploads', f"crop_{unique_filename}")
            cv2.imwrite(crop_path, crop)
            result['crop_url'] = f'/static/uploads/crop_{unique_filename}'

        return render_template('result.html', result=result)

    except Exception as e:
        logger.error(f"分析异常: {e}", exc_info=True)
        return jsonify({'error': f'分析失败，请重试'}), 500


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API接口：图片分析（JSON返回）"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'error': '没有上传文件'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'status': 'error', 'error': '无效文件'}), 400

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    logger.info(f"API分析: {filename}")

    try:
        ai_result, crop = _run_ai_pipeline(filepath)

        if ai_result is None:
            return jsonify({
                'status': 'no_vehicle',
                'message': '未检测到车辆',
                'image_url': f'/static/uploads/{unique_filename}'
            })

        response_data = {
            'status': 'success',
            'image_url': f'/static/uploads/{unique_filename}',
            'analysis': {
                'plate_number': ai_result['plate_number'],
                'brand': ai_result['brand'],
                'color': ai_result['color'],
                'type': ai_result['type'],
                'confidence': ai_result['confidence']
            },
            'judgment': ai_result['judgment'],
            'judgment_emoji': ai_result['judgment_emoji'],
            'score': ai_result['score'],
            'vehicle_count': ai_result['vehicle_count']
        }

        logger.info(f"API分析完成: {ai_result['plate_number']} - {ai_result['judgment'][:50]}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"API分析异常: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': '服务器内部错误'}), 500


@app.route('/api/health')
def health_check():
    """健康检查接口"""
    db_stats = get_database_stats(vehicle_db)

    return jsonify({
        'status': 'ok',
        'version': '0.2',
        'uptime': str(datetime.now() - START_TIME).split('.')[0],
        'database': {
            'records': db_stats.get('total_records', 0),
            'brands': db_stats.get('unique_brands', 0),
            'loaded': vehicle_db is not None and not vehicle_db.empty
        },
        'models': {
            'detector': _detector is not None,
            'plate_ocr': _plate_recognizer is not None,
            'color_classifier': _color_classifier is not None and _color_classifier.model is not None,
            'brand_classifier': _brand_classifier is not None and _brand_classifier.model is not None,
        }
    })


@app.route('/api/db/stats')
def db_stats():
    """数据库统计接口"""
    stats = get_database_stats(vehicle_db)
    return jsonify(stats)


@app.route('/result')
def result():
    """结果展示页"""
    return render_template('result.html', result={})


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': '文件过大，最大支持16MB'}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': '服务器内部错误，请重试'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  套牌车稽查系统 V0.2")
    print("  访问地址: http://localhost:5000")
    print(f"  数据库: {len(vehicle_db)} 条记录")
    print(f"  启动时间: {START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
