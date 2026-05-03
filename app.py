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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# 加载车辆数据库
vehicle_db = load_vehicle_database('vehicle-database.csv')

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


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
        # TODO: 调用AI管线
        # 1. YOLOv5车辆检测
        # detection_result = detect_vehicles(filepath)

        # 2. 车牌OCR识别
        # plate_number = recognize_plate(filepath)

        # 3. 车辆颜色识别
        # color = classify_color(filepath)

        # 4. 车辆品牌识别
        # brand = classify_brand(filepath)

        # 5. 数据库比对
        # comparison_result = compare_features(plate_number, brand, color, vehicle_db)

        # 临时返回测试结果
        result = {
            'image_url': f'/static/uploads/{unique_filename}',
            'plate_number': '待实现',
            'brand': '待实现',
            'color': '待实现',
            'judgment': '待实现',
            'message': 'AI分析功能待集成'
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
        # TODO: 调用AI管线
        result = {
            'status': 'success',
            'image_url': f'/static/uploads/{unique_filename}',
            'analysis': {
                'plate_number': '待实现',
                'brand': '待实现',
                'color': '待实现',
                'type': '待实现'
            },
            'judgment': '待实现'
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
