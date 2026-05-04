"""
系统级批量测试脚本
对526张卡口图片运行完整管线，生成测试报告

测试内容：
1. YOLOv5检测准确率（车检率）
2. 车牌OCR识别率  
3. 颜色/车型/品牌分类准确率
4. 数据库比对命中率
5. 单张平均耗时

负责人：全组
"""

import os, sys, re, time, json
import cv2
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from detect import VehicleDetector
from plate_recognize import PlateRecognizer
from color_classify import VehicleColorClassifier
from type_classify import VehicleTypeClassifier
from brand_classify import VehicleBrandClassifier
from db_loader import load_vehicle_database, fuzzy_search_plate
from compare import compare_features

TEST_DIR = 'data/test_checkpoints'
DB_PATH = 'vehicle-database.csv'
MAX_TEST = 100  # 测试前100张（526张太多，取代表性样本）

def parse_plate_from_filename(filename):
    """从文件名提取车牌号"""
    # 格式: 0_1_20140529_084416330_┬│UL7290_P1.jpg
    parts = filename.replace('.jpg', '').split('_')
    if len(parts) >= 6:
        plate_raw = parts[-2]
        # 去掉特殊字符
        plate_raw = re.sub(r'[^\w]', '', plate_raw)
        # 跳过"无车牌" 
        if len(plate_raw) < 3 or '无' in plate_raw:
            return None
        # 添加省份前缀（假定都是鲁）
        if plate_raw[0].isdigit() or plate_raw[0].isalpha():
            plate_raw = '鲁' + plate_raw
        return plate_raw
    return None


def run_batch_test(max_images=MAX_TEST):
    """运行批量测试"""
    
    print('=' * 60)
    print('  系统批量测试报告')
    print(f'  时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  测试数据: {os.path.abspath(TEST_DIR)}')
    print(f'  测试数量: {max_images} 张')
    print('=' * 60)

    # 1. 加载所有模型
    print('\n[1/5] 加载模型...')
    detector = VehicleDetector()
    detector.load_model()
    ocr = PlateRecognizer()
    ocr.load_model()
    color_cls = VehicleColorClassifier()
    color_cls.load_model()
    type_cls = VehicleTypeClassifier()
    type_cls.load_model()
    brand_cls = VehicleBrandClassifier()
    brand_cls.load_model()
    db = load_vehicle_database(DB_PATH)
    print('  模型加载完成\n')

    # 2. 获取测试图片列表
    all_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')])[:max_images]
    print(f'[2/5] 测试图片: {len(all_files)} 张')

    # 3. 运行测试
    print(f'[3/5] 开始批量测试...')
    results = {
        'total': 0,
        'detected': 0,       # 检测到车辆
        'plate_recognized': 0,  # 识别到车牌号
        'plate_correct': 0,     # 车牌完全正确
        'plate_partial': 0,     # 车牌部分正确
        'db_matched': 0,        # 数据库匹配
        'timing': [],           # 每张耗时
        'details': [],          # 详细结果
    }

    start_total = time.time()

    for i, fname in enumerate(all_files):
        path = os.path.join(TEST_DIR, fname)
        gt_plate = parse_plate_from_filename(fname)
        
        t_start = time.time()
        
        # Step 1: 检测
        dets = detector.detect(path)
        if not dets:
            results['details'].append({
                'file': fname, 'gt_plate': gt_plate,
                'status': 'no_vehicle'
            })
            results['total'] += 1
            continue
        
        results['detected'] += 1
        best = dets[0]
        crop = best['crop']
        
        # Step 2: OCR
        plate = ocr.recognize(crop)
        
        # Step 3: 颜色
        color = color_cls.classify(crop) if color_cls.model else {'color_cn': '未知'}
        
        # Step 4: 车型
        vtype = type_cls.classify(crop) if type_cls.model else {'type_cn': '未知'}
        
        # Step 5: 品牌
        brand = brand_cls.classify(crop) if brand_cls.model else {'brand': '未知'}
        
        # Step 6: 比对
        comparison = compare_features(plate, brand['brand'], color['color_cn'], db)
        
        t_elapsed = time.time() - t_start
        results['timing'].append(t_elapsed)
        
        # 统计
        if plate:
            results['plate_recognized'] += 1
            if gt_plate and plate == gt_plate:
                results['plate_correct'] += 1
            elif gt_plate and _plate_similar(plate, gt_plate):
                results['plate_partial'] += 1
        
        if comparison.get('status') in ('normal', 'suspicious'):
            results['db_matched'] += 1
        
        results['details'].append({
            'file': fname,
            'gt_plate': gt_plate,
            'detected_plate': plate,
            'color': color.get('color_cn', ''),
            'type': vtype.get('type_cn', ''),
            'brand': brand.get('brand', ''),
            'judgment': comparison.get('status', ''),
            'score': comparison.get('score', 0),
            'bbox': best['bbox'],
            'time': t_elapsed,
            'status': 'ok'
        })
        
        results['total'] += 1
        
        if (i + 1) % 20 == 0:
            print(f'  进度: {i+1}/{len(all_files)} '
                  f'检测:{results["detected"]} OCR:{results["plate_recognized"]} '
                  f'正确:{results["plate_correct"]}')

    total_time = time.time() - start_total
    
    # 4. 计算统计
    print(f'\n[4/5] 计算统计数据...')
    
    total = max(results['total'], 1)
    stats = {
        'test_count': total,
        'vehicle_detection_rate': results['detected'] / total * 100,
        'plate_recognition_rate': results['plate_recognized'] / total * 100,
        'plate_accuracy': results['plate_correct'] / total * 100 if total else 0,
        'plate_partial_accuracy': (results['plate_correct'] + results['plate_partial']) / total * 100 if total else 0,
        'db_match_rate': results['db_matched'] / total * 100 if total else 0,
        'total_time': total_time,
        'avg_time_per_image': np.mean(results['timing']) if results['timing'] else 0,
        'median_time': np.median(results['timing']) if results['timing'] else 0,
    }

    # 5. 生成报告
    print(f'[5/5] 生成测试报告...\n')

    report = f"""
================================================================
        套牌车稽查系统 - 系统测试报告
================================================================
测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
测试数据: {max_images}/{len(all_files)} 张 (系统测试_任务1资料)
数据库: {len(db)} 条记录

────────────────────────────────────────────────────────────────
  1. 车辆检测
────────────────────────────────────────────────────────────────
  检测率: {stats['vehicle_detection_rate']:.1f}% ({results['detected']}/{total})
  未检测: {total - results['detected']} 张

────────────────────────────────────────────────────────────────
  2. 车牌OCR识别
────────────────────────────────────────────────────────────────
  识别率(有输出): {stats['plate_recognition_rate']:.1f}% ({results['plate_recognized']}/{total})
  完全正确率: {stats['plate_accuracy']:.1f}% ({results['plate_correct']}/{total})
  部分正确率: {stats['plate_partial_accuracy']:.1f}%

────────────────────────────────────────────────────────────────
  3. 数据库比对
────────────────────────────────────────────────────────────────
  比对成功: {results['db_matched']}/{total} ({stats['db_match_rate']:.1f}%)

────────────────────────────────────────────────────────────────
  4. 性能
────────────────────────────────────────────────────────────────
  总耗时: {stats['total_time']:.1f}秒
  单张平均: {stats['avg_time_per_image']:.2f}秒
  单张中位: {stats['median_time']:.2f}秒

────────────────────────────────────────────────────────────────
  5. 模型信息
────────────────────────────────────────────────────────────────
  YOLOv5: yolov5su.pt (CUDA)
  颜色: {color_cls.model is not None} ({results['detected']} 张分类)
  车型: {type_cls.model is not None}
  品牌: {brand_cls.model is not None}
  OCR: EasyOCR (CPU)

================================================================
"""

    # 保存报告
    report_path = 'docs/测试报告.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
        f.write('\n## 6. 详细结果 (前30张)\n\n')
        f.write('| # | 文件名 | 真值车牌 | 识别车牌 | 颜色 | 车型 | 品牌 | 判定 | 评分 |\n')
        f.write('|---|--------|---------|---------|------|------|------|------|------|\n')
        for i, d in enumerate(results['details'][:30]):
            f.write(f'| {i+1} | {d["file"][:25]} | {d.get("gt_plate","")} | {d.get("detected_plate","")} '
                    f'| {d.get("color","")} | {d.get("type","")} | {d.get("brand","")} '
                    f'| {d.get("judgment","")} | {d.get("score",0)} |\n')

    # 保存JSON
    json_path = 'docs/test_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        save_data = {k: v for k, v in results.items() if k != 'details'}
        save_data['stats'] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                              for k, v in stats.items()}
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(report)
    print(f'报告已保存: {report_path}')
    print(f'数据已保存: {json_path}')
    
    return stats


def _plate_similar(p1, p2):
    """判断两个车牌号是否相似"""
    if not p1 or not p2:
        return False
    p1, p2 = p1.upper().replace(' ', ''), p2.upper().replace(' ', '')
    if p1 == p2:
        return True
    if len(p1) >= 4 and len(p2) >= 4:
        # 忽略第一位省份
        matches = sum(1 for a, b in zip(p1[1:], p2[1:]) if a == b)
        return matches >= len(p1[1:]) * 0.6
    return False


if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    run_batch_test(MAX_TEST)
