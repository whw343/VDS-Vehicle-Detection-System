"""
套牌比对模块
将AI提取的车辆特征与数据库比对，判定是否套牌

输入：车牌号、车辆品牌、车辆颜色
输出：判定结果

负责人：满瀚宇
"""

import pandas as pd


def compare_features(plate_number, detected_brand, detected_color, vehicle_db):
    """
    将提取的特征与数据库比对

    Args:
        plate_number: 识别出的车牌号（如：鲁B62B23）
        detected_brand: 识别出的车辆品牌（如：现代）
        detected_color: 识别出的车辆颜色（如：蓝色）
        vehicle_db: 车辆数据库 DataFrame

    Returns:
        dict: {
            'status': 'normal' | 'suspicious' | 'not_found',
            'message': 判定说明,
            'details': {
                'plate_match': True/False,
                'brand_match': True/False,
                'color_match': True/False,
                'db_record': 数据库中的记录
            }
        }
    """
    result = {
        'status': 'not_found',
        'message': '',
        'details': {
            'plate_match': False,
            'brand_match': False,
            'color_match': False,
            'db_record': None
        }
    }

    if vehicle_db is None or vehicle_db.empty:
        result['message'] = '数据库未加载'
        return result

    if not plate_number:
        result['message'] = '未识别出车牌号'
        return result

    # 1. 在数据库中查询车牌号
    db_record = vehicle_db[vehicle_db['plateNo'] == plate_number]

    if db_record.empty:
        result['status'] = 'suspicious'
        result['message'] = f'套牌嫌疑：车牌 {plate_number} 不存在于登记库'
        result['details']['plate_match'] = False
        return result

    result['details']['plate_match'] = True
    result['details']['db_record'] = db_record.iloc[0].to_dict()

    # 获取数据库中的品牌
    db_brand = db_record.iloc[0]['carBrand']

    # 2. 比对品牌
    brand_match = _compare_brand(detected_brand, db_brand)
    result['details']['brand_match'] = brand_match

    # 3. 判定结果
    if not brand_match:
        result['status'] = 'suspicious'
        result['message'] = (
            f'套牌嫌疑：车牌 {plate_number} 匹配，'
            f'但品牌不符（识别：{detected_brand}，登记：{db_brand}）'
        )
    else:
        result['status'] = 'normal'
        result['message'] = f'正常车辆：车牌 {plate_number}，品牌 {db_brand}'

    return result


def _compare_brand(detected_brand, db_brand):
    """
    比对车辆品牌（模糊匹配）

    Args:
        detected_brand: AI识别的品牌
        db_brand: 数据库中的品牌

    Returns:
        bool: 是否匹配
    """
    if not detected_brand or not db_brand:
        return False

    # 完全匹配
    if detected_brand == db_brand:
        return True

    # 包含匹配（如识别"现代"，数据库为"北京现代"）
    if detected_brand in db_brand or db_brand in detected_brand:
        return True

    # 品牌关键词匹配
    brand_keywords = {
        '大众': ['大众'],
        '丰田': ['丰田'],
        '本田': ['本田'],
        '日产': ['日产'],
        '现代': ['现代'],
        '起亚': ['起亚'],
        '宝马': ['宝马'],
        '奔驰': ['奔驰'],
        '奥迪': ['奥迪'],
        '福特': ['福特'],
        '雪佛兰': ['雪佛兰'],
        '别克': ['别克'],
        '比亚迪': ['比亚迪'],
        '长安': ['长安'],
        '长城': ['长城'],
        '奇瑞': ['奇瑞'],
        '江淮': ['江淮'],
        '五菱': ['五菱'],
    }

    for brand_key, aliases in brand_keywords.items():
        detected_match = any(alias in detected_brand for alias in aliases)
        db_match = any(alias in db_brand for alias in aliases)
        if detected_match and db_match:
            return True

    return False


def get_judgment_emoji(status):
    """获取状态对应的emoji"""
    return {
        'normal': '✅',
        'suspicious': '⚠️',
        'not_found': '❌',
    }.get(status, '❓')


if __name__ == '__main__':
    from db_loader import load_vehicle_database

    db = load_vehicle_database('vehicle-database.csv')

    # 测试正常车辆
    result = compare_features('鲁B62B23', '江淮卡车', '蓝色', db)
    print(f"[测试] 鲁B62B23: {result}")

    # 测试套牌嫌疑（品牌不符）
    result = compare_features('鲁B62B23', '宝马', '黑色', db)
    print(f"[测试] 鲁B62B23(宝马): {result}")

    # 测试不存在的车牌
    result = compare_features('鲁A00000', '丰田', '白色', db)
    print(f"[测试] 鲁A00000: {result}")
