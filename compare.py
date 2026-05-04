"""
套牌比对模块
将AI提取的车辆特征与数据库比对，判定是否套牌

判定逻辑：
1. 车牌精确匹配 → 数据库查询
2. 车牌模糊搜索（处理OCR误差）
3. 品牌比对（关键词+别名匹配）
4. 综合评分判定

输入：车牌号、车辆品牌、车辆颜色
输出：判定结果（含置信度评分）

负责人：满瀚宇
"""

import pandas as pd
from db_loader import fuzzy_search_plate


# 品牌别名映射（处理AI识别差异）
BRAND_ALIASES = {
    '大众': ['大众出租车', '大众家用车', '大众'],
    '丰田': ['丰田'],
    '本田': ['本田'],
    '日产': ['日产', '日产越野车'],
    '现代': ['现代', '现代越野车', '现代面包车'],
    '起亚': ['起亚'],
    '宝马': ['宝马'],
    '奔驰': ['奔驰'],
    '奥迪': ['奥迪'],
    '福特': ['福特', '福特面包车'],
    '雪佛兰': ['雪佛兰'],
    '别克': ['别克', '别克商务车'],
    '比亚迪': ['比亚迪'],
    '长安': ['长安', '长安面包车'],
    '长城': ['长城'],
    '江淮': ['江淮', '江淮卡车', '江淮面包车'],
    '五菱': ['五菱'],
    '东风': ['东风', '东风面包车', '东风卡车'],
    '福田': ['福田', '福田大卡', '福田小卡'],
    '金杯': ['金杯卡车', '金杯面包车'],
    '一汽': ['一汽', '一汽卡车', '一汽面包车'],
    '马自达': ['马自达'],
    '雪铁龙': ['雪铁龙'],
    '标致': ['标志'],
    '标志': ['标志'],
    '铃木': ['铃木'],
    '中华': ['中华'],
    '荣威': ['荣威'],
    '菲亚特': ['菲亚特'],
    '宝骏': ['宝骏'],
    '海马': ['海马'],
    '双龙': ['双龙'],
    '华泰': ['华泰'],
    '绅宝': ['绅宝'],
    '昌河': ['昌河'],
    '北汽': ['北汽'],
    '大宇': ['大宇'],
    '哈飞': ['哈飞'],
    '江铃': ['江铃'],
    '吉利': ['吉利'],
}


def compare_features(plate_number, detected_brand, detected_color, vehicle_db):
    """
    特征比对主函数

    Args:
        plate_number: OCR识别的车牌号
        detected_brand: AI识别的品牌
        detected_color: AI识别的颜色
        vehicle_db: 车辆数据库 DataFrame

    Returns:
        dict: 比对结果
    """
    result = {
        'status': 'unknown',
        'message': '',
        'score': 0,
        'details': {
            'plate_match': False,
            'brand_match': False,
            'color_available': bool(detected_color and detected_color != '未知'),
            'db_record': None,
            'fuzzy_candidates': [],
            'match_type': 'none'
        }
    }

    # 基础校验
    if vehicle_db is None or vehicle_db.empty:
        result['status'] = 'error'
        result['message'] = '数据库未加载'
        return result

    if not plate_number or plate_number == '未识别':
        result['status'] = 'error'
        result['message'] = '未识别出车牌号，无法比对'
        return result

    # 1. 车牌查询（精确 + 模糊）
    exact_match = vehicle_db[vehicle_db['plateNo'] == plate_number]

    if not exact_match.empty:
        # 精确匹配
        db_record = exact_match.iloc[0].to_dict()
        result['details']['plate_match'] = True
        result['details']['db_record'] = db_record
        result['details']['match_type'] = 'exact'

        db_brand = db_record.get('carBrand', '')
        brand_match, brand_score = _compare_brand_scored(detected_brand, db_brand)
        result['details']['brand_match'] = brand_match

        # 综合评分
        result['score'] = _calculate_score(
            plate_match=True,
            brand_match=brand_match,
            brand_score=brand_score
        )

        if not brand_match:
            result['status'] = 'suspicious'
            result['message'] = (
                f'⚠️ 套牌嫌疑：车牌 {plate_number} 在登记库中存在，'
                f'但品牌不符（识别：{detected_brand}，登记：{db_brand}）'
            )
        else:
            result['status'] = 'normal'
            result['message'] = (
                f'✅ 正常车辆：车牌 {plate_number}，'
                f'品牌 {db_brand}，特征匹配'
            )

    else:
        # 精确匹配失败 → 模糊搜索
        fuzzy_results = fuzzy_search_plate(plate_number, vehicle_db)

        if fuzzy_results and fuzzy_results[0]['similarity'] >= 0.8:
            # 高相似度模糊匹配
            best = fuzzy_results[0]
            result['details']['fuzzy_candidates'] = fuzzy_results[:3]
            result['details']['db_record'] = best
            result['details']['match_type'] = 'fuzzy'
            result['details']['plate_match'] = True

            db_brand = best.get('carBrand', '')
            brand_match, brand_score = _compare_brand_scored(detected_brand, db_brand)
            result['details']['brand_match'] = brand_match

            result['score'] = _calculate_score(
                plate_match=True,
                brand_match=brand_match,
                brand_score=brand_score
            ) * 0.9  # 模糊匹配扣分

            if brand_match:
                result['status'] = 'normal'
                result['message'] = (
                    f'✅ 车牌 {plate_number} 近似匹配 {best["plateNo"]}'
                    f'（相似度 {best["similarity"]:.0%}），品牌 {db_brand}'
                )
            else:
                result['status'] = 'suspicious'
                result['message'] = (
                    f'⚠️ 套牌嫌疑：车牌 {plate_number} 近似匹配 {best["plateNo"]}，'
                    f'但品牌不符（识别：{detected_brand}，登记：{db_brand}）'
                )
        else:
            # 未匹配
            result['status'] = 'suspicious'
            result['score'] = 10
            result['message'] = (
                f'⚠️ 套牌嫌疑：车牌 {plate_number} 不存在于登记库'
            )

            if fuzzy_results:
                result['details']['fuzzy_candidates'] = fuzzy_results[:3]

    return result


def _compare_brand_scored(detected_brand, db_brand):
    """
    品牌比对（带评分）

    Args:
        detected_brand: AI识别的品牌
        db_brand: 数据库中的品牌

    Returns:
        tuple: (是否匹配, 匹配分数 0-100)
    """
    if not detected_brand or not db_brand:
        return False, 0

    # 完全匹配
    if detected_brand == db_brand:
        return True, 100

    # 包含匹配
    if detected_brand in db_brand:
        return True, 90
    if db_brand in detected_brand:
        return True, 85

    # 别名匹配
    detected_key = _get_brand_key(detected_brand)
    db_key = _get_brand_key(db_brand)

    if detected_key and db_key and detected_key == db_key:
        return True, 80

    # 别名包含
    if detected_key and db_brand in BRAND_ALIASES.get(detected_key, []):
        return True, 75
    if db_key and detected_brand in BRAND_ALIASES.get(db_key, []):
        return True, 70

    # 部分字符串匹配
    short = min(len(detected_brand), len(db_brand))
    if short >= 2 and detected_brand[:2] == db_brand[:2]:
        return True, 50

    return False, 0


def _get_brand_key(brand_name):
    """获取品牌的主名称"""
    if not brand_name:
        return None

    for key, aliases in BRAND_ALIASES.items():
        if brand_name in aliases:
            return key
        if any(alias in brand_name for alias in aliases):
            return key

    return None


def _calculate_score(plate_match, brand_match, brand_score):
    """
    计算综合比对评分（0-100）

    评分组成：
    - 车牌匹配：40分
    - 品牌匹配：60分
    """
    score = 0

    if plate_match:
        score += 40
    else:
        score += 5  # 模糊匹配得5分

    if brand_match:
        score += brand_score * 0.6

    return min(score, 100)


def get_judgment_emoji(status):
    """获取状态emoji"""
    return {
        'normal': '✅',
        'suspicious': '⚠️',
        'error': '❌',
        'unknown': '❓',
    }.get(status, '❓')


def summarize_result(comparison):
    """
    生成简短的判定摘要（用于前端展示）

    Args:
        comparison: compare_features 返回的结果

    Returns:
        dict: 前端友好的摘要
    """
    emoji = get_judgment_emoji(comparison.get('status', 'unknown'))
    details = comparison.get('details', {})

    summary = {
        'emoji': emoji,
        'status': comparison.get('status', 'unknown'),
        'message': comparison.get('message', ''),
        'score': comparison.get('score', 0),
        'plate_found': details.get('plate_match', False),
        'brand_match': details.get('brand_match', False),
    }

    if details.get('db_record'):
        summary['registered_brand'] = details['db_record'].get('carBrand', '')
        summary['registered_plate'] = details['db_record'].get('plateNo', '')

    return summary


# ── 测试入口 ───────────────────────────────────

if __name__ == '__main__':
    from db_loader import load_vehicle_database

    db = load_vehicle_database('vehicle-database.csv')

    print("=" * 60)
    print("套牌比对模块测试")
    print("=" * 60)

    tests = [
        ('鲁B62B23', '江淮卡车', '蓝色', '正常车辆'),
        ('鲁B62B23', '宝马', '黑色', '品牌不符(套牌嫌疑)'),
        ('鲁A00000', '丰田', '白色', '车牌不存在(套牌嫌疑)'),
        ('鲁B62B23', '江淮', '蓝色', '品牌模糊匹配'),
    ]

    for plate, brand, color, desc in tests:
        print(f"\n--- 测试: {desc} ---")
        print(f"  输入: plate={plate}, brand={brand}, color={color}")
        result = compare_features(plate, brand, color, db)
        summary = summarize_result(result)
        print(f"  判定: {summary['emoji']} {summary['status']}")
        print(f"  评分: {summary['score']}")
        print(f"  信息: {summary['message'][:80]}")
