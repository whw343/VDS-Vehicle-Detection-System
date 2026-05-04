"""
车辆数据库加载器
从 vehicle-database.csv 加载车辆登记信息

字段：plateNo（车牌号）, carBrand（车辆品牌）

负责人：满瀚宇
"""

import os
import pandas as pd


# 全局缓存
_cached_db = None
_cached_stats = None


def load_vehicle_database(csv_path='vehicle-database.csv', force_reload=False):
    """
    加载车辆数据库（带缓存）

    Args:
        csv_path: CSV文件路径
        force_reload: 是否强制重新加载

    Returns:
        DataFrame: 车辆登记数据
    """
    global _cached_db, _cached_stats

    if _cached_db is not None and not force_reload:
        return _cached_db

    if not os.path.exists(csv_path):
        print(f"[数据库] 文件不存在：{csv_path}")
        return pd.DataFrame(columns=['plateNo', 'carBrand'])

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        # 数据清洗
        df = _clean_database(df)

        _cached_db = df
        _cached_stats = None  # 重新计算统计

        print(f"[数据库] 加载成功，共 {len(df)} 条记录")
        print(f"[数据库] 字段：{list(df.columns)}")
        print(f"[数据库] 车型种类：{df['carBrand'].nunique()} 种")

        return df

    except Exception as e:
        print(f"[数据库] 加载失败：{e}")
        return pd.DataFrame(columns=['plateNo', 'carBrand'])


def _clean_database(df):
    """清洗数据库"""
    # 去除重复车牌
    before = len(df)
    df = df.drop_duplicates(subset=['plateNo'], keep='first')
    after = len(df)
    if before > after:
        print(f"[数据库] 移除 {before - after} 条重复记录")

    # 去除空值
    df = df.dropna(subset=['plateNo', 'carBrand'])
    print(f"[数据库] 清洗后记录数: {len(df)}")

    return df


def query_vehicle(plate_number, db=None):
    """
    根据车牌号精确查询

    Args:
        plate_number: 车牌号
        db: 数据库DataFrame（None则使用缓存）

    Returns:
        dict or None
    """
    if db is None:
        db = _cached_db

    if db is None or db.empty:
        return None

    result = db[db['plateNo'] == plate_number]
    if result.empty:
        return None

    return result.iloc[0].to_dict()


def fuzzy_search_plate(plate_number, db=None, min_similarity=0.6):
    """
    模糊搜索车牌号（处理OCR识别错误）

    策略：
    1. 精确匹配
    2. 去掉省份简称后匹配（处理省份误识别）
    3. 字符级编辑距离匹配

    Args:
        plate_number: 识别出的车牌号
        db: 数据库DataFrame
        min_similarity: 最小相似度阈值

    Returns:
        list[dict]: 匹配结果列表，按相似度降序
    """
    if db is None:
        db = _cached_db

    if db is None or db.empty or not plate_number:
        return []

    results = []

    # 1. 精确匹配
    exact = db[db['plateNo'] == plate_number]
    if not exact.empty:
        return [{'plateNo': r['plateNo'], 'carBrand': r['carBrand'],
                 'similarity': 1.0, 'match_type': 'exact'}
                for _, r in exact.iterrows()]

    # 2. 去掉省份简称匹配（如 "鲁B62B23" → "B62B23"）
    if len(plate_number) >= 2:
        tail = plate_number[1:]
        partial = db[db['plateNo'].str.endswith(tail, na=False)]
        for _, row in partial.iterrows():
            results.append({
                'plateNo': row['plateNo'],
                'carBrand': row['carBrand'],
                'similarity': 0.9,
                'match_type': 'province_diff'
            })

    # 3. 字符相似度匹配（忽略省份，比较后续字符）
    if len(plate_number) >= 6:
        candidates = db[db['plateNo'].str.len() == len(plate_number)]
        for _, row in candidates.iterrows():
            db_plate = row['plateNo']
            sim = _char_similarity(plate_number[1:], db_plate[1:])
            if sim >= min_similarity:
                results.append({
                    'plateNo': db_plate,
                    'carBrand': row['carBrand'],
                    'similarity': sim,
                    'match_type': 'fuzzy'
                })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:5]  # 最多返回5条


def _char_similarity(s1, s2):
    """字符相似度（忽略大小写）"""
    s1, s2 = s1.upper(), s2.upper()
    if s1 == s2:
        return 1.0

    n = min(len(s1), len(s2))
    matches = sum(1 for i in range(n) if s1[i] == s2[i])
    return matches / max(len(s1), len(s2))


def get_brand_list(db=None):
    """获取所有车辆品牌列表"""
    if db is None:
        db = _cached_db
    if db is None or db.empty:
        return []
    return sorted(db['carBrand'].unique().tolist())


def get_database_stats(db=None):
    """获取数据库统计信息"""
    global _cached_stats

    if db is None:
        db = _cached_db

    if db is None or db.empty:
        return {'total_records': 0}

    if _cached_stats is not None:
        return _cached_stats

    stats = {
        'total_records': len(db),
        'unique_brands': db['carBrand'].nunique(),
        'top_brands': db['carBrand'].value_counts().head(10).to_dict(),
        'plate_prefixes': db['plateNo'].str[:2].value_counts().head(10).to_dict(),
    }

    _cached_stats = stats
    return stats


def reload_database(csv_path='vehicle-database.csv'):
    """强制重新加载数据库"""
    global _cached_db
    _cached_db = None
    return load_vehicle_database(csv_path, force_reload=True)


# ── 测试入口 ───────────────────────────────────

if __name__ == '__main__':
    db = load_vehicle_database()

    print(f"\n📊 数据库统计:")
    stats = get_database_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(f"\n🔍 精确查询测试:")
    result = query_vehicle('鲁B62B23')
    print(f"  鲁B62B23 → {result}")

    print(f"\n🔍 模糊搜索测试:")
    results = fuzzy_search_plate('B62B23')
    for r in results:
        print(f"  {r['plateNo']} ({r['carBrand']}) 相似度: {r['similarity']:.0%} [{r['match_type']}]")
