"""
车辆数据库加载器
从 vehicle-database.csv 加载车辆登记信息

字段：plateNo（车牌号）, carBrand（车辆品牌）
"""

import pandas as pd


def load_vehicle_database(csv_path='vehicle-database.csv'):
    """
    加载车辆数据库

    Args:
        csv_path: CSV文件路径

    Returns:
        DataFrame: 车辆登记数据
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        # 清理字段名中的BOM字符
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')
        print(f"[数据库] 加载成功，共 {len(df)} 条记录")
        print(f"[数据库] 字段：{list(df.columns)}")
        print(f"[数据库] 车型种类：{df['carBrand'].nunique()} 种")
        return df
    except FileNotFoundError:
        print(f"[数据库] 文件不存在：{csv_path}")
        return pd.DataFrame(columns=['plateNo', 'carBrand'])
    except Exception as e:
        print(f"[数据库] 加载失败：{e}")
        return pd.DataFrame(columns=['plateNo', 'carBrand'])


def query_vehicle(plate_number, db):
    """
    根据车牌号查询车辆信息

    Args:
        plate_number: 车牌号（如：鲁B62B23）
        db: 车辆数据库 DataFrame

    Returns:
        dict 或 None: 匹配的车辆信息
    """
    result = db[db['plateNo'] == plate_number]
    if result.empty:
        return None
    return result.iloc[0].to_dict()


def get_brand_list(db):
    """获取所有车辆品牌列表"""
    return sorted(db['carBrand'].unique().tolist())


if __name__ == '__main__':
    # 测试
    db = load_vehicle_database()
    print(f"\n[测试] 前5条记录：")
    print(db.head())
    print(f"\n[测试] 查询 鲁B62B23:")
    result = query_vehicle('鲁B62B23', db)
    print(result)
