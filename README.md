# 基于车辆特征分析的套牌车稽查系统

> 广西科技大学 · 软件工程 · 2025-2026第二学期 · 人工智能基础课程项目

## 📋 项目简介

本项目构建一个**基于人工智能的套牌车稽查系统**，通过分析交通卡口摄像头拍摄的过车图片，自动提取车辆结构化特征（车牌号、车辆品牌、车辆颜色、车辆类型），与车管所登记信息库比对，快速识别套牌嫌疑车辆。

## 👥 小组成员

| 成员 | 角色 | 职责 |
|------|------|------|
| 满瀚宇（组长） | Web开发 + 系统集成 | Flask后端、前端页面、数据库、项目管理 |
| 宋柄儒 | 深度学习 + 车辆检测 | YOLOv5检测、车型分类、品牌分类、模型训练 |
| 肖岱彤 | 特征识别 + 文档 | 车牌OCR、颜色分类、测试验证、文档整理 |

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- CUDA >= 11.8（GPU加速，可选）
- Windows / Linux

### 安装依赖

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics easyocr opencv-python flask pandas pillow lxml
```

### 准备模型权重

将以下权重文件放入 `weights/` 目录：

| 文件 | 说明 | 来源 |
|------|------|------|
| `yolov5su.pt` | YOLOv5车辆检测 | Ultralytics |
| `color_model.pth` | 颜色分类(ResNet18) | 训练产出 |
| `type_model.pth` | 车型分类(ResNet18) | 训练产出 |
| `brand_model.pth` | 品牌分类(ResNet18) | 训练产出 |
| `brand_labels.txt` | 品牌标签列表 | 训练产出 |

```bash
# 也可使用YOLOv5官方权重
cp阶段三-任务11-weights/yolov5s.pt weights/
```

### 启动系统

```bash
cd VDS-Vehicle-Detection-System
python app.py
```

访问 http://localhost:5000

## 🏗️ 系统架构

```
用户 → Web前端(Bootstrap+Jinja2) → Flask后端
                                        ↓
                              ┌─────────┼─────────┐
                              ↓         ↓         ↓
                          YOLOv5     EasyOCR    ResNet18
                          车辆检测    车牌OCR    颜色/车型/品牌分类
                              └─────────┼─────────┘
                                        ↓
                              数据库比对(compare.py)
                                        ↓
                               vehicle-database.csv
                              (3581条登记记录)
```

## 📁 项目结构

```
VDS-Vehicle-Detection-System/
├── app.py                  # Flask Web应用入口
├── detect.py               # YOLOv5车辆检测模块
├── plate_recognize.py      # 车牌OCR识别模块
├── color_classify.py       # 车辆颜色分类模块
├── type_classify.py        # 车辆类型分类模块
├── brand_classify.py       # 车辆品牌分类模块
├── compare.py              # 特征比对与套牌判定
├── db_loader.py            # 数据库加载器
├── inference.py            # 统一推理接口
├── templates/              # Web前端模板
│   ├── base.html           # 基础模板
│   ├── index.html          # 上传页面
│   └── result.html         # 结果展示页
├── static/uploads/         # 上传图片存储
├── weights/                # 模型权重
├── data/                   # 训练和测试数据
├── docs/                   # 文档和测试报告
└── vehicle-database.csv    # 车辆登记数据库
```

## 🔧 核心模块

### 1. 车辆检测 (detect.py)

使用 **YOLOv5** 检测图片中的车辆，输出边界框和裁剪图。

```python
from detect import VehicleDetector
detector = VehicleDetector()
detector.load_model()
detections = detector.detect('test.jpg')
```

### 2. 车牌识别 (plate_recognize.py)

使用 **EasyOCR** 识别车牌号，支持蓝牌/黄牌/绿牌区域检测。

```python
from plate_recognize import PlateRecognizer
ocr = PlateRecognizer()
ocr.load_model()
plate = ocr.recognize(image)  # → "鲁B62B23"
```

### 3. 特征分类

| 模块 | 类别数 | 准确率 | 技术 |
|------|--------|--------|------|
| color_classify.py | 8色 | 94.3% | ResNet18 |
| type_classify.py | 4类 | 91.2% | ResNet18 |
| brand_classify.py | 47品牌 | - | ResNet18 |

### 4. 数据库比对 (compare.py)

多因素评分系统（0-100分）：
- 车牌精确匹配/模糊匹配
- 品牌关键词+别名匹配
- 综合评分判定套牌嫌疑

## 📊 API接口

| 路由 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 系统首页 |
| `/upload` | POST | 上传图片并分析 |
| `/api/analyze` | POST | JSON API分析 |
| `/api/health` | GET | 健康检查 |
| `/api/db/stats` | GET | 数据库统计 |
| `/result` | GET | 结果展示页 |

## 📈 测试结果

| 指标 | 结果 |
|------|------|
| YOLOv5检测率 | 95% (95/100) |
| 单张平均耗时 | 1.62秒 |
| 颜色分类准确率 | 94.3% |
| 车型分类准确率 | 91.2% |

详见 `docs/测试报告.md`

## 📝 训练模型

```bash
# 颜色分类
python train_color_model.py

# 车型分类
python train_type_model.py

# 品牌分类(需要阶段三数据)
python -c "exec(open('train_all_real.py').read())"
```

## ⚠️ 注意事项

1. 模型权重文件需手动放入 `weights/` 目录（被.gitignore排除）
2. OCR在卡口全景图上识别率有限（约25%），建议配合车牌检测模型使用
3. GPU加速显著提升YOLOv5推理速度
4. 数据库 `vehicle-database.csv` 需保持UTF-8编码

## 📄 许可证

本项目为广西科技大学人工智能基础课程作业，仅供学习交流使用。
