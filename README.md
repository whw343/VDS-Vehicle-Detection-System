# 基于车辆特征分析的套牌车稽查系统

> 人工智能基础课程大作业（2025-2026 第二学期）
> 广西科技大学 · 软件工程

## 项目简介

通过分析交通卡口摄像头拍摄的过车图片，自动提取车辆结构化特征（车牌号、车辆品牌、车辆颜色、车辆类型），与车管所登记信息库比对，快速识别套牌嫌疑车辆。

## 技术栈

- **深度学习**：YOLOv5（车辆检测）+ ResNet/VGG（车辆分类）
- **车牌OCR**：EasyOCR / PaddleOCR
- **Web后端**：Flask + Jinja2
- **Web前端**：Bootstrap
- **数据处理**：Pandas + OpenCV

## 项目结构

```
├── detect.py           # YOLOv5 车辆检测模块
├── plate_recognize.py  # 车牌 OCR 识别模块
├── color_classify.py   # 车辆颜色分类模块
├── brand_classify.py   # 车辆品牌分类模块
├── compare.py          # 特征比对与套牌判定
├── inference.py        # 统一推理接口
├── app.py              # Flask Web 应用
├── vehicle-database.csv # 车辆登记数据库
├── requirements.txt    # Python 依赖
├── templates/          # HTML 模板
│   ├── base.html
│   └── index.html
├── static/             # 静态资源
│   ├── css/
│   └── js/
├── data/               # 数据集（不提交到GitHub）
│   ├── train/
│   └── test/
├── weights/            # 模型权重（不提交到GitHub）
│   ├── yolov5m.pt
│   └── best.pt
└── docs/               # 项目文档
    ├── 项目完整分析文档.md
    └── 分工计划.md
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型权重
bash weights/download_weights.sh

# 3. 运行Web应用
python app.py
```

## 小组成员

| 姓名 | 角色 | 职责 |
|------|------|------|
| 满瀚宇（组长） | Web开发 + 系统集成 | Flask后端、数据库、前端页面 |
| 宋柄儒 | 深度学习工程师 | YOLOv5车辆检测、车辆类型识别 |
| 肖岱彤 | 图像处理工程师 | 车牌OCR、颜色识别、多特征融合 |

## 开发周期

- 5/3 — 5/4：环境搭建与数据准备
- 5/5 — 5/9：核心模块开发
- 5/10 — 5/12：系统集成与测试
- 5/13 — 5/15：收尾与交付

## 数据集说明

- `vehicle-database.csv`：车管所车辆登记库（6736条，字段：plateNo, carBrand）
- 训练数据见 `data/` 目录（需从课程平台下载）
