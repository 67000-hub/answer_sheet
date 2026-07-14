# 答题卡识别系统

基于 Python + OpenCV 的答题卡自动识别与判分系统，支持**单选、4选项、多列布局**的答题卡，并支持**主观题 OCR 文字识别**。

## 功能特性

- 自动检测答题卡外框并透视变换矫正倾斜
- 自适应识别圆形气泡选项（支持带字母的空心气泡）
- 自动识别多列布局（1 列、2 列、3 列等）
- 相对阈值判断填涂状态，不受光照影响
- 未填涂题目自动识别（不画圈、不计分）
- 图形化展示识别结果，终端打印答案与分数
- **主观题 OCR 识别**：手动框选主观题区域，PaddleOCR 自动转文字
- **两种模式**：仅客观题（全图自动检测） / 客观+主观（手动框选区域）

## 环境依赖

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy
- imutils
- tkinter（Python 内置）
- **可选**：PaddleOCR + PaddlePaddle（主观题 OCR 识别需要）

### 安装依赖

```bash
pip install numpy opencv-python imutils
```

### 安装 OCR 依赖（主观题识别用）

```bash
pip install rapidocr-onnxruntime
```

> 推荐使用 **RapidOCR**（ONNX Runtime 版本），兼容性好，安装简单。
> 也支持 PaddleOCR：`pip install paddleocr paddlepaddle`（Python 3.12 及以下）。
> 第一次运行时会自动下载模型文件。

## 使用方法

### 运行程序

```bash
python get_answer.py
```

### 操作步骤

1. **选择图片**：运行后弹出文件选择对话框，选择答题卡图片
2. **输入答案**：弹出输入框，填写每题正确答案（用逗号或空格分隔）
   - 示例：`B,D,A,C,B` 或 `B D A C B`
3. **选择模式**：
   - **仅客观题**：自动检测全图答题卡，识别客观题
   - **客观+主观**：
     1. 输入主观题数量
     2. 依次用鼠标拖拽框选每道主观题区域（按回车确认）
     3. 框选客观题区域（按回车确认）
4. **查看结果**：
   - 依次弹出各步骤的调试窗口（按任意键继续）
   - 最终 `Exam` 窗口展示判分结果
   - 终端输出主观题识别文字、学生答案、正确答案和得分

### 框选操作说明

- **鼠标左键按下拖动**：画出选框
- **回车键**：确认当前选择
- **R 键**：重新选择
- **C 键**：清除当前选择
- **ESC 键**：取消选择

### 识别结果说明

- **蓝色圈**：学生填涂的选项
- **绿色圈**：答对的正确答案
- **红色圈**：答错的正确答案
- 未填涂的题目不画蓝色圈

## 支持的答题卡格式

| 项目 | 说明 |
|------|------|
| 题型 | 单项选择题 |
| 选项数 | 4 个（A/B/C/D） |
| 布局 | 单列、多列均可（自动检测列数） |
| 气泡样式 | 实心填涂 / 空心带字母均可 |

## 识别原理

### 整体流程

```
答题卡照片
    ↓
灰度化 + 高斯模糊 + Canny边缘检测
    ↓
找最大四边形轮廓 → 透视变换矫正
    ↓
Otsu 自动阈值二值化（反色：填涂=白）
    ↓
形态学闭运算（补全圆环断裂）
    ↓
RETR_TREE 层级轮廓检测 → 动态尺寸筛选 → 中心去重
    ↓
按 x 坐标分列（自动检测列数）
    ↓
每列内 top-to-bottom 排序 → 每 4 个一组 = 一题
    ↓
每组内 left-to-right 排序 → A/B/C/D
    ↓
圆形掩膜测填涂量 → 相对比较判定填涂
    ↓
对比正确答案 → 评分 + 可视化输出
```

### 关键算法

1. **透视变换**：用 `approxPolyDP` 近似四边形，`getPerspectiveTransform` 做单应性变换，将倾斜拍摄的答题卡拉正
2. **气泡检测**：形态学闭运算补全断裂 + RETR_TREE 层级检测 + 动态尺寸筛选（基于宽度中位数）+ 中心去重
3. **多列布局**：根据气泡 x 坐标间距自动分列，列内排序分组后按列拼接
4. **填涂判定**：同一题内比较各选项的填涂像素量，最高值明显高于其他平均值（1.3 倍 + 50 像素）则判为填涂

## 项目结构

```
answer_sheet/
├── get_answer.py         # 主入口（GUI + 主流程）
├── config.py             # 配置常量（阈值、颜色、选项字母等）
├── utils/                # 工具函数
│   ├── __init__.py
│   ├── geometry.py       # 几何变换（四点排序、透视变换）
│   ├── contour_utils.py  # 轮廓工具（排序、气泡填涂测量）
│   ├── viz.py            # 可视化工具（cv_show）
│   ├── roi_selector.py   # ROI 框选工具（鼠标拖拽选择）
│   └── ocr.py            # OCR 文字识别（PaddleOCR 封装）
├── detector/             # 检测核心
│   ├── __init__.py
│   ├── sheet_detector.py # 答题卡外框检测 + 透视矫正
│   ├── bubble_detector.py# 气泡检测 + 筛选 + 去重
│   ├── layout.py         # 多列布局聚类分组
│   ├── grading.py        # 填涂判定 + 评分 + 结果可视化
│   └── subjective.py     # 主观题处理（框选 + OCR）
├── images/               # 测试图片目录
└── README.md
```

## 主要模块

| 模块 | 说明 |
|------|------|
| `config.py` | 集中管理所有阈值、颜色、选项数等配置常量 |
| `utils/geometry.py` | `order_points` 四点排序、`four_point_transform` 透视变换 |
| `utils/contour_utils.py` | `sort_contours` 轮廓排序、`get_bubble_fill` 气泡填涂测量 |
| `utils/roi_selector.py` | `ROISelector` 鼠标拖拽框选 ROI 工具 |
| `utils/ocr.py` | `OCRRecognizer` PaddleOCR 文字识别封装 |
| `detector/sheet_detector.py` | 答题卡外框检测与矫正 |
| `detector/bubble_detector.py` | 气泡轮廓检测、动态尺寸筛选、中心去重 |
| `detector/layout.py` | 多列布局自动聚类分组 |
| `detector/grading.py` | 填涂判定、评分、结果可视化 |
| `detector/subjective.py` | 主观题处理：模式选择、多题框选、OCR 识别 |

## 常见问题

### Q: 检测到的题目数不对？
A: 检查 `columns_debug` 窗口，看分列是否正确。如果列间距阈值不合适，可以调整 `col_gap_threshold = avg_w * 3` 中的系数。

### Q: 填涂识别不准？
A: 看终端的 `[DEBUG]` 输出，检查每题的 totals 值。如果最大值和其他值差距不大，可以调小判定系数（1.3 改为更小的值）。

### Q: 气泡漏检？
A: 检查 `thresh_closed` 和 `all_contours` 窗口。形态学闭运算的核大小（当前 3×3）或尺寸筛选范围（0.6~1.5 倍中位数）可以适当放宽。

### Q: 主观题 OCR 识别不准？
A: 确保框选区域只包含该题文字，不要包含太多空白区域。手写体识别效果可能不佳，建议用于印刷体题目。

### Q: 提示未安装 OCR 引擎？
A: 执行 `pip install rapidocr-onnxruntime` 安装（推荐，兼容性好）。
也可以安装 PaddleOCR：`pip install paddleocr paddlepaddle`。
如果不需要主观题识别功能，选择"仅客观题"模式即可。

## 许可证

MIT License
