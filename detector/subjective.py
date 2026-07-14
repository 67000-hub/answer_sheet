import tkinter as tk
from tkinter import simpledialog, messagebox

from utils.roi_selector import select_roi_interactive, crop_roi
from utils.ocr import ocr_recognize, ocr_available


def ask_subjective_count():
    root = tk.Tk()
    root.withdraw()
    result = simpledialog.askinteger(
        "主观题数量",
        "请输入主观题数量：",
        minvalue=0,
        maxvalue=100,
        parent=root,
    )
    if result is None:
        return 0
    return result


def ask_mode():
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno(
        "选择模式",
        "是否包含主观题？\n\n"
        "是：手动框选主观题 + 客观题区域\n"
        "否：仅自动识别客观题（全图检测）",
    )
    return result


def process_subjective_questions(image, num_questions):
    if num_questions <= 0:
        return []

    has_ocr = ocr_available()
    if not has_ocr:
        print("[WARNING] OCR 不可用，仅框选区域，不进行文字识别")
        print("         安装命令: pip install paddleocr paddlepaddle")

    results = []
    for i in range(num_questions):
        prompt = f"第 {i+1}/{num_questions} 道主观题\n框选后按回车确认"
        roi = select_roi_interactive(
            image,
            title=f"主观题 {i+1}",
            prompt=prompt,
        )
        if roi is None:
            print(f"[INFO] 主观题 {i+1} 未选择，跳过")
            results.append({"index": i + 1, "roi": None, "text": None})
            continue

        cropped = crop_roi(image, roi)
        if cropped is None or cropped.size == 0:
            print(f"[WARNING] 主观题 {i+1} 区域无效")
            results.append({"index": i + 1, "roi": roi, "text": None})
            continue

        text = None
        if has_ocr:
            print(f"[INFO] 正在识别主观题 {i+1} ...")
            text = ocr_recognize(cropped)

        results.append({"index": i + 1, "roi": roi, "text": text})

        print(f"\n{'='*50}")
        print(f"【主观题 {i+1}】")
        print(f"{'='*50}")
        if text:
            print(text)
        elif has_ocr:
            print("(未识别到文字)")
        else:
            print("(OCR 未安装，跳过文字识别)")
        print(f"{'='*50}\n")

    return results


def select_objective_region(image):
    prompt = "框选客观题区域，按回车确认"
    roi = select_roi_interactive(
        image,
        title="选择客观题区域",
        prompt=prompt,
    )
    if roi is None:
        print("[WARNING] 未选择客观题区域")
    return roi
