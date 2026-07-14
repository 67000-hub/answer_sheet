import tkinter as tk
from tkinter import simpledialog, messagebox

from utils.roi_selector import select_roi_interactive, crop_roi
from utils.ocr import ocr_recognize, ocr_available


def ask_subjective_count():
    root = tk.Tk()
    root.withdraw()
    result = simpledialog.askinteger(
        "Subjective Count",
        "Number of subjective questions:",
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
        "Select Mode",
        "Contains subjective questions?\n\n"
        "Yes: Manually select subjective + objective regions\n"
        "No: Auto-detect objective questions (full image)",
    )
    return result


def process_subjective_questions(image, num_questions):
    if num_questions <= 0:
        return []

    has_ocr = ocr_available()
    if not has_ocr:
        print("[WARNING] OCR not available, only select regions (no text recognition)")
        print("         Install: pip install rapidocr-onnxruntime")

    results = []
    for i in range(num_questions):
        prompt = f"Subjective Q{i+1}/{num_questions}\nDrag to select, Enter to add region, N to finish"
        rois = select_roi_interactive(
            image,
            title=f"Subjective Q{i+1}",
            prompt=prompt,
            allow_multiple=True,
        )

        if not rois:
            print(f"[INFO] Subjective Q{i+1} skipped (no region selected)")
            results.append({"index": i + 1, "rois": [], "texts": []})
            continue

        texts = []
        for j, roi in enumerate(rois):
            cropped = crop_roi(image, roi)
            if cropped is None or cropped.size == 0:
                print(f"[WARNING] Q{i+1} region #{j+1} invalid")
                texts.append(None)
                continue

            text = None
            if has_ocr:
                print(f"[INFO] Recognizing Q{i+1} region #{j+1} ...")
                text = ocr_recognize(cropped)
            texts.append(text)

        results.append({"index": i + 1, "rois": rois, "texts": texts})

        print(f"\n{'='*60}")
        print(f"【Subjective Q{i+1}】 ({len(rois)} region(s))")
        print(f"{'='*60}")
        for j, text in enumerate(texts):
            if text:
                print(f"\n--- Region #{j+1} ---")
                print(text)
        if not any(texts):
            if has_ocr:
                print("(No text recognized)")
            else:
                print("(OCR not installed, text recognition skipped)")
        print(f"{'='*60}\n")

    return results


def select_objective_region(image):
    prompt = "Select objective region, Enter to confirm"
    roi = select_roi_interactive(
        image,
        title="Select Objective Region",
        prompt=prompt,
    )
    if roi is None:
        print("[WARNING] No objective region selected")
    return roi
