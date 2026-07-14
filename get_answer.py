import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog

from config import ANSWER_LETTERS, NUM_OPTIONS
from detector import (
    find_sheet, warp_sheet, detect_bubbles,
    cluster_into_columns, grade_answers, visualize_results,
    ask_mode, ask_subjective_count,
    process_subjective_questions, select_objective_region,
)
from utils import crop_roi


def select_image():
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(
        title="选择答题卡图片",
        filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
    )
    if not image_path:
        print("[ERROR] 未选择图片，程序退出")
        exit()
    return image_path


def input_answers():
    root = tk.Tk()
    root.withdraw()
    answer_input = simpledialog.askstring(
        "输入正确答案",
        f"请输入每题正确答案（用{'/'.join(ANSWER_LETTERS)}表示，逗号或空格分隔）\n例如: B,D,A,C",
        parent=root,
    )
    if not answer_input:
        print("[ERROR] 未输入答案，程序退出")
        exit()

    answers = [a.strip().upper() for a in answer_input.replace(",", " ").split()]
    answer_key = {}
    for i, a in enumerate(answers):
        if a not in ANSWER_LETTERS:
            print(f"[ERROR] 无效答案 '{a}'，有效选项为 {ANSWER_LETTERS}")
            exit()
        answer_key[i] = ANSWER_LETTERS.index(a)

    return answer_key, len(answer_key)


def process_objective_full(image, answer_key, num_questions, debug=True):
    docCnt, gray = find_sheet(image, debug=debug)
    if docCnt is None:
        print("[ERROR] 未检测到答题卡外框")
        return None

    warped = warp_sheet(gray, docCnt, debug=debug)
    questionCnts, thresh = detect_bubbles(warped, debug=debug)

    all_questions = cluster_into_columns(
        questionCnts, num_options=NUM_OPTIONS,
        warped_img=warped, debug=debug,
    )

    detected_questions = len(all_questions)
    if detected_questions != num_questions:
        print(f"[WARNING] 检测到 {detected_questions} 题，"
              f"但输入了 {num_questions} 个答案")

    student_answers, correct, score = grade_answers(
        all_questions, thresh, answer_key,
        ANSWER_LETTERS, num_questions, debug=debug,
    )

    return warped, thresh, all_questions, student_answers, correct, score


def process_objective_roi(image, roi, answer_key, num_questions, debug=True):
    cropped = crop_roi(image, roi)
    if cropped is None or cropped.size == 0:
        print("[ERROR] 客观题区域无效")
        return None

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped

    warped = gray
    if debug:
        from utils.viz import cv_show
        cv_show('objective_roi', cropped)

    questionCnts, thresh = detect_bubbles(warped, debug=debug)

    all_questions = cluster_into_columns(
        questionCnts, num_options=NUM_OPTIONS,
        warped_img=warped, debug=debug,
    )

    detected_questions = len(all_questions)
    if detected_questions != num_questions:
        print(f"[WARNING] 检测到 {detected_questions} 题，"
              f"但输入了 {num_questions} 个答案")

    student_answers, correct, score = grade_answers(
        all_questions, thresh, answer_key,
        ANSWER_LETTERS, num_questions, debug=debug,
    )

    return warped, thresh, all_questions, student_answers, correct, score


def main():
    image_path = select_image()
    answer_key, num_questions = input_answers()

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] 无法读取图片: {image_path}")
        exit()

    has_subjective = ask_mode()

    if has_subjective:
        print("[INFO] 模式：客观题 + 主观题")

        num_subjective = ask_subjective_count()
        print(f"[INFO] 主观题数量: {num_subjective}")

        if num_subjective > 0:
            print("\n" + "=" * 60)
            print("开始处理主观题")
            print("=" * 60)
            subjective_results = process_subjective_questions(image, num_subjective)
        else:
            subjective_results = []

        print("\n" + "=" * 60)
        print("请框选客观题区域")
        print("=" * 60)
        objective_roi = select_objective_region(image)
        if objective_roi is None:
            print("[ERROR] 未选择客观题区域，程序退出")
            exit()

        result = process_objective_roi(
            image, objective_roi, answer_key, num_questions, debug=True,
        )
        if result is None:
            exit()

        warped, thresh, all_questions, student_answers, correct, score = result

        warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        result_img = visualize_results(
            warped_bgr, all_questions, thresh, answer_key,
            ANSWER_LETTERS, num_questions, score,
        )

        print("\n" + "=" * 60)
        print("【主观题识别结果】")
        print("=" * 60)
        for res in subjective_results:
            idx = res["index"]
            if res["text"]:
                text = res["text"]
            elif res["roi"] is None:
                text = "(未选择区域)"
            else:
                text = "(OCR 未安装，跳过文字识别)"
            print(f"\n--- 主观题 {idx} ---")
            print(text)

        print("\n" + "=" * 60)
        print("【客观题结果】")
        print("=" * 60)
        print("[INFO] 学生答案:", " ".join(student_answers))
        print("[INFO] 正确答案:",
              " ".join([ANSWER_LETTERS[answer_key[q]] for q in range(num_questions)]))
        print("[INFO] score: {:.2f}%".format(score))

        cv2.imshow("Original", image)
        cv2.imshow("Exam", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("[INFO] 模式：仅客观题（全图自动检测）")
        result = process_objective_full(image, answer_key, num_questions, debug=True)
        if result is None:
            exit()

        warped, thresh, all_questions, student_answers, correct, score = result

        print("[INFO] 学生答案:", " ".join(student_answers))
        print("[INFO] 正确答案:",
              " ".join([ANSWER_LETTERS[answer_key[q]] for q in range(num_questions)]))
        print("[INFO] score: {:.2f}%".format(score))

        warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        result_img = visualize_results(
            warped_bgr, all_questions, thresh, answer_key,
            ANSWER_LETTERS, num_questions, score,
        )

        cv2.imshow("Original", image)
        cv2.imshow("Exam", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
