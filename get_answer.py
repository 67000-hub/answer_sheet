import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

from config import ANSWER_LETTERS, NUM_OPTIONS, SHEET_TYPE_CIRCLE, SHEET_TYPE_RECT
from detector import (
    find_sheet, warp_sheet, detect_options,
    cluster_into_columns, grade_answers, visualize_results,
    ask_mode, ask_subjective_count,
    process_subjective_questions, select_objective_region,
    list_templates, load_template, create_template,
)
from utils import crop_roi
from utils.viz import resize_for_display
from utils.ocr import ocr_recognize, ocr_available


def select_image():
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(
        title="Select Answer Sheet Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
    )
    if not image_path:
        print("[ERROR] No image selected, exiting")
        exit()
    return image_path


def select_images():
    root = tk.Tk()
    root.withdraw()
    image_paths = filedialog.askopenfilenames(
        title="Select Answer Sheet Images",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
    )
    if not image_paths:
        print("[ERROR] No images selected, exiting")
        exit()
    return image_paths


def input_answers():
    root = tk.Tk()
    root.withdraw()
    answer_input = simpledialog.askstring(
        "Input Answers",
        f"Enter correct answers ({'/'.join(ANSWER_LETTERS)}, comma/space separated)\n"
        f"e.g. B,D,A,C",
        parent=root,
    )
    if not answer_input:
        print("[ERROR] No answers entered, exiting")
        exit()

    answers = [a.strip().upper() for a in answer_input.replace(",", " ").split()]
    answer_key = {}
    for i, a in enumerate(answers):
        if a not in ANSWER_LETTERS:
            print(f"[ERROR] Invalid answer '{a}', valid options: {ANSWER_LETTERS}")
            exit()
        answer_key[i] = ANSWER_LETTERS.index(a)

    return answer_key, len(answer_key)


def ask_run_mode():
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesnocancel(
        "Select Run Mode",
        "Run with template?\n\n"
        "Yes: Use existing template (auto-select regions)\n"
        "No: Manual mode (select regions each time)\n"
        "Cancel: Create new template",
    )
    return result


def ask_sheet_type():
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno(
        "Select Sheet Type",
        "Answer sheet type:\n\n"
        "Yes: Rectangular brackets [A][B][C][D]\n"
        "No: Circular bubbles",
    )
    return SHEET_TYPE_RECT if result else SHEET_TYPE_CIRCLE


def select_template():
    templates = list_templates()
    if not templates:
        print("[INFO] No templates found. Please create one first.")
        return None

    root = tk.Tk()
    root.withdraw()
    names = [f"{t['name']} ({t['num_objective_answers']}Q, {t['num_subjective']}subj) [{t['filename']}]"
             for t in templates]

    choice = simpledialog.askstring(
        "Select Template",
        "Available templates:\n" +
        "\n".join(f"  {i+1}. {n}" for i, n in enumerate(names)) +
        "\n\nEnter template number:",
        parent=root,
    )
    if not choice:
        return None
    try:
        idx = int(choice.strip()) - 1
        if 0 <= idx < len(templates):
            return load_template(templates[idx]["filename"])
    except ValueError:
        pass
    print("[ERROR] Invalid template selection")
    return None


def process_objective_full(image, answer_key, num_questions,
                           sheet_type=SHEET_TYPE_CIRCLE, debug=True):
    docCnt, gray = find_sheet(image, debug=debug)
    if docCnt is None:
        print("[ERROR] Answer sheet border not detected")
        return None

    warped = warp_sheet(gray, docCnt, debug=debug)
    questionCnts, thresh = detect_options(warped, sheet_type=sheet_type, debug=debug)

    all_questions = cluster_into_columns(
        questionCnts, num_options=NUM_OPTIONS,
        warped_img=warped, debug=debug,
    )

    detected_questions = len(all_questions)
    if detected_questions != num_questions:
        print(f"[WARNING] Detected {detected_questions} questions, "
              f"but {num_questions} answers entered")

    student_answers, correct, score = grade_answers(
        all_questions, thresh, answer_key,
        ANSWER_LETTERS, num_questions, debug=debug,
        shape_type=sheet_type,
    )

    return warped, thresh, all_questions, student_answers, correct, score


def process_objective_roi(image, roi, answer_key, num_questions,
                          sheet_type=SHEET_TYPE_CIRCLE, debug=True):
    cropped = crop_roi(image, roi)
    if cropped is None or cropped.size == 0:
        print("[ERROR] Objective region invalid")
        return None

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped

    warped = gray
    if debug:
        from utils.viz import cv_show
        cv_show('objective_roi', cropped)

    questionCnts, thresh = detect_options(warped, sheet_type=sheet_type, debug=debug)

    all_questions = cluster_into_columns(
        questionCnts, num_options=NUM_OPTIONS,
        warped_img=warped, debug=debug,
    )

    detected_questions = len(all_questions)
    if detected_questions != num_questions:
        print(f"[WARNING] Detected {detected_questions} questions, "
              f"but {num_questions} answers entered")

    student_answers, correct, score = grade_answers(
        all_questions, thresh, answer_key,
        ANSWER_LETTERS, num_questions, debug=debug,
        shape_type=sheet_type,
    )

    return warped, thresh, all_questions, student_answers, correct, score


def run_with_template(image, template, debug=False):
    answer_key = {int(k): v for k, v in template["answer_key"].items()}
    num_questions = template["num_questions"]
    num_subjective = template.get("num_subjective", 0)
    objective_roi = template.get("objective_roi")
    subjective_rois = template.get("subjective_rois", [])
    sheet_type = template.get("sheet_type", SHEET_TYPE_CIRCLE)

    has_ocr = ocr_available()

    subjective_results = []
    if num_subjective > 0 and subjective_rois:
        for i in range(min(num_subjective, len(subjective_rois))):
            rois = subjective_rois[i]
            texts = []
            for j, roi in enumerate(rois):
                cropped = crop_roi(image, roi)
                if cropped is None or cropped.size == 0:
                    texts.append(None)
                    continue
                text = None
                if has_ocr:
                    print(f"[INFO] Recognizing Q{i+1} region #{j+1} ...")
                    text = ocr_recognize(cropped)
                texts.append(text)
            subjective_results.append({
                "index": i + 1,
                "rois": rois,
                "texts": texts,
            })

    if objective_roi:
        result = process_objective_roi(
            image, objective_roi, answer_key, num_questions,
            sheet_type=sheet_type, debug=debug,
        )
    else:
        result = process_objective_full(
            image, answer_key, num_questions,
            sheet_type=sheet_type, debug=debug,
        )

    return result, subjective_results, answer_key, num_questions


def print_results(subjective_results, student_answers, answer_key,
                  num_questions, score, has_subjective=True):
    if has_subjective and subjective_results:
        print("\n" + "=" * 60)
        print("Subjective Results")
        print("=" * 60)
        for res in subjective_results:
            idx = res["index"]
            rois = res.get("rois", [])
            texts = res.get("texts", [])
            if not rois:
                print(f"\n--- Subjective Q{idx} --- (no region)")
                continue
            print(f"\n--- Subjective Q{idx} --- ({len(rois)} region(s))")
            for j, text in enumerate(texts):
                if text:
                    print(f"\n  [Region #{j+1}]")
                    print(f"  {text}")
                else:
                    print(f"\n  [Region #{j+1}] (no text recognized)")

    print("\n" + "=" * 60)
    print("Objective Results")
    print("=" * 60)
    print("[INFO] Student:", " ".join(student_answers))
    print("[INFO] Answer: ",
          " ".join([ANSWER_LETTERS[answer_key[q]] for q in range(num_questions)]))
    print("[INFO] Score: {:.2f}%".format(score))


def show_results(image, warped, all_questions, thresh, answer_key,
                 num_questions, score, sheet_type=SHEET_TYPE_CIRCLE):
    warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    result_img = visualize_results(
        warped_bgr, all_questions, thresh, answer_key,
        ANSWER_LETTERS, num_questions, score,
        shape_type=sheet_type,
    )
    display_image, _ = resize_for_display(image)
    display_result, _ = resize_for_display(result_img)
    cv2.imshow("Original", display_image)
    cv2.imshow("Exam", display_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    run_mode = ask_run_mode()

    if run_mode is None:
        # Create new template
        print("\n" + "=" * 60)
        print("CREATE NEW TEMPLATE")
        print("=" * 60)

        image_path = select_image()
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Cannot read image: {image_path}")
            exit()

        answer_key, num_questions = input_answers()
        sheet_type = ask_sheet_type()
        has_subjective = ask_mode()
        num_subjective = ask_subjective_count() if has_subjective else 0

        root = tk.Tk()
        root.withdraw()
        template_name = simpledialog.askstring(
            "Template Name",
            "Enter a name for this template:",
            parent=root,
        )
        if not template_name:
            template_name = "template"

        template, filename = create_template(
            image, answer_key, num_questions,
            num_subjective=num_subjective,
            name=template_name,
        )
        template["sheet_type"] = sheet_type
        from detector.template import save_template
        save_template(template)

        print(f"\n[INFO] Template '{template_name}' created: {filename}")
        print("[INFO] You can now use this template for batch processing.")

        result, subj_results, answer_key, num_questions = run_with_template(
            image, template, debug=False,
        )
        if result:
            warped, thresh, all_questions, student_answers, correct, score = result
            print_results(subj_results, student_answers, answer_key, num_questions, score)
            show_results(image, warped, all_questions, thresh, answer_key,
                         num_questions, score, sheet_type=sheet_type)

    elif run_mode:
        # Use existing template
        template = select_template()
        if template is None:
            print("[ERROR] No template selected, exiting")
            exit()

        sheet_type = template.get("sheet_type", SHEET_TYPE_CIRCLE)
        image_paths = select_images()
        num_subjective = template.get("num_subjective", 0)
        has_subjective = num_subjective > 0

        for idx, image_path in enumerate(image_paths):
            print(f"\n{'='*60}")
            print(f"Processing [{idx+1}/{len(image_paths)}]: {image_path}")
            print(f"{'='*60}")

            image = cv2.imread(image_path)
            if image is None:
                print(f"[ERROR] Cannot read image: {image_path}")
                continue

            result, subj_results, answer_key, num_questions = run_with_template(
                image, template, debug=False,
            )
            if result is None:
                print(f"[ERROR] Processing failed for: {image_path}")
                continue

            warped, thresh, all_questions, student_answers, correct, score = result
            print_results(subj_results, student_answers, answer_key, num_questions, score, has_subjective)
            show_results(image, warped, all_questions, thresh, answer_key,
                         num_questions, score, sheet_type=sheet_type)

    else:
        # Manual mode
        image_path = select_image()
        answer_key, num_questions = input_answers()
        sheet_type = ask_sheet_type()

        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Cannot read image: {image_path}")
            exit()

        has_subjective = ask_mode()

        if has_subjective:
            print("[INFO] Mode: Objective + Subjective")

            num_subjective = ask_subjective_count()
            print(f"[INFO] Subjective questions: {num_subjective}")

            if num_subjective > 0:
                print("\n" + "=" * 60)
                print("Processing subjective questions")
                print("=" * 60)
                subjective_results = process_subjective_questions(image, num_subjective)
            else:
                subjective_results = []

            print("\n" + "=" * 60)
            print("Select objective region")
            print("=" * 60)
            objective_roi = select_objective_region(image)
            if objective_roi is None:
                print("[ERROR] No objective region selected, exiting")
                exit()

            result = process_objective_roi(
                image, objective_roi, answer_key, num_questions,
                sheet_type=sheet_type, debug=True,
            )
            if result is None:
                exit()

            warped, thresh, all_questions, student_answers, correct, score = result
            print_results(subjective_results, student_answers, answer_key, num_questions, score)
            show_results(image, warped, all_questions, thresh, answer_key,
                         num_questions, score, sheet_type=sheet_type)

        else:
            print("[INFO] Mode: Objective only (full image auto-detect)")
            result = process_objective_full(
                image, answer_key, num_questions,
                sheet_type=sheet_type, debug=True,
            )
            if result is None:
                exit()

            warped, thresh, all_questions, student_answers, correct, score = result
            print("[INFO] Student:", " ".join(student_answers))
            print("[INFO] Answer:",
                  " ".join([ANSWER_LETTERS[answer_key[q]] for q in range(num_questions)]))
            print("[INFO] Score: {:.2f}%".format(score))

            warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            result_img = visualize_results(
                warped_bgr, all_questions, thresh, answer_key,
                ANSWER_LETTERS, num_questions, score,
                shape_type=sheet_type,
            )
            display_image, _ = resize_for_display(image)
            display_result, _ = resize_for_display(result_img)
            cv2.imshow("Original", display_image)
            cv2.imshow("Exam", display_result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
