import cv2
from utils.contour_utils import get_bubble_fill
from config import (
    FILL_RATIO_THRESHOLD,
    FILL_MIN_DIFF,
    DRAW_COLOR_STUDENT,
    DRAW_COLOR_CORRECT,
    DRAW_COLOR_WRONG,
    DRAW_THICKNESS_STUDENT,
    DRAW_THICKNESS_ANSWER,
)


def is_bubbled(totals):
    max_total = max(totals)
    others = sum(totals) - max_total
    avg_others = others / (len(totals) - 1) if len(totals) > 1 else 0
    result = max_total > avg_others * FILL_RATIO_THRESHOLD and max_total > avg_others + FILL_MIN_DIFF
    return result, max_total, avg_others


def grade_answers(all_questions, thresh, answer_key, answer_letters,
                  num_questions, debug=False):
    student_answers = []
    correct = 0
    num_grade = min(len(all_questions), num_questions)

    for q in range(num_grade):
        cnts = all_questions[q]
        totals = []
        for c in cnts:
            total, _ = get_bubble_fill(thresh, c)
            totals.append(total)

        bubbled, max_total, avg_others = is_bubbled(totals)
        bubbled_idx = totals.index(max_total) if bubbled else None

        if debug:
            print(f"[DEBUG] Q{q + 1}: totals={totals}, max={max_total}, "
                  f"avg_others={avg_others:.0f}, filled={bubbled}")

        if bubbled:
            student_answers.append(answer_letters[bubbled_idx])
            if q in answer_key and answer_key[q] == bubbled_idx:
                correct += 1
        else:
            student_answers.append("-")

    score = (correct / num_questions) * 100 if num_questions > 0 else 0
    return student_answers, correct, score


def visualize_results(warped_bgr, all_questions, thresh, answer_key,
                      answer_letters, num_questions, score):
    num_grade = min(len(all_questions), num_questions)

    for q in range(num_grade):
        if q not in answer_key:
            continue
        cnts = all_questions[q]
        k = answer_key[q]

        totals = []
        for c in cnts:
            total, _ = get_bubble_fill(thresh, c)
            totals.append(total)

        bubbled, _, _ = is_bubbled(totals)
        bubbled_idx = totals.index(max(totals)) if bubbled else None

        if bubbled:
            cv2.drawContours(warped_bgr, [cnts[bubbled_idx]], -1,
                             DRAW_COLOR_STUDENT, DRAW_THICKNESS_STUDENT)

        is_correct = bubbled and k == bubbled_idx
        color = DRAW_COLOR_CORRECT if is_correct else DRAW_COLOR_WRONG
        cv2.drawContours(warped_bgr, [cnts[k]], -1, color, DRAW_THICKNESS_ANSWER)

    cv2.putText(warped_bgr, "{:.2f}%".format(score), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, DRAW_COLOR_WRONG, 2)

    return warped_bgr
