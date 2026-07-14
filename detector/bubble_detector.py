import cv2
import numpy as np
from utils.viz import cv_show
from config import (
    MORPH_KERNEL_SIZE,
    BUBBLE_MIN_SIZE,
    BUBBLE_AR_MIN,
    BUBBLE_AR_MAX,
    BUBBLE_SIZE_MULTIPLIER_LOW,
    BUBBLE_SIZE_MULTIPLIER_HIGH,
)


def threshold_image(warped, debug=False):
    thresh = cv2.threshold(warped, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if debug:
        cv_show('thresh', thresh)
    return thresh


def morphological_close(thresh, kernel_size=MORPH_KERNEL_SIZE, debug=False):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    if debug:
        cv_show('thresh_closed', thresh_closed)
    return thresh_closed


def find_all_contours(thresh_closed, debug=False):
    cnts, hierarchy = cv2.findContours(thresh_closed.copy(), cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        all_contours_img = thresh_closed.copy()
        all_contours_img = cv2.cvtColor(all_contours_img, cv2.COLOR_GRAY2BGR)
        for i, c in enumerate(cnts):
            color = (0, 255, 0) if hierarchy[0][i][3] == -1 else (255, 0, 0)
            cv2.drawContours(all_contours_img, [c], -1, color, 1)
        cv_show('all_contours', all_contours_img)

    return cnts, hierarchy


def filter_candidates(cnts, min_size=BUBBLE_MIN_SIZE,
                      ar_min=BUBBLE_AR_MIN, ar_max=BUBBLE_AR_MAX):
    candidates = []
    for i, c in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if w < min_size or h < min_size:
            continue
        ar = w / float(h)
        if ar < ar_min or ar > ar_max:
            continue
        area = cv2.contourArea(c)
        candidates.append((x, y, w, h, area, c, i))
    return candidates


def filter_by_size(candidates, low_mult=BUBBLE_SIZE_MULTIPLIER_LOW,
                   high_mult=BUBBLE_SIZE_MULTIPLIER_HIGH):
    if len(candidates) == 0:
        return []

    ws = [w for x, y, w, h, area, c, idx in candidates]
    median_w = np.median(ws)
    min_w = median_w * low_mult
    max_w = median_w * high_mult

    filtered = [cand for cand in candidates
                if min_w <= cand[2] <= max_w and min_w <= cand[3] <= max_w]

    print(f"[DEBUG] 尺寸筛选后候选数: {len(filtered)}, "
          f"median_w={median_w:.0f}, range=[{min_w:.0f}, {max_w:.0f}]")
    return filtered


def deduplicate_bubbles(candidates):
    candidates.sort(key=lambda c: c[4], reverse=True)
    questionCnts = []
    used_centers = []
    for x, y, w, h, area, c, idx in candidates:
        cx, cy = x + w // 2, y + h // 2
        too_close = False
        for ux, uy in used_centers:
            dist = np.sqrt((cx - ux) ** 2 + (cy - uy) ** 2)
            if dist < min(w, h) * 0.4:
                too_close = True
                break
        if not too_close:
            questionCnts.append(c)
            used_centers.append((cx, cy))
    return questionCnts


def detect_bubbles(warped, debug=False):
    thresh = threshold_image(warped, debug=debug)
    thresh_closed = morphological_close(thresh, debug=debug)
    cnts, hierarchy = find_all_contours(thresh_closed, debug=debug)

    candidates = filter_candidates(cnts)
    print(f"[DEBUG] 候选轮廓数: {len(candidates)}")
    for i, (x, y, w, h, area, c, idx) in enumerate(candidates[:30]):
        print(f"  候选{i}: x={x}, y={y}, w={w}, h={h}, area={area:.0f}, ar={w/h:.2f}")

    candidates = filter_by_size(candidates)
    questionCnts = deduplicate_bubbles(candidates)

    print(f"[DEBUG] 去重后气泡数: {len(questionCnts)}")

    if debug and len(questionCnts) > 0:
        bubble_debug = warped.copy()
        if len(bubble_debug.shape) == 2:
            bubble_debug = cv2.cvtColor(bubble_debug, cv2.COLOR_GRAY2BGR)
        for c in questionCnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(bubble_debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv_show('detected_bubbles', bubble_debug)

    return questionCnts, thresh
