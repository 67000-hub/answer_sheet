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
    NUM_OPTIONS,
    RECT_MIN_SIZE,
    RECT_AR_MIN,
    RECT_AR_MAX,
    RECT_SIZE_MULTIPLIER_LOW,
    RECT_SIZE_MULTIPLIER_HIGH,
    RECT_BRACKET_MIN_W,
    RECT_BRACKET_MIN_H,
    RECT_BRACKET_AR_MAX,
    RECT_BRACKET_FILL_MIN,
    RECT_BRACKET_FILL_MAX,
    RECT_BRACKET_Y_TOL,
    RECT_BRACKET_GAP_MIN,
    RECT_BRACKET_GAP_MAX,
    RECT_SOLID_FILL_MIN,
    RECT_RECT_PAD_X,
    RECT_RECT_PAD_Y,
    RECT_INNER_FILL_MIN,
    RECT_INNER_PAD_RATIO_X,
    RECT_INNER_PAD_RATIO_Y,
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
                   high_mult=BUBBLE_SIZE_MULTIPLIER_HIGH, debug_label="size"):
    if len(candidates) == 0:
        return []

    ws = [w for x, y, w, h, area, c, idx in candidates]
    median_w = np.median(ws)
    min_w = median_w * low_mult
    max_w = median_w * high_mult

    filtered = [cand for cand in candidates
                if min_w <= cand[2] <= max_w and min_w <= cand[3] <= max_w]

    print(f"[DEBUG] {debug_label} filter: {len(filtered)} candidates, "
          f"median_w={median_w:.0f}, range=[{min_w:.0f}, {max_w:.0f}]")
    return filtered


def deduplicate_by_center(candidates, dist_ratio=0.4):
    candidates.sort(key=lambda c: c[4], reverse=True)
    result = []
    used_centers = []
    for x, y, w, h, area, c, idx in candidates:
        cx, cy = x + w // 2, y + h // 2
        too_close = False
        for ux, uy in used_centers:
            dist = np.sqrt((cx - ux) ** 2 + (cy - uy) ** 2)
            if dist < min(w, h) * dist_ratio:
                too_close = True
                break
        if not too_close:
            result.append(c)
            used_centers.append((cx, cy))
    return result


def detect_bubbles(warped, debug=False):
    thresh = threshold_image(warped, debug=debug)
    thresh_closed = morphological_close(thresh, debug=debug)
    cnts, hierarchy = find_all_contours(thresh_closed, debug=debug)

    candidates = filter_candidates(cnts)
    print(f"[DEBUG] circle candidates: {len(candidates)}")
    for i, (x, y, w, h, area, c, idx) in enumerate(candidates[:30]):
        print(f"  #{i}: x={x}, y={y}, w={w}, h={h}, area={area:.0f}, ar={w/h:.2f}")

    candidates = filter_by_size(candidates, debug_label="circle")
    questionCnts = deduplicate_by_center(candidates)

    print(f"[DEBUG] circle bubbles after dedup: {len(questionCnts)}")

    if debug and len(questionCnts) > 0:
        bubble_debug = warped.copy()
        if len(bubble_debug.shape) == 2:
            bubble_debug = cv2.cvtColor(bubble_debug, cv2.COLOR_GRAY2BGR)
        for c in questionCnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(bubble_debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv_show('detected_bubbles', bubble_debug)

    return questionCnts, thresh


def has_inner_content(thresh, rx, ry, rw, rh,
                       min_fill=RECT_INNER_FILL_MIN,
                       pad_ratio_x=RECT_INNER_PAD_RATIO_X,
                       pad_ratio_y=RECT_INNER_PAD_RATIO_Y,
                       debug=False):
    """检查矩形内部是否有内容（字母 A/B/C/D）。

    选项框 [A] 内部有字母，有黑色像素；
    题号 "11" 配对后内部是空白，几乎没有黑色像素。
    """
    pad_x = max(2, int(rw * pad_ratio_x))
    pad_y = max(2, int(rh * pad_ratio_y))
    ix = rx + pad_x
    iy = ry + pad_y
    iw = rw - 2 * pad_x
    ih = rh - 2 * pad_y
    if iw <= 0 or ih <= 0:
        return True
    ix2 = min(ix + iw, thresh.shape[1])
    iy2 = min(iy + ih, thresh.shape[0])
    ix = max(0, ix)
    iy = max(0, iy)
    inner = thresh[iy:iy2, ix:ix2]
    if inner.size == 0:
        return True
    fill = np.count_nonzero(inner) / float(inner.size)
    if debug:
        print(f"  inner fill={fill:.4f} at ({ix},{iy},{ix2-ix},{iy2-iy})")
    return fill >= min_fill


def _cluster_rows(rects, tol_ratio=0.5):
    rects_sorted = sorted(rects, key=lambda r: r[1])
    rows = []
    current_row = [rects_sorted[0]]
    for rect in rects_sorted[1:]:
        avg_y = sum(r[1] for r in current_row) / len(current_row)
        avg_h = sum(r[3] for r in current_row) / len(current_row)
        if abs(rect[1] - avg_y) <= avg_h * tol_ratio:
            current_row.append(rect)
        else:
            rows.append(current_row)
            current_row = [rect]
    rows.append(current_row)
    return rows


def _row_pattern_score(row_rects):
    if len(row_rects) < 2:
        return 0.0
    sorted_x = sorted(row_rects, key=lambda r: r[0])
    widths = [r[2] for r in sorted_x]
    heights = [r[3] for r in sorted_x]
    xs = [r[0] for r in sorted_x]
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]

    def cv(arr):
        if not arr:
            return 0.0
        mean = sum(arr) / len(arr)
        if mean == 0:
            return 0.0
        variance = sum((x - mean) ** 2 for x in arr) / len(arr)
        std = variance ** 0.5
        return std / mean

    w_cv = cv(widths)
    h_cv = cv(heights)
    gap_cv = cv(gaps) if gaps else 0.0

    score = 1.0 - (w_cv * 0.3 + h_cv * 0.2 + gap_cv * 0.5)
    return max(0.0, score)


def filter_by_row_pattern(all_rects, num_options=4, min_score=0.6, debug=False):
    if len(all_rects) <= num_options:
        return all_rects

    rows = _cluster_rows(all_rects)
    if debug:
        print(f"[DEBUG] pattern filter: {len(rows)} rows, "
              f"sizes: {[len(r) for r in rows]}")

    filtered = []
    for row_idx, row in enumerate(rows):
        if len(row) <= num_options:
            filtered.extend(row)
            continue

        remaining = sorted(row, key=lambda r: r[0])
        row_result = []

        while len(remaining) >= num_options:
            best_score = -1.0
            best_start = -1

            for start in range(len(remaining) - num_options + 1):
                group = remaining[start:start + num_options]
                score = _row_pattern_score(group)
                if score > best_score:
                    best_score = score
                    best_start = start

            if best_score < min_score or best_start < 0:
                break

            best_group = remaining[best_start:best_start + num_options]
            row_result.extend(best_group)

            if debug:
                print(f"[DEBUG] row {row_idx}: group start_x={best_group[0][0]}, "
                      f"score={best_score:.3f}")

            remaining = remaining[:best_start] + remaining[best_start + num_options:]

        if row_result:
            filtered.extend(row_result)
        else:
            filtered.extend(row[:num_options])

    filtered.sort(key=lambda r: (r[1], r[0]))
    print(f"[DEBUG] pattern filter: {len(all_rects)} -> {len(filtered)}")
    return filtered


def detect_rect_options(warped, debug=False):
    thresh = threshold_image(warped, debug=debug)
    h, w = thresh.shape

    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    brackets = []
    solid_rects = []
    for c in cnts:
        (bx, by, bw, bh) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if bw < RECT_BRACKET_MIN_W or bh < RECT_BRACKET_MIN_H:
            continue
        if bw > w // 5 or bh > h // 5:
            continue
        ar = bw / float(bh)
        rect_area = bw * bh
        if rect_area == 0:
            continue
        fill_ratio = area / rect_area

        if fill_ratio > RECT_SOLID_FILL_MIN and RECT_AR_MIN <= ar <= RECT_AR_MAX:
            solid_rects.append((bx, by, bw, bh, area, c))
        elif ar <= RECT_BRACKET_AR_MAX and RECT_BRACKET_FILL_MIN <= fill_ratio <= RECT_BRACKET_FILL_MAX:
            brackets.append((bx, by, bw, bh, area, c))

    print(f"[DEBUG] bracket candidates: {len(brackets)}, solid rects: {len(solid_rects)}")
    if debug:
        dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR) if len(warped.shape) == 2 else warped.copy()
        for idx, (bx, by, bw, bh, area, c) in enumerate(brackets):
            cv2.rectangle(dbg, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
            cv2.putText(dbg, str(idx), (bx, by - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for idx, (bx, by, bw, bh, area, c) in enumerate(solid_rects):
            cv2.rectangle(dbg, (bx, by), (bx + bw, by + bh), (255, 0, 255), 2)
            cv2.putText(dbg, f"S{idx}", (bx, by - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        cv_show('rect_brackets_and_solid', dbg)

    all_rects = []

    brackets.sort(key=lambda b: (b[1], b[0]))
    used = [False] * len(brackets)

    for i in range(len(brackets)):
        if used[i]:
            continue
        bx1, by1, bw1, bh1, area1, c1 = brackets[i]

        best_j = -1
        best_score = float('inf')

        for j in range(i + 1, len(brackets)):
            if used[j]:
                continue
            bx2, by2, bw2, bh2, area2, c2 = brackets[j]

            dy = abs(by1 - by2)
            if dy > RECT_BRACKET_Y_TOL:
                continue

            dh = abs(bh1 - bh2)
            if dh > max(bh1, bh2) * 0.3:
                continue

            dw = abs(bw1 - bw2)
            if dw > max(bw1, bw2) * 0.5:
                continue

            gap = bx2 - (bx1 + bw1)
            if gap < RECT_BRACKET_GAP_MIN or gap > RECT_BRACKET_GAP_MAX:
                continue

            score = dy * 2 + gap * 0.5 + dh
            if score < best_score:
                best_score = score
                best_j = j

        if best_j >= 0:
            bx2, by2, bw2, bh2, area2, c2 = brackets[best_j]

            rx = bx1 - RECT_RECT_PAD_X
            ry = min(by1, by2) - RECT_RECT_PAD_Y
            rw = (bx2 + bw2) - rx + RECT_RECT_PAD_X
            rh = max(by1 + bh1, by2 + bh2) - ry + RECT_RECT_PAD_Y

            rx = max(0, rx)
            ry = max(0, ry)
            rw = min(rw, w - rx)
            rh = min(rh, h - ry)

            if (rw >= RECT_MIN_SIZE and rh >= RECT_MIN_SIZE and
                RECT_AR_MIN <= rw / float(rh) <= RECT_AR_MAX):
                if has_inner_content(thresh, rx, ry, rw, rh, debug=debug):
                    all_rects.append((rx, ry, rw, rh))
                elif debug:
                    print(f"[DEBUG] 跳过空内容矩形: ({rx},{ry},{rw},{rh})")
                used[i] = True
                used[best_j] = True

    for (bx, by, bw, bh, area, c) in solid_rects:
        rx = bx - RECT_RECT_PAD_X
        ry = by - RECT_RECT_PAD_Y
        rw = bw + 2 * RECT_RECT_PAD_X
        rh = bh + 2 * RECT_RECT_PAD_Y
        rx = max(0, rx)
        ry = max(0, ry)
        rw = min(rw, w - rx)
        rh = min(rh, h - ry)
        if (rw >= RECT_MIN_SIZE and rh >= RECT_MIN_SIZE and
            RECT_AR_MIN <= rw / float(rh) <= RECT_AR_MAX):
            all_rects.append((rx, ry, rw, rh))

    all_rects.sort(key=lambda r: (r[1], r[0]))

    print(f"[DEBUG] rect from bracket pairs + solid: {len(all_rects)}")

    if len(all_rects) == 0:
        return [], thresh

    rect_contours = []
    for (rx, ry, rw, rh) in all_rects:
        rect = np.array([
            [rx, ry],
            [rx + rw, ry],
            [rx + rw, ry + rh],
            [rx, ry + rh],
        ], dtype=np.int32)
        rect_contours.append(rect)

    candidates = []
    for i, c in enumerate(rect_contours):
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        candidates.append((x, y, cw, ch, area, c, i))

    candidates = filter_by_size(
        candidates,
        low_mult=RECT_SIZE_MULTIPLIER_LOW,
        high_mult=RECT_SIZE_MULTIPLIER_HIGH,
        debug_label="rect",
    )

    rect_list = [(x, y, cw, ch) for x, y, cw, ch, area, c, idx in candidates]
    rect_list = filter_by_row_pattern(
        rect_list, num_options=NUM_OPTIONS, debug=debug
    )

    questionCnts = []
    for (rx, ry, rw, rh) in rect_list:
        rect = np.array([
            [rx, ry],
            [rx + rw, ry],
            [rx + rw, ry + rh],
            [rx, ry + rh],
        ], dtype=np.int32)
        questionCnts.append(rect)
    print(f"[DEBUG] rect options final: {len(questionCnts)}")

    if debug and len(questionCnts) > 0:
        bubble_debug = warped.copy()
        if len(bubble_debug.shape) == 2:
            bubble_debug = cv2.cvtColor(bubble_debug, cv2.COLOR_GRAY2BGR)
        for c in questionCnts:
            (x, y, cw, ch) = cv2.boundingRect(c)
            cv2.rectangle(bubble_debug, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
        cv_show('detected_rect_options', bubble_debug)

    return questionCnts, thresh


def detect_options(warped, sheet_type="circle", debug=False):
    if sheet_type == "rect":
        return detect_rect_options(warped, debug=debug)
    else:
        return detect_bubbles(warped, debug=debug)
