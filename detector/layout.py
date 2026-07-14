import cv2
from utils.contour_utils import sort_contours
from utils.viz import cv_show
from config import COL_GAP_MULTIPLIER


def cluster_into_columns(bubbles, num_options=4, warped_img=None, debug=False):
    if len(bubbles) == 0:
        return []

    sorted_by_x = sorted(bubbles, key=lambda c: cv2.boundingRect(c)[0])
    xs = [cv2.boundingRect(c)[0] for c in sorted_by_x]

    widths = [cv2.boundingRect(c)[2] for c in bubbles]
    avg_w = sum(widths) / len(widths) if widths else 20

    col_gap_threshold = avg_w * COL_GAP_MULTIPLIER

    print(f"[DEBUG] 平均气泡宽度: {avg_w:.0f}, 列间距阈值: {col_gap_threshold:.0f}")

    columns = [[sorted_by_x[0]]]
    for i in range(1, len(sorted_by_x)):
        gap = xs[i] - xs[i - 1]
        if gap > col_gap_threshold:
            columns.append([])
        columns[-1].append(sorted_by_x[i])

    num_cols = len(columns)
    print(f"[DEBUG] 检测到 {num_cols} 列，每列气泡数: {[len(col) for col in columns]}")

    all_questions = []
    for col_idx, col in enumerate(columns):
        col_sorted = sort_contours(col, method="top-to-bottom")[0]
        for j in range(0, len(col_sorted), num_options):
            group = col_sorted[j:j + num_options]
            if len(group) == num_options:
                group_sorted = sort_contours(group, method="left-to-right")[0]
                all_questions.append(group_sorted)

    print(f"[DEBUG] 组装完成，共 {len(all_questions)} 题")

    if debug and warped_img is not None and len(columns) > 0:
        col_debug = warped_img.copy()
        if len(col_debug.shape) == 2:
            col_debug = cv2.cvtColor(col_debug, cv2.COLOR_GRAY2BGR)
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255),
                  (255, 255, 0), (255, 0, 255)]
        for col_idx, col in enumerate(columns):
            color = colors[col_idx % len(colors)]
            for c in col:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(col_debug, (x, y), (x + w, y + h), color, 2)
            cv2.putText(col_debug, f"Col{col_idx + 1}",
                        (cv2.boundingRect(col[0])[0], 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv_show('columns_debug', col_debug)

    return all_questions
