import numpy as np
import cv2
from config import BUBBLE_RADIUS_RATIO, RECT_PADDING


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    if len(cnts) == 0:
        return [], []
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes


def get_bubble_fill(thresh_img, contour, shape_type="circle"):
    (x, y, w, h) = cv2.boundingRect(contour)
    mask = np.zeros(thresh_img.shape, dtype="uint8")

    if shape_type == "rect":
        pad = RECT_PADDING
        rx1 = max(0, x + pad)
        ry1 = max(0, y + pad)
        rx2 = min(thresh_img.shape[1], x + w - pad)
        ry2 = min(thresh_img.shape[0], y + h - pad)
        cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 255, -1)
    else:
        cx, cy = x + w // 2, y + h // 2
        radius = int(min(w, h) * BUBBLE_RADIUS_RATIO)
        cv2.circle(mask, (cx, cy), radius, 255, -1)

    filled = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)
    total = cv2.countNonZero(filled)
    mask_area = cv2.countNonZero(mask)
    return total, mask_area
