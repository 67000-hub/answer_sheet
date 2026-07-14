import numpy as np
import cv2


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


def get_bubble_fill(thresh_img, contour):
    (x, y, w, h) = cv2.boundingRect(contour)
    cx, cy = x + w // 2, y + h // 2
    radius = int(min(w, h) * 0.45)
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    filled = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)
    total = cv2.countNonZero(filled)
    circle_area = cv2.countNonZero(mask)
    return total, circle_area
