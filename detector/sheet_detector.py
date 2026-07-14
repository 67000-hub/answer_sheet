import cv2
import imutils
import numpy as np
from utils.viz import cv_show
from utils.geometry import four_point_transform
from config import CANNY_LOW, CANNY_HIGH, GAUSSIAN_KERNEL


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    edged = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    return gray, blurred, edged


def find_sheet(image, debug=False):
    contours_img = image.copy()
    gray, blurred, edged = preprocess_image(image)

    if debug:
        cv_show('blurred', blurred)
        cv_show('edged', edged)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if debug:
        cv2.drawContours(contours_img, cnts, -1, (0, 0, 255), 3)
        cv_show('contours_img', contours_img)

    docCnt = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    return docCnt, gray


def warp_sheet(gray, docCnt, debug=False):
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    if debug:
        cv_show('warped', warped)
    return warped
