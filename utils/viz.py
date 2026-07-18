import cv2
from config import DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT


def resize_for_display(img, max_width=DISPLAY_MAX_WIDTH, max_height=DISPLAY_MAX_HEIGHT):
    h, w = img.shape[:2]
    if w <= max_width and h <= max_height:
        return img, 1.0
    scale_w = max_width / float(w)
    scale_h = max_height / float(h)
    scale = min(scale_w, scale_h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def cv_show(name, img, auto_resize=True):
    if auto_resize:
        display_img, _ = resize_for_display(img)
    else:
        display_img = img
    cv2.imshow(name, display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
