import cv2
import numpy as np


class ROISelector:
    def __init__(self, window_name="选择区域", max_width=1000, max_height=700):
        self.window_name = window_name
        self.max_width = max_width
        self.max_height = max_height
        self.ref_point = []
        self.cropping = False
        self.original_image = None
        self.display_image = None
        self.clone = None
        self.scale = 1.0
        self.prompt_text = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_point = [(x, y)]
            self.cropping = True
        elif event == cv2.EVENT_MOUSEMOVE and self.cropping:
            self.clone = self.display_image.copy()
            cv2.rectangle(self.clone, self.ref_point[0], (x, y), (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.clone)
        elif event == cv2.EVENT_LBUTTONUP and self.cropping:
            self.cropping = False
            self.ref_point.append((x, y))
            self.clone = self.display_image.copy()
            cv2.rectangle(self.clone, self.ref_point[0], self.ref_point[1], (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.clone)

    def _scale_image(self, image):
        h, w = image.shape[:2]
        scale = 1.0

        if w > self.max_width:
            scale = min(scale, self.max_width / w)
        if h > self.max_height:
            scale = min(scale, self.max_height / h)

        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return scaled, scale
        return image, 1.0

    def _redraw_with_prompt(self):
        self.clone = self.display_image.copy()
        if self.prompt_text:
            lines = self.prompt_text.split("\n")
            for i, line in enumerate(lines):
                cv2.putText(self.clone, line, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        help_text = "回车=确认  R=重选  C=清除  ESC=取消"
        cv2.putText(self.clone, help_text, (10, self.clone.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def select_roi(self, image, prompt_text=None):
        self.original_image = image.copy()
        self.prompt_text = prompt_text
        self.ref_point = []
        self.cropping = False

        if len(self.original_image.shape) == 2:
            img_bgr = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = self.original_image.copy()

        self.display_image, self.scale = self._scale_image(img_bgr)
        self._redraw_with_prompt()

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        cv2.imshow(self.window_name, self.clone)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 13 and len(self.ref_point) == 2:
                break
            elif key == ord("c") or key == ord("C"):
                self.ref_point = []
                self._redraw_with_prompt()
                cv2.imshow(self.window_name, self.clone)
            elif key == ord("r") or key == ord("R"):
                self.ref_point = []
                self._redraw_with_prompt()
                cv2.imshow(self.window_name, self.clone)
            elif key == 27:
                self.ref_point = []
                break

        cv2.destroyWindow(self.window_name)

        if len(self.ref_point) == 2:
            x1, y1 = self.ref_point[0]
            x2, y2 = self.ref_point[1]
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            orig_x1 = int(x1 / self.scale)
            orig_y1 = int(y1 / self.scale)
            orig_x2 = int(x2 / self.scale)
            orig_y2 = int(y2 / self.scale)

            h, w = self.original_image.shape[:2]
            orig_x1 = max(0, min(orig_x1, w - 1))
            orig_y1 = max(0, min(orig_y1, h - 1))
            orig_x2 = max(0, min(orig_x2, w - 1))
            orig_y2 = max(0, min(orig_y2, h - 1))

            return (orig_x1, orig_y1, orig_x2, orig_y2)
        else:
            return None


def select_roi_interactive(image, title="选择区域", prompt=None, max_w=1000, max_h=700):
    selector = ROISelector(title, max_width=max_w, max_height=max_h)
    return selector.select_roi(image, prompt)


def crop_roi(image, roi):
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    return image[y1:y2, x1:x2]
