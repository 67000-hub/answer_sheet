class OCRRecognizer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ocr = None
            cls._instance._available = None
            cls._instance._backend = None
        return cls._instance

    def _init_ocr(self):
        if self._ocr is not None or self._available is False:
            return self._available

        try:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
            self._backend = "rapidocr"
            self._available = True
            print("[INFO] RapidOCR 加载成功")
            return self._available
        except ImportError:
            pass
        except Exception as e:
            print(f"[WARNING] RapidOCR 加载失败: {e}")

        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            self._backend = "paddleocr"
            self._available = True
            print("[INFO] PaddleOCR 加载成功")
            return self._available
        except ImportError:
            pass
        except Exception as e:
            print(f"[WARNING] PaddleOCR 加载失败: {e}")

        self._available = False
        print("[WARNING] 未安装 OCR 引擎，OCR 功能不可用")
        print("         安装命令（推荐）: pip install rapidocr-onnxruntime")
        print("         或: pip install paddleocr paddlepaddle")

        return self._available

    @property
    def is_available(self):
        return self._init_ocr()

    def recognize(self, image):
        if not self._init_ocr():
            return None

        try:
            if self._backend == "rapidocr":
                result, _ = self._ocr(image)
                if not result:
                    return ""
                lines = []
                for line in result:
                    text = line[1]
                    lines.append(text)
                return "\n".join(lines)

            elif self._backend == "paddleocr":
                result = self._ocr.ocr(image, cls=True)
                if not result or not result[0]:
                    return ""
                lines = []
                for line in result[0]:
                    text = line[1][0]
                    lines.append(text)
                return "\n".join(lines)

        except Exception as e:
            print(f"[ERROR] OCR 识别失败: {e}")
            return None


def ocr_recognize(image):
    recognizer = OCRRecognizer()
    return recognizer.recognize(image)


def ocr_available():
    recognizer = OCRRecognizer()
    return recognizer.is_available
