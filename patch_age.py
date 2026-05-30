from pathlib import Path
path = Path('age_estimation_pro.py')
text = path.read_text(encoding='utf-8')
needle = "# 加载人脸检测器\nface_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')\n"
replacement = needle + "\n" + "def imread_unicode(path: str):\n    try:\n        data = np.fromfile(path, dtype=np.uint8)\n        if data.size == 0:\n            return None\n        img = cv2.imdecode(data, cv2.IMREAD_COLOR)\n        return img\n    except Exception:\n        return None\n\n"
if needle not in text:
    raise SystemExit('needle not found')
text = text.replace(needle, replacement)
text = text.replace('cv2.imread(img_path)', 'imread_unicode(img_path)')
text = text.replace('cv2.imread(self.image_paths[idx])', 'imread_unicode(self.image_paths[idx])')
text = text.replace('cv2.imread(img_path)', 'imread_unicode(img_path)')
text = text.replace('cv2.imread(img_path)', 'imread_unicode(img_path)')
path.write_text(text, encoding='utf-8')
print('patched')
