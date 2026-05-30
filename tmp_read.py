import os
import numpy as np
import cv2

path = os.path.join(os.getcwd(), 'UTKFace', '100_0_0_20170112213500903.jpg.chip.jpg')
print('exists', os.path.exists(path), os.path.getsize(path))
data = np.fromfile(path, dtype=np.uint8)
print('data size', data.size)
img = cv2.imdecode(data, cv2.IMREAD_COLOR)
print('img', None if img is None else img.shape)
