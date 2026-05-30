import os
import glob
import sys
import cv2
from PIL import Image

print('python version:', sys.version)
base_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, 'UTKFace')
print('base_dir:', base_dir)
print('data_dir:', data_dir)
print('exists:', os.path.isdir(data_dir))
paths = sorted(glob.glob(os.path.join(data_dir, '*.jpg')))
paths_chip = sorted(glob.glob(os.path.join(data_dir, '*.jpg.chip.jpg')))
print('jpg count:', len(paths))
print('jpg.chip.jpg count:', len(paths_chip))

sample = paths_chip[:8] if paths_chip else paths[:8]
print('sample paths:')
for p in sample:
    print(' -', p)
    print('   size:', os.path.getsize(p))
    try:
        img = cv2.imread(p)
        print('   cv2:', None if img is None else img.shape)
    except Exception as e:
        print('   cv2 error:', e)
    try:
        with Image.open(p) as im:
            im.verify()
        print('   PIL: ok')
    except Exception as e:
        print('   PIL error:', e)
