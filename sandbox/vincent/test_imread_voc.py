from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

path = '/Users/vincent/Downloads/VOCdevkit/VOC2012/SegmentationClass/2007_001311.png'

img0 = Image.open(path)

img1 = img0.convert('RGB')

img2 = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


f,ax = plt.subplots(1,3)
ax[0].imshow(img0, cmap='gray')
ax[1].imshow(img1)
ax[2].imshow(img2)
plt.show()