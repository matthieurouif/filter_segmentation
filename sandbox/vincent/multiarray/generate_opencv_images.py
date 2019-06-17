import coremltools
import os
import matplotlib.pyplot as plt
import cv2
from glob import glob
import sys
import time

from tools import *


# Paths
model_path = '../../../models/MultiArrayDeepLab.mlmodel'
output_folder = '/Users/vincent/Desktop'

# Read the model
model = coremltools.models.MLModel(model_path)

# Image path
image_path = '../../../samples/dog_guinness.jpg'
# image_path = '../../../samples/moto_man.jpg'
# input_path = '../../../samples/flowers.jpg'
# image_path = '/Users/vincent/Downloads/VOCdevkit/VOC2012/JPEGImages/2007_000452.jpg'

# Read the input image
img_bgr_full = cv2.imread(image_path)
img_rgb_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB)

# Scale the color image
max_size = 1024
img_rgb_small = scale_to_fit(img_rgb_full, max_size, cv2.INTER_LINEAR)
height, width, _ = img_rgb_small.shape

# Prediction
pred = predict(model, img_rgb_small)

id = get_main_class(pred)

tmp = pred[id,:,:]
tmp = (tmp*255).astype(np.uint8)

plt.figure()
plt.imshow(tmp)
plt.show()


cv2.imwrite(os.path.join(output_folder, 'ref_image.png'), cv2.cvtColor(img_rgb_small, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_folder, 'ref_pred.png'), tmp)