import coremltools
import os
import matplotlib.pyplot as plt
import cv2

from tools import *

# Paths
model_path = '../../../models/MultiArrayDeepLab.mlmodel'

# Read the model
model = coremltools.models.MLModel(model_path)

# Image path
# image_path = '../../../samples/person_matthieu.jpg'
# image_path = '../../../samples/moto_man.jpg'
# image_path = '../../../samples/car_nico.jpg'
image_path = '../../../samples/woman_car.jpg'

# Read the input image
img_bgr_full = cv2.imread(image_path)
img_rgb_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB)

# Scale the color image
max_size = 1024
img_rgb_small = scale_to_fit(img_rgb_full, max_size, cv2.INTER_LINEAR)
img_y_small = cv2.cvtColor(img_rgb_small, cv2.COLOR_RGB2GRAY)
height, width, _ = img_rgb_small.shape

# Prediction
pred = predict(model, img_rgb_small)

# Id we are interested in
id = get_main_class(pred)
print id, labels[id]

# Argmax
mask0 = compute_mask_from_prediction(pred, id)

# Threshold
threshold = 0.5
mask1 = np.zeros(mask0.shape, dtype=np.uint8)
tmp = pred[id,:,:]
mask1[tmp >= threshold] = 1

# Filter the prediction
radius = 10
eps = 0.001
img_y2 = img_y_small.astype(np.float32) / 255.
pred2 = pred.copy()
for c in range(pred.shape[0]):
    channel = pred[c,:,:]
    pred2[c,:,:] = np.clip(cv2.ximgproc.guidedFilter(img_y2, channel, radius, eps), 0., 1.)
pred2 = pred2 / np.sum(pred2, axis=0)

# Filter + argmax
mask2 = compute_mask_from_prediction(pred2, id)

# Filter + threshold
mask3 = np.zeros(mask0.shape, dtype=np.uint8)
tmp3 = pred2[id,:,:]
mask3[tmp3 >= threshold] = 1


red0 = generate_red_illustration(img_rgb_small, mask0)
red1 = generate_red_illustration(img_rgb_small, mask1)
red2 = generate_red_illustration(img_rgb_small, mask2)
red3 = generate_red_illustration(img_rgb_small, mask3)

# f, ax = plt.subplots(2,2)
# ax[0,0].imshow(red0)
# ax[0,1].imshow(red1)
# ax[1,0].imshow(red2)
# ax[1,1].imshow(red3)
# plt.show()

top = np.concatenate([red0, red1], axis=1)
bottom = np.concatenate([red2, red3], axis=1)
out = np.concatenate([top, bottom], axis=0)

plt.figure()
plt.imshow(out)
plt.axis('off')
plt.show()

