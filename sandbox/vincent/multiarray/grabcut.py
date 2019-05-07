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
image_path = '../../../samples/woman_car.jpg'
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

# Id we are interested in
id = get_main_class(pred)
print labels[id]

id = 12

# plt.figure()
# plt.imshow(img_rgb_small)
# plt.axis('off')
# plt.show()
# exit()


# Mask
mask = compute_mask_from_prediction(pred, id)

# Grabcut mask
output_mask = compute_grabcut_mask(mask, img_rgb_small)
seg = compute_grabcut_segmentation(img_rgb_small, pred)

# Create red illustrations
mask_red = generate_red_illustration(img_rgb_small, mask, apply_rgb2bgr_conversion=False)
gc_red = generate_red_illustration(img_rgb_small, output_mask, apply_rgb2bgr_conversion=False)

# Generate RGB masks
mask = np.tile(mask[:,:,np.newaxis] * 255, (1,1,3))
output_mask = np.tile(output_mask[:,:,np.newaxis] * 255, (1,1,3))

# Stick images together
top = np.concatenate([mask, mask_red], axis=1)
bottom = np.concatenate([output_mask, gc_red], axis=1)
total = np.concatenate([top, bottom], axis=0)

# # Save image
# name = os.path.splitext(os.path.split(image_path)[1])[0]
# cv2.imwrite('/Users/vincent/Desktop/grabcut/%s.png' % name, cv2.cvtColor(total, cv2.COLOR_RGB2BGR))

# Display
plt.figure()
plt.imshow(seg)
plt.axis('off')
plt.show()
exit()
