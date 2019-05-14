import coremltools
import os
import matplotlib.pyplot as plt
import cv2

from tools import *


def grapbcut_quadmap(mask):
    # Element
    radius = int(max(mask.shape) / 50)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)

    # Sure foreground, background and unknown areas
    mask_ero = cv2.erode(mask, kernel)
    mask_dil = cv2.dilate(mask, kernel)
    sure_fg = mask_ero.astype(np.bool_)
    sure_bg = np.logical_not(mask_dil)
    pr_fg = np.logical_and(mask, np.logical_not(sure_fg))
    pr_bg = np.logical_and(mask_dil, np.logical_not(mask))

    # Sanity check
    if np.count_nonzero(sure_fg) == 0 or np.count_nonzero(sure_bg) == 0:
        return mask

    # Build input GrabCut mask
    grabcut_mask = np.zeros(mask.shape, dtype=np.uint8)
    grabcut_mask[sure_bg] = cv2.GC_BGD
    grabcut_mask[sure_fg] = cv2.GC_FGD
    grabcut_mask[pr_bg] = cv2.GC_PR_BGD
    grabcut_mask[pr_fg] = cv2.GC_PR_FGD

    return grabcut_mask


def foo(pred, id):
    layer = pred[id,:,:]
    return layer


def bar(pred, id):

    layer = pred[id,:,:]

    t0 = 0.05
    t1 = 0.95

    sure_fg = layer >= t1
    sure_bg = layer <= t0
    pr_fg = np.logical_and(layer >= 0.5, layer < t1)
    pr_bg = np.logical_and(layer < 0.5, layer >= t0)

    grabcut_mask = np.zeros(layer.shape, dtype=np.uint8)
    grabcut_mask[sure_bg] = cv2.GC_BGD
    grabcut_mask[sure_fg] = cv2.GC_FGD
    grabcut_mask[pr_bg] = cv2.GC_PR_BGD
    grabcut_mask[pr_fg] = cv2.GC_PR_FGD

    return grabcut_mask


# Paths
model_path = '../../../models/MultiArrayDeepLab.mlmodel'

# Read the model
model = coremltools.models.MLModel(model_path)

# Image path
# image_path = '../../../samples/person_matthieu.jpg'
# image_path = '../../../samples/moto_man.jpg'
image_path = '../../../samples/car_nico.jpg'
# image_path = '../../../samples/bike.jpg'

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
print id, labels[id]

# Mask
mask0 = compute_mask_from_prediction(pred, id)

mask1 = foo(pred, id)

quad2 = bar(pred, id)

quad3 = grapbcut_quadmap(mask0)

# print quad3.shape


mask2 = compute_grabcut_mask_from_quadmap(quad2, img_rgb_small)
mask3 = compute_grabcut_mask_from_quadmap(quad3, img_rgb_small)


red0 = generate_red_illustration(img_rgb_small, mask0, apply_rgb2bgr_conversion=False)
red2 = generate_red_illustration(img_rgb_small, mask2, apply_rgb2bgr_conversion=False)
red3 = generate_red_illustration(img_rgb_small, mask3, apply_rgb2bgr_conversion=False)


sticker0 = generate_sticker_illustration(img_rgb_small, mask0, apply_rgb2bgr_conversion=False)
sticker2 = generate_sticker_illustration(img_rgb_small, mask2, apply_rgb2bgr_conversion=False)
sticker3 = generate_sticker_illustration(img_rgb_small, mask3, apply_rgb2bgr_conversion=False)



top = np.concatenate([red0, red3, red2], axis=1)
bottom = np.concatenate([sticker0, sticker3, sticker2], axis=1)
output = np.concatenate([top, bottom], axis=0)
plt.figure()
plt.imshow(output)
plt.axis('off')
plt.show()
# exit()

cv2.imwrite('/Users/vincent/Desktop/car_nico.png', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))




# f, ax = plt.subplots(2,4)
# ax[0, 0].imshow(mask0, cmap='gray')
# ax[0, 1].imshow(mask1, cmap='gray')
# ax[0, 2].imshow(quad2, cmap='gray')
# ax[0, 3].imshow(quad3, cmap='gray')
# ax[1, 0].imshow(mask2, cmap='gray')
# ax[1, 1].imshow(red2, cmap='gray')
# ax[1, 2].imshow(mask3, cmap='gray')
# ax[1, 3].imshow(red3, cmap='gray')
# ax[0, 0].set_axis_off()
# ax[0, 1].set_axis_off()
# ax[0, 2].set_axis_off()
# ax[0, 3].set_axis_off()
# ax[1, 0].set_axis_off()
# ax[1, 1].set_axis_off()
# ax[1, 2].set_axis_off()
# ax[1, 3].set_axis_off()
# ax[0, 0].set_title('(0) argmax mask')
# ax[0, 1].set_title('(1) prediction')
# ax[0, 2].set_title('(2) threshold on prediction')
# ax[0, 3].set_title('(3) morpho math on argmax')
# ax[1, 0].set_title('grabcut on (2)')
# ax[1, 1].set_title('grabcut on (2)')
# ax[1, 2].set_title('grabcut on (3)')
# ax[1, 3].set_title('grabcut on (3)')
# plt.show()
# exit()

