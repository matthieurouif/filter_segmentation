import coremltools
import os
import matplotlib.pyplot as plt
import cv2
from glob import glob
import sys

from tools import *


def trimap_v1(mask):

    # Element
    radius = 10 # int(max(mask.shape) / 50)
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


def trimap_v2(layer):

    t0 = 0.05
    t1 = 0.95

    sure_fg = layer >= t1
    sure_bg = layer <= t0
    pr_fg = np.logical_and(layer >= 0.5, layer < t1)
    pr_bg = np.logical_and(layer < 0.5, layer >= t0)

    trimap = np.zeros(layer.shape, dtype=np.uint8)
    trimap[sure_bg] = cv2.GC_BGD
    trimap[sure_fg] = cv2.GC_FGD
    trimap[pr_bg] = cv2.GC_PR_BGD
    trimap[pr_fg] = cv2.GC_PR_FGD

    return trimap


def trimap_v3(layer):

    layer2 = cv2.GaussianBlur(layer, (21,21), 3)

    t0 = 0.05
    t1 = 0.95

    sure_fg = layer2 >= t1
    sure_bg = layer2 <= t0
    pr_fg = np.logical_and(layer2 >= 0.5, layer2 < t1)
    pr_bg = np.logical_and(layer2 < 0.5, layer2 >= t0)

    trimap = np.zeros(layer.shape, dtype=np.uint8)
    trimap[sure_bg] = cv2.GC_BGD
    trimap[sure_fg] = cv2.GC_FGD
    trimap[pr_bg] = cv2.GC_PR_BGD
    trimap[pr_fg] = cv2.GC_PR_FGD

    return trimap


def process_image(model, input_path, output_path):

    # Read the input image
    img_bgr_full = cv2.imread(input_path)
    img_rgb_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB)

    # Scale the color image
    max_size = 1024
    img_rgb_small = scale_to_fit(img_rgb_full, max_size, cv2.INTER_LINEAR)
    height, width, _ = img_rgb_small.shape

    # Prediction
    pred = predict(model, img_rgb_small)

    # Id we are interested in
    id = get_main_class(pred)

    mask0 = compute_mask_from_prediction(pred, id)
    red0 = generate_red_illustration(img_rgb_small, mask0)
    sticker0 = generate_sticker_illustration(img_rgb_small, mask0)

    trimap1 = trimap_v1(mask0)
    mask1 = compute_grabcut_mask_from_quadmap(trimap1, img_rgb_small)
    red1 = generate_red_illustration(img_rgb_small, mask1)
    sticker1 = generate_sticker_illustration(img_rgb_small, mask1)

    layer_id = pred[id, :, :]
    trimap2 = trimap_v2(layer_id)
    mask2 = compute_grabcut_mask_from_quadmap(trimap2, img_rgb_small)
    red2 = generate_red_illustration(img_rgb_small, mask2)
    sticker2 = generate_sticker_illustration(img_rgb_small, mask2)

    trimap3 = trimap_v3(layer_id)
    mask3 = compute_grabcut_mask_from_quadmap(trimap3, img_rgb_small)
    red3 = generate_red_illustration(img_rgb_small, mask3)
    sticker3 = generate_sticker_illustration(img_rgb_small, mask3)

    f, ax = plt.subplots(4,4)
    ax[0,0].imshow(mask0)
    ax[0,1].imshow(trimap1)
    ax[0,2].imshow(trimap2)
    ax[0,3].imshow(trimap3)
    ax[1,0].imshow(mask0)
    ax[1,1].imshow(mask1)
    ax[1,2].imshow(mask2)
    ax[1,3].imshow(mask3)
    ax[2,0].imshow(red0)
    ax[2,1].imshow(red1)
    ax[2,2].imshow(red2)
    ax[2,3].imshow(red3)
    ax[3,0].imshow(sticker0)
    ax[3,1].imshow(sticker1)
    ax[3,2].imshow(sticker2)
    ax[3,3].imshow(sticker3)
    plt.show()
    exit()

    row0 = np.concatenate([red0, red1, red2, red3], axis=1)
    row1 = np.concatenate([sticker0, sticker1, sticker2, sticker3], axis=1)
    total = np.concatenate([row0, row1], axis=0)

    # plt.figure()
    # plt.imshow(total)
    # plt.axis('off')
    # plt.show()
    # exit()

    cv2.imwrite(output_path, cv2.cvtColor(total, cv2.COLOR_RGB2BGR))


def main():
    
    # Paths
    model_path = '../../../models/MultiArrayDeepLab.mlmodel'
    output_folder = '/Users/vincent/background/results/2019.05.13_grabcut.comparision'

    # Read the model
    model = coremltools.models.MLModel(model_path)

    # Image path
    # input_path = '../../../samples/person_matthieu.jpg'
    # input_path = '../../../samples/moto_man.jpg'
    input_path = '../../../samples/dog.jpg'
    # input_path = '/Users/vincent/Downloads/VOCdevkit/VOC2012/JPEGImages/2007_000452.jpg'

    base = os.path.basename(input_path)
    name = os.path.splitext(base)[0]
    output_path = os.path.join(output_folder, name + '.png') 
    print output_path

    process_image(model, input_path, output_path)


def main2():

    # Paths
    model_path = '../../../models/MultiArrayDeepLab.mlmodel'
    input_folder = '/Users/vincent/data/Work/Projects/filter_segmentation/samples'
    output_folder = '/Users/vincent/background/results/2019.05.13_grabcut.comparision'

    # Read the model
    model = coremltools.models.MLModel(model_path)

    image_path_list = sorted(glob(os.path.join(input_folder, '*.jpg')))
    for input_path in image_path_list:

        print input_path
        sys.stdout.flush()

        base = os.path.basename(input_path)
        name = os.path.splitext(base)[0]
        output_path = os.path.join(output_folder, name + '.png') 

        process_image(model, input_path, output_path)


def main3():

    # Paths
    model_path = '../../../models/MultiArrayDeepLab.mlmodel'
    output_folder = '/Users/vincent/background/results/2019.05.13_grabcut.comparision'

    # Read the model
    model = coremltools.models.MLModel(model_path)

    # Image path
    # image_path = '../../../samples/person_matthieu.jpg'
    # image_path = '../../../samples/moto_man.jpg'
    input_path = '../../../samples/train_2.jpg'
    # image_path = '/Users/vincent/Downloads/VOCdevkit/VOC2012/JPEGImages/2007_000452.jpg'

    # Read the input image
    img_bgr_full = cv2.imread(input_path)
    img_rgb_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB)

    # Scale the color image
    max_size = 1024
    img_rgb_small = scale_to_fit(img_rgb_full, max_size, cv2.INTER_LINEAR)
    height, width, _ = img_rgb_small.shape

    # Prediction
    pred = predict(model, img_rgb_small)

    # Id we are interested in
    id = get_main_class(pred)
    layer_id = pred[id, :, :]

    mask0 = compute_mask_from_prediction(pred, id)

    trimap = trimap_v3(layer_id)

    mask1 = compute_grabcut_mask_from_quadmap(trimap, img_rgb_small, 0)
    mask2 = compute_grabcut_mask_from_quadmap(trimap, img_rgb_small, 2)
    mask3 = compute_grabcut_mask_from_quadmap(trimap, img_rgb_small, 5)
    mask4 = compute_grabcut_mask_from_quadmap(trimap, img_rgb_small, 10)
    red0 = generate_red_illustration(img_rgb_small, mask0)
    red1 = generate_red_illustration(img_rgb_small, mask1)
    red2 = generate_red_illustration(img_rgb_small, mask2)
    red3 = generate_red_illustration(img_rgb_small, mask3)
    red4 = generate_red_illustration(img_rgb_small, mask4)


    row0 = np.concatenate([mask0, mask1, mask2, mask3, mask4], axis=1)
    row1 = np.concatenate([red0, red1, red2, red3, red4], axis=1)
    # total = np.concatenate([row0, row1], axis=0)

    # plt.figure()
    # plt.imshow(row1)
    # plt.axis('off')
    # plt.show()
    # exit()

    f, ax = plt.subplots(2,1)
    ax[0].imshow(row0)
    ax[1].imshow(row1)
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.show()



if __name__ == '__main__':
    main()
    # main2()
    # main3()
