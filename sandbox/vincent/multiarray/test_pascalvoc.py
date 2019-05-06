import coremltools
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

from glob import glob
from os import path, mkdir
from PIL import Image
from tools import *



if __name__ == '__main__':

	# Paths
	pascalvoc2012_folder = '/Users/vincent/Downloads/VOCdevkit/VOC2012'
	model_path = '../../../models/MultiArrayDeepLab.mlmodel'
	output_folder = '/Users/vincent/background/results/pascalvoc2012'

	# Create output folder if needed
	if not path.isdir(output_folder):
		mkdir(output_folder)

	# Read the model
	model = coremltools.models.MLModel(model_path)

	# List of classification images
	classif_path_list = sorted(glob(path.join(pascalvoc2012_folder, 'SegmentationClass', '*.png')))
	count = 0
	for classif_path in classif_path_list:
		
		# classif_path = '/Users/vincent/Downloads/VOCdevkit/VOC2012/SegmentationClass/2011_000771.png'

		print classif_path
		sys.stdout.flush()

		# Get image name and path
		image_name = path.splitext(path.split(classif_path)[1])[0]
		image_path = path.join(pascalvoc2012_folder, 'JPEGImages', image_name + '.jpg')

		# Grountruth segmentation
		gt_seg = np.array(Image.open(classif_path))
		gt_seg[gt_seg == 255] = 0  # Remap 'void' pixels to 'background'

		# Main class
		id = get_main_class_from_segmentation(gt_seg)

		# Read the input image
		img_bgr = cv2.imread(image_path)
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		img_y = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

		# Prediction
		pred = predict(model, img_rgb)

		# Naive segmentation
		naive_seg = np.argmax(pred, axis=0)

		# Guided filter segmentation
		guided_seg = compute_guided_filter_segmentation(img_y, pred)

		# Masks
		gt_mask = compute_mask_from_segmentation(gt_seg, id)
		naive_mask = compute_mask_from_segmentation(naive_seg, id)
		guided_mask = compute_mask_from_segmentation(guided_seg, id)
		grabcut_mask = compute_grabcut_mask(naive_mask, img_rgb)

		# Red overlay
		gt_red = generate_red_illustration(img_rgb, gt_mask)
		naive_red = generate_red_illustration(img_rgb, naive_mask)
		guided_red = generate_red_illustration(img_rgb, guided_mask)
		grabcut_red = generate_red_illustration(img_rgb, grabcut_mask)

		# Output path
		gt_out_path = path.join(output_folder, image_name + '_0_groundtruth.png')
		naive_out_path = path.join(output_folder, image_name + '_1_naive.png')
		guided_out_path = path.join(output_folder, image_name + '_2_guided.png')
		grabcut_out_path = path.join(output_folder, image_name + '_3_grabcut.png')

		# Write images
		cv2.imwrite(gt_out_path, gt_red)
		cv2.imwrite(naive_out_path, naive_red)
		cv2.imwrite(guided_out_path, guided_red)
		cv2.imwrite(grabcut_out_path, grabcut_red)

		count += 1
		if count > 200:
			break
