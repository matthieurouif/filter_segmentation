import coremltools
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from PIL import Image
from glob import glob
from time import time

from tools import *


def main():

	# Paths
	model_path = '../../../models/MultiArrayDeepLab.mlmodel'

	# Read the model
	model = coremltools.models.MLModel(model_path)

	# Image path
	image_path = '../../../samples/woman_car.jpg'

	# Get image name
	image_name = os.path.splitext(os.path.split(image_path)[1])[0]

	# Read the input image
	img_bgr_full = cv2.imread(image_path)
	img_rgb_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB)

	# Resize the image to fit 1024x1024
	img_rgb = scale_to_fit(img_rgb_full, 1024)
	img_y = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

	# Prediction
	pred = predict(model, img_rgb)

	# Get the main class
	id = get_main_class(pred)
	# print('Main class = %d -> %s' % (id, labels[id]))

	# Get segmentation uing argmax on prediciton
	mask0 = compute_mask_from_prediction(pred, id)
	rgb0 = generate_red_illustration(img_rgb, mask0)

	# Get segmentation uing guided filter on prediciton
	seg1 = compute_guided_filter_segmentation(img_y, pred)
	mask1 = compute_mask_from_segmentation(seg1, id)
	rgb1 = generate_red_illustration(img_rgb, mask1)

	plt.figure()
	plt.imshow(mask1)
	plt.show()


def process_all_images():

	# Paths
	model_path = '../../../models/MultiArrayDeepLab.mlmodel'
	input_folder = '../../../samples/'
	output_folder = '/Users/vincent/background/results/'

	# Create output folder if needed
	if not os.path.isdir(output_folder):
		os.makedirs(output_folder)

	# Read the model
	model = coremltools.models.MLModel(model_path)

	# Go through all images
	for image_path in sorted(glob(os.path.join(input_folder, '*.jpg'))):

		print('Processing %s' % image_path)
		sys.stdout.flush()

		# Get image name
		image_name = os.path.splitext(os.path.split(image_path)[1])[0]

		# Read the input image
		img_bgr_full = cv2.imread(image_path)
		img_rgb_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB)

		# Resize the image to fit 1024x1024
		img_rgb = scale_to_fit(img_rgb_full, 1024)
		img_y = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

		# Prediction
		pred = predict(model, img_rgb)

		# Get the main class
		id = get_main_class(pred)

		# Get segmentation uing argmax on prediciton
		tic = time()
		mask0 = compute_mask_from_prediction(pred, id)
		toc = time()
		print('  Argmax        : %4d ms' % (int(1000 * (toc - tic))))
		bgr0 = generate_red_illustration(img_rgb, mask0)
		path0 = os.path.join(output_folder, image_name + '_' + labels[id] + '_0_argmax.png')
		cv2.imwrite(path0, bgr0)

		# Get segmentation uing guided filter on prediciton
		tic = time()
		seg1 = compute_guided_filter_segmentation(img_y, pred)
		mask1 = compute_mask_from_segmentation(seg1, id)
		toc = time()
		print('  Guided filter : %4d ms' % (int(1000 * (toc - tic))))
		bgr1 = generate_red_illustration(img_rgb, mask1)
		path1 = os.path.join(output_folder, image_name + '_' + labels[id] + '_1_guided_filter.png')
		cv2.imwrite(path1, bgr1)

		# Get segmentation uing grabcut on prediciton
		tic = time()
		mask2 = compute_grabcut_mask(mask0, img_rgb)
		toc = time()
		print('  GrabCut       : %4d ms' % (int(1000 * (toc - tic))))
		bgr2 = generate_red_illustration(img_rgb, mask2)
		path2 = os.path.join(output_folder, image_name + '_' + labels[id] + '_2_grabcut.png')
		cv2.imwrite(path2, bgr2)
		
		# break


def compute_score(gt_mask, test_mask):

	tp = np.logical_and(gt_mask, test_mask).astype(np.uint8)
	tn = np.logical_not(np.logical_or(gt_mask, test_mask)).astype(np.uint8)
	fp = np.logical_and(test_mask, np.logical_not(gt_mask)).astype(np.uint8)
	fn = np.logical_and(gt_mask, np.logical_not(test_mask)).astype(np.uint8)

	# Accuracy
	accuracy = 100. * (np.sum(tp) + np.sum(tn)) / float(gt_mask.size)

	# IoU
	intersection = np.logical_and(gt_mask, test_mask)
	union = np.logical_or(gt_mask, test_mask)
	iou = 100. * np.sum(intersection) / float(np.sum(union))

	# sensitivity = np.sum(tp) / float(np.sum(tp) + np.sum(fn))
	# specificity = np.sum(tn) / float(np.sum(tn) + np.sum(fp))
	# precision = 100. * np.sum(tp) / float(np.sum(tp) + np.sum(fp))
	# recall = 100. * np.sum(tp) / float(np.sum(tp) + np.sum(fn))

	return accuracy, iou


def measure_guided_filter():

	# Paths
	pascalvoc2012_folder = '/Users/vincent/Downloads/VOCdevkit/VOC2012'
	model_path = '../../../models/MultiArrayDeepLab.mlmodel'

	# Read the model
	model = coremltools.models.MLModel(model_path)

	# Containers for precision, recall, accuracy and iou
	naive_precision = []
	naive_recall = []
	naive_accuracy = []
	naive_iou = []
	guided_precision = []
	guided_recall = []
	guided_accuracy = []
	guided_iou = []

	# List of classification images
	classif_path_list = sorted(glob(os.path.join(pascalvoc2012_folder, 'SegmentationClass', '*.png')))
	count = 0
	tic = time()
	for classif_path in classif_path_list:

		count += 1

		# Get image name and path
		image_name = os.path.splitext(os.path.split(classif_path)[1])[0]
		image_path = os.path.join(pascalvoc2012_folder, 'JPEGImages', image_name + '.jpg')

		# Grountruth classification
		gt_classif = np.array(Image.open(classif_path))

		# Main class based on the grountruth
		id = get_main_class_from_groundtruth(gt_classif)

		# Grountruth mask
		gt_mask = compute_groundtruth_mask(gt_classif, id)

		# Grountruth segmentation
		gt_seg = gt_classif
		gt_seg[gt_seg == 255] = 0

		# Read the input image
		img_bgr = cv2.imread(image_path)
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		img_y = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

		# Prediction
		pred = predict(model, img_rgb)

		# Naive mask
		naive_mask = compute_argmax_mask(pred, id)

		# Compute naive scores
		# precision, recall, acc, iou = compute_score(gt_mask, naive_mask)
		acc, iou = compute_score(gt_mask, naive_mask)
		# naive_precision.append(precision)
		# naive_recall.append(recall)
		naive_accuracy.append(acc)
		naive_iou.append(iou)

		# Guided filter mask
		guided_mask = compute_guided_filter_mask(img_y, pred, id)

		# Compute naive scores
		# precision, recall, acc, iou = compute_score(gt_mask, guided_mask)
		acc, iou = compute_score(gt_mask, guided_mask)
		# guided_precision.append(precision)
		# guided_recall.append(recall)
		guided_accuracy.append(acc)
		guided_iou.append(iou)

		# Print during iterations
		if count % 100 == 0:
			toc = time()
			print('Image         : %5d / %5d (%.1f s)' % (count, len(classif_path_list), toc-tic))
			# print('Naive         : Precision = %5.2f %%, Recall = %5.2f %%, Accuracy = %5.2f %%, IoU = %5.2f %%' % (np.mean(naive_precision), np.mean(naive_recall), np.mean(naive_accuracy), np.mean(naive_iou)))
			# print('Guided filter : Precision = %5.2f %%, Recall = %5.2f %%, Accuracy = %5.2f %%, IoU = %5.2f %%' % (np.mean(guided_precision), np.mean(guided_recall), np.mean(guided_accuracy), np.mean(guided_iou)))
			print('Naive         : Accuracy = %5.2f %%, IoU = %5.2f %%' % (np.mean(naive_accuracy), np.mean(naive_iou)))
			print('Guided filter : Accuracy = %5.2f %%, IoU = %5.2f %%' % (np.mean(guided_accuracy), np.mean(guided_iou)))
			print
			sys.stdout.flush()
			# break

	# print('Naive         : Precision = %5.2f %%, Recall = %5.2f %%, Accuracy = %5.2f %%, IoU = %5.2f %%' % (np.mean(naive_precision), np.mean(naive_recall), np.mean(naive_accuracy), np.mean(naive_iou)))
	# print('Guided filter : Precision = %5.2f %%, Recall = %5.2f %%, Accuracy = %5.2f %%, IoU = %5.2f %%' % (np.mean(guided_precision), np.mean(guided_recall), np.mean(guided_accuracy), np.mean(guided_iou)))
	print('Naive         : Accuracy = %5.2f %%, IoU = %5.2f %%' % (np.mean(naive_accuracy), np.mean(naive_iou)))
	print('Guided filter : Accuracy = %5.2f %%, IoU = %5.2f %%' % (np.mean(guided_accuracy), np.mean(guided_iou)))


def update_confusion_matrix(seg0, seg1, matrix, nb_classes):
	# for y, x in zip(seg0, seg1):
	# 	matrix[y,x] += 1
	for x in np.ravel_multi_index((seg0, seg1), (nb_classes, nb_classes)):
		matrix.flat[x] += 1


def compute_miou(matrix):
	intersection = matrix.diagonal()
	sum_h = matrix.sum(axis=0)
	sum_v = matrix.sum(axis=1)
	union = sum_h + sum_v - intersection
	iou = intersection / union.astype(np.float64)
	miou = np.mean(iou).astype(np.float32)
	return miou


def measure_miou():

	# Paths
	pascalvoc2012_folder = '/Users/vincent/Downloads/VOCdevkit/VOC2012'
	model_path = '../../../models/MultiArrayDeepLab.mlmodel'

	# Read the model
	model = coremltools.models.MLModel(model_path)

	# Initialize the confusion matrices
	nb_classes = len(labels)
	naive_matrix = np.zeros((nb_classes, nb_classes), dtype=np.int64)
	guided_matrix = np.zeros((nb_classes, nb_classes), dtype=np.int64)
	grabcut_matrix = np.zeros((nb_classes, nb_classes), dtype=np.int64)

	# List of classification images
	classif_path_list = sorted(glob(os.path.join(pascalvoc2012_folder, 'SegmentationClass', '*.png')))
	count = 0
	tic = time()
	for classif_path in classif_path_list:

		# Get image name and path
		image_name = os.path.splitext(os.path.split(classif_path)[1])[0]
		image_path = os.path.join(pascalvoc2012_folder, 'JPEGImages', image_name + '.jpg')

		print image_path
		sys.stdout.flush()

		# Grountruth segmentation
		gt_seg = np.array(Image.open(classif_path))
		gt_seg[gt_seg == 255] = 0  # Remap 'void' pixels as background
		gt_seg = gt_seg.flatten()

		# f,ax = plt.subplots(1,2)
		# ax[0].imshow(gt_seg)
		# ax[1].imshow(tmp)
		# plt.show()

		# Read the input image
		img_bgr = cv2.imread(image_path)
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		img_y = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

		# Prediction
		pred = predict(model, img_rgb)

		# Naive segmentation
		naive_seg = np.argmax(pred, axis=0)
		naive_seg = naive_seg.flatten()

		# Update the confusion matrix
		update_confusion_matrix(gt_seg, naive_seg, naive_matrix, nb_classes)

		# Guided filter segmentation
		guided_seg = compute_guided_filter_segmentation(img_y, pred)
		guided_seg = guided_seg.flatten()

		# Update the confusion matrix
		update_confusion_matrix(gt_seg, guided_seg, guided_matrix, nb_classes)

		# Grabcut segmentation
		grabcut_seg = compute_grabcut_segmentation(img_rgb, pred)
		grabcut_seg = grabcut_seg.flatten()

		# Update the confusion matrix
		update_confusion_matrix(gt_seg, grabcut_seg, grabcut_matrix, nb_classes)

		# Print
		count += 1
		if count % 100 == 0:
			toc = time()
			print('Image        : %5d / %5d (%.1f s)' % (count, len(classif_path_list), toc-tic))
			print('Naive  mIoU  : %5.2f %%' % (100. * compute_miou(naive_matrix)))
			print('Guided mIoU  : %5.2f %%' % (100. * compute_miou(guided_matrix)))
			print('Grabcut mIoU : %5.2f %%' % (100. * compute_miou(grabcut_matrix)))
			print
			sys.stdout.flush()
			tic = toc

	# Final print
	print('Naive  mIoU  : %5.2f %%' % (100. * compute_miou(naive_matrix)))
	print('Guided mIoU  : %5.2f %%' % (100. * compute_miou(guided_matrix)))
	print('Grabcut mIoU : %5.2f %%' % (100. * compute_miou(grabcut_matrix)))


if __name__ == '__main__':

	# main()
	process_all_images()
	# measure_guided_filter()
	# measure_miou()
