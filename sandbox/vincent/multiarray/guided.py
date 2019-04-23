import coremltools
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from PIL import Image
from glob import glob
from time import time


# List of labels
labels = ['Background','Plane','Bicycle','Bird','Boat','Bottle','Bus','Car','Cat','Chair','Cow','Diningtable','Dog','Horse','Motorbike','Person','Pottedplant','Sheep','Sofa','Train','Tvmonitor']
label2id = {'Background':0,'Plane':1,'Bicycle':2,'Bird':3,'Boat':4,'Bottle':5,'Bus':6,'Car':7,'Cat':8,'Chair':9,'Cow':10,'Diningtable':11,'Dog':12,'Horse':13,'Motorbike':14,'Person':15,'Pottedplant':16,'Sheep':17,'Sofa':18,'Train':19,'Tvmonitor':20}


def scale_down(input_img):

	# Max size
	max_size = 1024

	# Input image size
	input_height, input_width, _ = input_img.shape

	# If the image is smaller than 1024, just return
	if max(input_height, input_width) <= max_size:
		return input_img

	# Compute output image size
	if input_width >= input_height:
	    output_width = max_size
	    output_height = input_height * max_size / input_width
	else:
	    output_height = max_size
	    output_width = input_width * max_size / input_height

	# Resize
	input_small = cv2.resize(input_img, (output_width, output_height), interpolation = cv2.INTER_LINEAR)
	return input_small
	

def predict(model, img):

	# Get the model's input image size
	input_width = model.get_spec().description.input[0].type.imageType.width
	input_height = model.get_spec().description.input[0].type.imageType.height

	# Resize the image
	input_small = cv2.resize(img, (input_width, input_height), interpolation = cv2.INTER_AREA)

	# Convert it as a PIL image
	input_pil = Image.fromarray(input_small)

	# Predict using the model
	pred0 = model.predict({'ImageTensor__0': input_pil}, usesCPUOnly=False)
	pred1 = pred0['ResizeBilinear_3__0']
	pred2 = pred1.astype(np.float32)

	# Resize the prediction
	nb_labels = pred2.shape[0]
	height, width, _ = img.shape
	pred_resized = np.zeros((nb_labels, height, width), np.float32)
	for i in range(nb_labels):
		pred_resized[i, :, :] = cv2.resize(pred2[i, :, :], (width, height), interpolation = cv2.INTER_AREA)

	# Compute softmax
	pred3 = np.exp(pred_resized)
	pred4 = pred3 / np.sum(pred3, axis=0)
	pred5 = pred4.astype(np.float32)
	return pred5


def get_main_class(pred):

	# Find the most probable label for each pixel
	argmax = np.argmax(pred, axis = 0)

	# Count the number of pixels for each label
	count = []
	for id in range(len(labels)):
		count.append(np.sum(argmax == id) * 100. / argmax.size)

	# Ignore the background pixels
	count[0] = 0

	# Get the most seen label
	if np.sum(count) == 0:
		return label2id['Background']
	# if count[label2id['Person']] > 0:
	# 	# print count[label2id['Person']], '%%'
	# 	return label2id['Person']
	else:
		return np.argmax(count)


def get_main_class_from_groundtruth(gt):

	# Count the number of pixels for each label
	count = []
	for id in range(len(labels)):
		count.append(np.sum(gt == id) * 100. / gt.size)

	# Ignore the background pixels
	count[0] = 0

	# Get the most seen label
	if np.sum(count) == 0:
		return label2id['Background']
	if count[label2id['Person']] > 0:
		# print count[label2id['Person']], '%%'
		return label2id['Person']
	else:
		return np.argmax(count)


def compute_groundtruth_mask(classif, id):
	mask = np.zeros(classif.shape, np.uint8)
	mask[classif == id] = 1
	return mask


def compute_argmax_mask(pred, id):
	argmax = np.argmax(pred, axis = 0)
	mask = np.zeros(argmax.shape, np.uint8)
	mask[argmax == id] = 1
	return mask


def compute_guided_filter_mask(img_y, pred, id):
	# Parameters
	radius = 10
	eps = 0.001

	# Convert image
	img_y2 = img_y.astype(np.float32) / 255.

	# Apply guided filter on each prediction channel
	pred2 = pred.copy()
	for c in range(pred.shape[0]):
	    channel = pred[c,:,:]
	    pred2[c,:,:] = np.clip(cv2.ximgproc.guidedFilter(img_y2, channel, radius, eps), 0., 1.)

	# Compute the mask
	argmax = np.argmax(pred2, axis = 0)
	mask = np.zeros(argmax.shape, np.uint8)
	mask[argmax == id] = 1
	return mask


def generate_red_illustration(rgb, mask):
	output = rgb.copy()
	output[:, :, 0][mask == 1] = 255
	return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


def main():
	print(sys.version)

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
	img_rgb = scale_down(img_rgb_full)
	img_y = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

	# Prediction
	pred = predict(model, img_rgb)

	# Get the main class
	id = get_main_class(pred)
	# print('Main class = %d -> %s' % (id, labels[id]))

	# Get segmentation uing argmax on prediciton
	mask0 = compute_argmax_mask(pred, id)
	rgb0 = generate_red_illustration(img_rgb, mask0)

	# Get segmentation uing guided filter on prediciton
	mask1 = compute_guided_filter_mask(img_y, pred, id)
	rgb1 = generate_red_illustration(img_rgb, mask1)

	plt.figure()
	plt.imshow(mask1)
	plt.show()


def process_all_images():

	# Paths
	model_path = '../../../models/MultiArrayDeepLab.mlmodel'
	input_folder = '../../../samples/'
	output_folder = '/Users/vincent/Desktop/results/'

	# Create output folder if needed
	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)

	# Read the model
	model = coremltools.models.MLModel(model_path)

	# Go through all images
	for image_path in sorted(glob(os.path.join(input_folder, '*.jpg'))):

		print('Processing %s' % image_path)

		# Get image name
		image_name = os.path.splitext(os.path.split(image_path)[1])[0]

		# Read the input image
		img_bgr_full = cv2.imread(image_path)
		img_rgb_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB)

		# Resize the image to fit 1024x1024
		img_rgb = scale_down(img_rgb_full)
		img_y = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

		# Prediction
		pred = predict(model, img_rgb)

		# Get the main class
		id = get_main_class(pred)
		# continue

		# Get segmentation uing argmax on prediciton
		tic = time()
		mask0 = compute_argmax_mask(pred, id)
		toc = time()
		print('  Argmax        : %4d ms' % (int(1000 * (toc - tic))))
		bgr0 = generate_red_illustration(img_rgb, mask0)
		path0 = os.path.join(output_folder, image_name + '_' + labels[id] + '_0_argmax.png')
		cv2.imwrite(path0, bgr0)

		# Get segmentation uing guided filter on prediciton
		tic = time()
		mask1 = compute_guided_filter_mask(img_y, pred, id)
		toc = time()
		print('  Guided filter : %4d ms' % (int(1000 * (toc - tic))))
		bgr1 = generate_red_illustration(img_rgb, mask1)
		path1 = os.path.join(output_folder, image_name + '_' + labels[id] + '_1_guided_filter.png')
		cv2.imwrite(path1, bgr1)
		
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


def guided_filter_precision():

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
		if count % 10 == 0:
			print('Image         : %5d / %5d' % (count, len(classif_path_list)))
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
	

if __name__ == '__main__':

	# main()
	# process_all_images()
	guided_filter_precision()
