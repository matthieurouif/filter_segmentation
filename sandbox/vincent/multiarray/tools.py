import cv2

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt


# List of labels
labels = ['Background','Plane','Bicycle','Bird','Boat','Bottle','Bus','Car','Cat','Chair','Cow','Diningtable','Dog','Horse','Motorbike','Person','Pottedplant','Sheep','Sofa','Train','Tvmonitor']
label2id = {'Background':0,'Plane':1,'Bicycle':2,'Bird':3,'Boat':4,'Bottle':5,'Bus':6,'Car':7,'Cat':8,'Chair':9,'Cow':10,'Diningtable':11,'Dog':12,'Horse':13,'Motorbike':14,'Person':15,'Pottedplant':16,'Sheep':17,'Sofa':18,'Train':19,'Tvmonitor':20}


def scale_to_fit(input, max_size=1024, interpolation=cv2.INTER_LINEAR):

	# Input image size
	input_height, input_width, _ = input.shape

	# If the image is smaller than max_size, just return
	if max(input_height, input_width) <= max_size:
		return input

	# Compute output image size
	if input_width >= input_height:
	    output_width = max_size
	    output_height = input_height * max_size / input_width
	else:
	    output_height = max_size
	    output_width = input_width * max_size / input_height

	# Resize
	output = cv2.resize(input, (output_width, output_height), interpolation=interpolation)
	return output


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
	# 	return label2id['Person']
	else:
		return np.argmax(count)


def get_main_class_from_segmentation(seg):

	# Count the number of pixels for each label
	count = []
	for id in range(len(labels)):
		count.append(np.sum(seg == id) * 100. / seg.size)

	# Ignore the background pixels
	count[0] = 0

	# Get the most seen label
	if np.sum(count) == 0:
		return label2id['Background']
	# if count[label2id['Person']] > 0:
	# 	return label2id['Person']
	else:
		return np.argmax(count)


def compute_mask_from_prediction(pred, id):
	seg = np.argmax(pred, axis = 0)
	return compute_mask_from_segmentation(seg, id)


def compute_mask_from_segmentation(seg, id):
	mask = np.zeros(seg.shape, np.uint8)
	mask[seg == id] = 1
	return mask


def generate_red_illustration(rgb, mask, apply_rgb2bgr_conversion=True):
	output = rgb.copy()
	output[:, :, 0][mask == 1] = 255
	if apply_rgb2bgr_conversion:
		return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
	else:
		return output


def compute_guided_filter_segmentation(img_y, pred):
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
	return argmax


def compute_grabcut_mask(mask, img):
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

	# Apply the GrabCut algorithm
	output_mask = grabcut_mask.copy()
	bgdModel = np.zeros((1,65), np.float64)
	fgdModel = np.zeros((1,65), np.float64)	
	cv2.grabCut(img, output_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

	# Deduce final mask
	output_mask = np.where((output_mask==2)|(output_mask==0), 0, 1).astype('uint8')
	return output_mask


def compute_grabcut_segmentation(img_rgb, pred):
	argmax = np.argmax(pred, axis = 0)
	seg = np.zeros(argmax.shape, np.uint8)
	for c in range(pred.shape[0]):
		# print '  Channel ID :', c
		mask_in = np.zeros(argmax.shape, np.uint8)
		mask_in[argmax == c] = 1
		if np.sum(mask_in) > 0:
			mask_out = compute_grabcut_mask(mask_in, img_rgb)
			seg[mask_out==1] = c
	return seg

