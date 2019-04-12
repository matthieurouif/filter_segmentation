import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import coremltools

input_folder = '../../../samples'

paths = glob.glob(os.path.join(input_folder, '*.jpg'))
for path in paths:

	# File name
	name = os.path.split(path)[1]
	name_no_ext = os.path.splitext(name)[0]
	print('File name : ', name)

	# Read the image
	img_rgb = Image.open(path)
	# plt.figure()
	# plt.imshow(img_rgb)
	# plt.show()

	# Load the model and it's specs
	model = coremltools.models.MLModel('../../../models/DeepLab.mlmodel')
	# input_width = model.get_spec().description.input[0].type.imageType.width
	# input_height = model.get_spec().description.input[0].type.imageType.height

	# # Resize the image to fit model's input size
	# small_img = img_rgb.resize((input_width, input_height), Image.BILINEAR)

	# # Predict the classes
	# y = model.predict({"image": small_img}, usesCPUOnly=False)
	# small_pred = y['scores']

	# # Resize the prediction to input image size
	# pred = small_pred.resize((img_rgb.width, img_rgb.height), PIL.Image.NEAREST)

	# # Plot
	# f, ax = plt.subplots(1,3)
	# ax[0].imshow(img_rgb)
	# ax[1].imshow(img_y, cmap='gray')
	# ax[2].imshow(pred)
	# ax[0].set_axis_off()
	# ax[1].set_axis_off()
	# ax[2].set_axis_off()
	# plt.show()


	break