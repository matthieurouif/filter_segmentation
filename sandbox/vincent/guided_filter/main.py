import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc

def get_region(arr, x, y, radius):
	x_start = max(0, x-radius)
	x_end = min(arr.shape[1], x+radius+1)
	y_start = max(0, y-radius)
	y_end = min(arr.shape[0], y+radius+1)
	return x_start, x_end, y_start, y_end

def mean(arr, x, y, radius):
	x_start, x_end, y_start, y_end = get_region(arr, x, y, radius)
	return np.mean(arr[y_start:y_end, x_start:x_end])

def var(arr, x, y, radius):
	x_start, x_end, y_start, y_end = get_region(arr, x, y, radius)
	return np.var(arr[y_start:y_end, x_start:x_end])

def count(arr, x, y, radius):
	x_start, x_end, y_start, y_end = get_region(arr, x, y, radius)
	return (x_end - x_start) * (y_end - y_start)

def guided_filter(I, p, radius, eps):

	# Compute I * p
	Ip = np.multiply(I, p)

	# Compute some values
	I_mean = np.zeros(I.shape, dtype=np.float32)
	p_mean = np.zeros(I.shape, dtype=np.float32)
	I_var = np.zeros(I.shape, dtype=np.float32)
	Ip_mean = np.zeros(I.shape, dtype=np.float32)
	for y in range(I.shape[0]):
		for x in range(I.shape[1]):
			I_mean[y,x] = mean(I, x, y, radius)
			p_mean[y,x] = mean(p, x, y, radius)
			Ip_mean[y,x] = mean(Ip, x, y, radius)
			I_var = var(I, x, y, radius)

	# Compute a and b
	a = (Ip_mean - np.multiply(I_mean, p_mean)) / (I_var + eps)
	b = p_mean - np.multiply(a, I_mean)

	# Deduce a_mean and b_mean
	a_mean = np.zeros(I.shape, dtype=np.float32)
	b_mean = np.zeros(I.shape, dtype=np.float32)
	for y in range(I.shape[0]):
		for x in range(I.shape[1]):
			a_mean[y,x] = mean(a, x, y, radius)
			b_mean[y,x] = mean(b, x, y, radius)

	# Finally get the output mask
	q = np.multiply(a_mean, I) + b_mean

	# Clip values in [0,1]
	q = np.clip(q, 0, 1)

	return q

# def box(img, r):
#     """ O(1) box filter
#         img - >= 2d image
#         r   - radius of box filter
#     """
#     (rows, cols) = img.shape[:2]
#     imDst = np.zeros_like(img)


#     tile = [1] * img.ndim
#     tile[0] = r
#     imCum = np.cumsum(img, 0)
#     imDst[0:r+1, :, ...] = imCum[r:2*r+1, :, ...]
#     imDst[r+1:rows-r, :, ...] = imCum[2*r+1:rows, :, ...] - imCum[0:rows-2*r-1, :, ...]
#     imDst[rows-r:rows, :, ...] = np.tile(imCum[rows-1:rows, :, ...], tile) - imCum[rows-2*r-1:rows-r-1, :, ...]

#     tile = [1] * img.ndim
#     tile[1] = r
#     imCum = np.cumsum(imDst, 1)
#     imDst[:, 0:r+1, ...] = imCum[:, r:2*r+1, ...]
#     imDst[:, r+1:cols-r, ...] = imCum[:, 2*r+1 : cols, ...] - imCum[:, 0 : cols-2*r-1, ...]
#     imDst[:, cols-r: cols, ...] = np.tile(imCum[:, cols-1:cols, ...], tile) - imCum[:, cols-2*r-1 : cols-r-1, ...]

#     return imDst

if __name__ == '__main__':
	
	# Paths
	image_path = 'source.tiff'
	mask_path = 'mask.tiff'


	# Load the image (grayscale) and the mask
	img = np.asarray(Image.open(image_path).convert('L'))#.copy()
	mask = np.asarray(Image.open(mask_path))[:,:,0]#.copy()

	# Normalize images to float in [0,1]
	# Change the name to fit the paper
	I = img.astype(np.float32) / 255.
	p = mask.astype(np.float32) / 255.

	# Filter the mask
	radius = 10
	eps = 0.01
	q = guided_filter(I, p, radius, eps)

	# Display
	plt.imshow(q, cmap='gray')
	plt.show()


# scipy.misc.imsave('mask1.png', p)
# scipy.misc.imsave('mask2.png', q)

# p.save("out1", "PNG")
# q.save("out2", "PNG")