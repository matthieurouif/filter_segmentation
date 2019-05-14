import coremltools
import os
import matplotlib.pyplot as plt
import cv2
from glob import glob
import sys
import sklearn.neighbors
import scipy.sparse
import warnings
from tools import *

def trimap_v1(mask):

    # Element
    radius = int(max(mask.shape) / 100)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)

    # Sure foreground, background and unknown areas
    mask_ero = cv2.erode(mask, kernel)
    mask_dil = cv2.dilate(mask, kernel)
    sure_fg = mask_ero.astype(np.bool_)
    sure_bg = np.logical_not(mask_dil)

    # Sanity check
    if np.count_nonzero(sure_fg) == 0 or np.count_nonzero(sure_bg) == 0:
        return mask

    # Build input GrabCut mask
    trimap = np.zeros(mask.shape, dtype=np.uint8)
    trimap[:,:] = 128
    trimap[sure_bg] = 0
    trimap[sure_fg] = 255

    return trimap


def trimap_v2(layer):

    t0 = 0.05
    t1 = 0.95

    sure_fg = layer >= t1
    sure_bg = layer <= t0
    pr_fg = np.logical_and(layer >= 0.5, layer < t1)
    pr_bg = np.logical_and(layer < 0.5, layer >= t0)

    trimap = np.zeros(layer.shape, dtype=np.uint8)
    trimap[:,:] = 128
    trimap[sure_bg] = 0
    trimap[sure_fg] = 255

    return trimap


def knn_matte(img, trimap, mylambda=100):
    [m, n, c] = img.shape
    img, trimap = img/255.0, trimap/255.0
    foreground = (trimap > 0.99).astype(int)
    background = (trimap < 0.01).astype(int)
    all_constraints = foreground + background



    print('Finding nearest neighbors')
    a, b = np.unravel_index(np.arange(m*n), (m, n))
    feature_vec = np.append(np.transpose(img.reshape(m*n,c)), [ a, b]/np.sqrt(m*m + n*n), axis=0).T
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=10, n_jobs=4).fit(feature_vec)
    knns = nbrs.kneighbors(feature_vec)[1]

    # Compute Sparse A
    print('Computing sparse A')
    row_inds = np.repeat(np.arange(m*n), 10)
    col_inds = knns.reshape(m*n*10)
    vals = 1 - np.linalg.norm(feature_vec[row_inds] - feature_vec[col_inds], axis=1)/(c+2)
    A = scipy.sparse.coo_matrix((vals, (row_inds, col_inds)),shape=(m*n, m*n))

    D_script = scipy.sparse.diags(np.ravel(A.sum(axis=1)))
    L = D_script-A
    D = scipy.sparse.diags(np.ravel(all_constraints[:,:, 0]))
    v = np.ravel(foreground[:,:,0])
    c = 2*mylambda*np.transpose(v)
    H = 2*(L + mylambda*D)

    print('Solving linear system for alpha')
    warnings.filterwarnings('error')
    alpha = []
    try:
        alpha = np.minimum(np.maximum(scipy.sparse.linalg.spsolve(H, c), 0), 1).reshape(m, n)
    except Warning:
        x = scipy.sparse.linalg.lsqr(H, c)
        alpha = np.minimum(np.maximum(x[0], 0), 1).reshape(m, n)
    return alpha


def main(model, input_path, output_path):

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

    # Basic approach
    mask0 = compute_mask_from_prediction(pred, id)
    red0 = generate_red_illustration(img_rgb_small, mask0)
    sticker0 = generate_sticker_illustration(img_rgb_small, mask0)


    tmp = trimap_v1(mask0)
    # tmp = trimap_v2(pred[id, :, :])
    trimap = np.zeros((height, width, 3), dtype=np.uint8)
    trimap[:,:,0] = tmp
    trimap[:,:,1] = tmp
    trimap[:,:,2] = tmp


    import time
    tic = time.time()
    alpha = knn_matte(img_rgb_small, trimap)
    toc = time.time()
    print 'Processing time : %f s' % (toc-tic)

    red = img_rgb_small.copy()
    a = np.ones((height, width), np.uint8) * 255
    red[:,:,0] = alpha * a + (1-alpha) * red[:,:,0]

    white = np.ones((height, width), np.uint8) * 255
    sticker = img_rgb_small.copy()
    sticker[:,:,0] = alpha * sticker[:,:,0] + (1-alpha) * white
    sticker[:,:,1] = alpha * sticker[:,:,1] + (1-alpha) * white
    sticker[:,:,2] = alpha * sticker[:,:,2] + (1-alpha) * white

    # # scipy.misc.imsave('donkeyAlpha.png', alpha)
    # f, ax = plt.subplots(1,3)
    # plt.title('Alpha Matte')
    # ax[0].imshow(img_rgb_small)
    # ax[1].imshow(trimap)
    # # ax[2].imshow(alpha, cmap='gray')
    # ax[2].imshow(sticker)
    # plt.show()

    top = np.concatenate([red0, red], axis=1)
    bottom = np.concatenate([sticker0, sticker], axis=1)
    out = np.concatenate([top, bottom], axis = 0)

    # plt.figure()
    # plt.imshow(out)
    # plt.show()


    cv2.imwrite(output_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':

    # Paths
    model_path = '../../../models/MultiArrayDeepLab.mlmodel'
    input_folder = '/Users/vincent/data/Work/Projects/filter_segmentation/samples'
    output_folder = '/Users/vincent/background/results/2019.05.14_knn_morphomath'

    # Read the model
    model = coremltools.models.MLModel(model_path)

    image_path_list = sorted(glob(os.path.join(input_folder, '*.jpg')))
    for input_path in image_path_list:

        print input_path
        sys.stdout.flush()

        base = os.path.basename(input_path)
        name = os.path.splitext(base)[0]
        output_path = os.path.join(output_folder, name + '.png')

        if os.path.exists(output_path):
            continue 

        main(model, input_path, output_path)

        # break


# # Paths
# model_path = '../../../models/MultiArrayDeepLab.mlmodel'
# output_folder = '/Users/vincent/background/results/2019.05.13_grabcut.comparision'

# # Read the model
# model = coremltools.models.MLModel(model_path)

# # Image path
# image_path = '../../../samples/person_matthieu.jpg'
# # image_path = '../../../samples/moto_man.jpg'
# # input_path = '../../../samples/train_2.jpg'
# # image_path = '/Users/vincent/Downloads/VOCdevkit/VOC2012/JPEGImages/2007_000452.jpg'

# # base = os.path.basename(input_path)
# # name = os.path.splitext(base)[0]
# # output_path = os.path.join(output_folder, name + '.png') 
# # print output_path


