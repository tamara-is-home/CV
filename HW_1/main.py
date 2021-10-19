import os
import logging
import numpy as np
from matplotlib import pyplot as plt

from HW_1.mods import MODS
from HW_1.utils import warp_with_homography, read_2_gray_img, h_euc_distance, find_homography_wxbs

logging.basicConfig(format='%(levelname)s: %(filename)s: %(message)s', level=logging.INFO)
logger = logging.getLogger()


def evaluate_evd():

    logger.info('<<< EVALUATING ON EVD DATASET >>>')

    hs_true = []  # this array will keep true values of homography matrices for EVD
    im_names = []  # surprise, img names
    h_h_distances = []  # this will keep euc dist between true and predicted homography matrices

    for file in os.listdir('images/h'):
        im_names.append(file.split('.')[0])
        with open(f'images/h/{file}', 'r') as f:
            hs_true.append(np.loadtxt(f))

    for im_name, h_true in zip(im_names, hs_true):

        logger.info(f'Reading image {im_name}.png')
        img1 = read_2_gray_img(f'images/1/{im_name}.png')
        img2 = read_2_gray_img(f'images/2/{im_name}.png')

        mods = MODS(num_iter=7, matches_ratio=0.9)
        H = mods.match_until_success(img1, img2, plot_best=False)

        warp_with_homography(H, img1, img2, 'predicted')
        warp_with_homography(h_true, img1, img2, 'ground truth')

        h_h_distances.append(h_euc_distance(h_true, H))

    plt.plot(im_names, h_h_distances, color='red')
    plt.title('Euclidean distances between true and predicted H matrices for all images')
    plt.show()


def evaluate_wxbs():
    logger.info('<<< EVALUATING ON WxBS DATASET >>>')

    pts_true = {}  # this dict will keep ground truth values of matched points (pts1, pts2)
    images = {}  # this dict will keep image name and image path
    h_h_distances = []  # this will keep euc dist between true and predicted homography matrices
    hs_true = []

    for dir in os.listdir('images/WxBS'):
        images[f'{dir}'] = list()
        pts_true[f'{dir}'] = list()
        for file in os.listdir(f'images/WxBS/{dir}'):
            if str(file).endswith('png'):
                images[dir].append(f'images/WxBS/{dir}/{file}')
            elif str(file).endswith('txt') and len(str(file)) == 8:
                with open(f'images/WxBS/{dir}/{file}', 'r') as f:
                    pts_true[dir].append(np.loadtxt(f))

    # calculate H matrix for WxBS dataset by given good matches pts1 and pts2
    for k, v in pts_true.items():
        hs_true.append(find_homography_wxbs(v[0], v[1]))

    for h_true, (k, v) in zip(hs_true, images.items()):
        img1 = read_2_gray_img(v[0])
        img2 = read_2_gray_img(v[1])

        mods = MODS(num_iter=7, matches_ratio=0.9)
        H = mods.match_until_success(img1, img2, plot_best=False)

        warp_with_homography(H, img1, img2, 'predicted')
        warp_with_homography(h_true, img1, img2, 'ground truth')
        h_h_distances.append(h_euc_distance(h_true, H))

    plt.plot(list(images.keys()), h_h_distances, color='red')
    plt.title('Euclidean distances between true and predicted H matrices for all images')
    plt.show()


if __name__ == '__main__':
    evaluate_evd()
    evaluate_wxbs()
