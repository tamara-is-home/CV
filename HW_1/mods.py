import cv2
import logging
import numpy as np

from HW_1.affine_transform import affine_skew, plot_transformed
from HW_1.utils import det_mser, des_orb, det_orb, det_brisk, des_brisk, match_two, find_homography_ransac

logging.basicConfig(format='%(levelname)s: %(filename)s: %(message)s', level=logging.INFO)
logger = logging.getLogger()


class MODS:
    def __init__(self, matches_ratio: float, num_iter: int):

        self.inliers_ratio = matches_ratio
        self.num_iter = num_iter  # 7 is the upper gap
        # here we initiate detectors objects that we will choose depending on the iteration
        self.detectors = {'ORB': [cv2.ORB_create(), det_orb, des_orb],
                          'MSER': [cv2.MSER_create(), det_mser, des_orb],
                          'BRISK': [cv2.BRISK_create(), det_brisk, des_brisk]}
        self.current_detector = 'ORB'

    @staticmethod
    def generate_affine(img1, img2, plot=False):

        import random
        tilt1 = float(random.randint(1, 5)/random.randint(1, 5))
        tilt2 = float(random.randint(1, 5)/random.randint(1, 5))
        # here we generate some random values for rotation angle and scaling
        phi1 = random.randint(1, 360)
        phi2 = random.randint(1, 360)

        img1_af, _, _ = affine_skew(img=img1, tilt=tilt1, phi=phi1)
        img2_af, _, _ = affine_skew(img=img2, tilt=tilt2, phi=phi2)

        if plot:
            plot_transformed(img1, img1_af)
            plot_transformed(img2, img2_af)

        return img1_af, img2_af

    def choose_detector(self, iteration_num):
        if iteration_num < 2:
            logger.info('Choosing ORB detector and descriptor for this iteration')
            return self.detectors['ORB']
        elif 2 <= iteration_num < 4:
            logger.info('Choosing MSER detector and ORB descriptor for this iteration')
            self.current_detector = 'MSER'
            return self.detectors['MSER']
        else:
            logger.info('Choosing BRISK detector and descriptor for this iteration')
            self.current_detector = 'BRISK'
            return self.detectors['BRISK']

    def match_until_success(self, input_img1, input_img2, dist_ratio=0.75, plot_best=False):
        # this method returns homography matrix, which can be later verified
        H = np.array([])
        counter = 0
        matches_ratio = 0
        best_match = {'kp1': None, 'kp2': None, 'des1': None,
                      'des2': None, 'img1': None, 'img2': None}
        best_detector = self.current_detector

        while matches_ratio < self.inliers_ratio and counter < self.num_iter:
            logger.info(f'::: ITER NUM {counter+1} :::')
            if counter == 0:
                img1, img2 = input_img1, input_img2
            else:
                img1, img2 = self.generate_affine(input_img1, input_img2)

            det_des_obj, det_method, des_method = self.choose_detector(iteration_num=counter)

            kp1 = det_method(img1, det_des_obj, False)
            kp2 = det_method(img2, det_des_obj, False)

            if self.current_detector == 'MSER':
                # here we change obj used as descriptor, cause MSER doesnt have compute method
                det_des_obj = self.detectors['ORB'][0]

            kp1, des1 = des_method(kp1, img1, det_des_obj)
            kp2, des2 = des_method(kp2, img2, det_des_obj)

            good_matches = match_two(kp1, des1, kp2, des2, img1, img2,
                                     BF=False, dist_ratio=dist_ratio, plot=False)

            H_new, inliers_ratio_new = find_homography_ransac(kp1, kp2, good_matches)

            if inliers_ratio_new > matches_ratio:
                H = H_new
                matches_ratio = inliers_ratio_new
                best_match['kp1'], best_match['kp2'], best_match['des1'], best_match['des2'] = kp1, kp2, des1, des2
                best_match['img1'], best_match['img2'] = img1, img2
                best_detector = self.current_detector
            counter += 1

        logger.info('Stopping to iterate')
        if counter == self.num_iter:
            logger.info(f'Maximum matches ratio that was achieved during all iterations is {round(matches_ratio,3)}')
        else:
            logger.info(f'Wanted matches ratio was achieved on {counter} iter with {best_detector}! Current ratio: {round(matches_ratio,3)}')

        if plot_best:
            logger.info('Plotting best found matches:')
            match_two(**best_match,
                      BF=False, dist_ratio=0.75, plot=True)

        return H
