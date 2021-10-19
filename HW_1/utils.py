import logging
import numpy as np
import cv2
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (12,8)

logging.basicConfig(format='%(levelname)s: %(filename)s: %(message)s', level=logging.INFO)
logger = logging.getLogger()

def read_2_gray_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[0] > 800 or img.shape[1] > 800:
        img = cv2.resize(img, (800, 800))
    return img


def det_orb(img, orb, plot=False):

    kp = orb.detect(img, None)
    if plot:
        img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        plt.imshow(img2), plt.title('KP detected with ORB'), plt.show()
    return kp


def det_mser(img, mser, plot=False):

    kp = mser.detect(img)
    if plot:
        img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        plt.imshow(img2), plt.title('KP detected with MSER'), plt.show()
    return kp


def des_orb(kp, img, orb):
    kp, des = orb.compute(img, kp)
    return kp, des


def det_brisk(img, brisk, plot=False):
    kp = brisk.detect(img, None)
    if plot:
        img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        plt.imshow(img2), plt.title('KP detected with BRISK'), plt.show()
    return kp


def des_brisk( kp, img, brisk):
    kp, des = brisk.compute(img, kp)
    return kp, des


def match_two(kp1, des1, kp2, des2 , img1, img2,
              BF=True, dist_ratio=0.7, plot=True):

    if not BF:
        logger.info('Using FLANN matcher')
        FLANN_INDEX_LSH = 6  # just some magic 6 dunno
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)

        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)
    else:
        logger.info('Using Brute-Force matcher')
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

    logger.info(f'Found {len(matches)} matches at total')

    # Need to draw only good matches, so create a mask
    good_matches = [[0, 0] for i in range(len(matches))]

    # ratio test
    g_m = []
    
    for i, m in enumerate(matches):
        if len(m) > 1 and m[0].distance < dist_ratio * m[1].distance:
            g_m.append(i)
            good_matches[i] = [1, 0]

    if plot:
        draw_params = dict(matchColor=(255, 0, 0),
                           singlePointColor=(0, 255, 0),
                           matchesMask=good_matches,
                           flags=0)

        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

        plt.imshow(img3), plt.title('Match results'), plt.show()

    logger.info(f'Filtered {len(g_m)} good matches!')
    return np.asarray([matches[i] for i in g_m])


def find_homography_ransac(kp1, kp2, good_matches):
    if len(good_matches) > 4:

        pts1 = np.float64([kp1[m.queryIdx].pt for m in good_matches[:, 0]])
        pts2 = np.float64([kp2[m.trainIdx].pt for m in good_matches[:, 0]])

        (H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
        ratio = float(status.sum()) / status.size
        logger.info(f'Ratio of the number of good matched keypoints (inliers) to the total number of keypoints is {round(ratio,3)}')
        return H, ratio

    # No matches were found
    else:
        logger.info('less than 4 matched points were found!')
        return -1, 0


def warp_with_homography(H, source_img, dest_img, extra: str = None):
    if type(H) == int:
        logger.info('Homography matrix wasn\'t found!')
        return 0
    else:
        try:
            h, w = dest_img.shape[:2]
            im_out = cv2.warpPerspective(source_img, H, (h, w))
            titles = ['Source img', 'Destination img', f'Wrapped source img {extra}']
            for i, image in enumerate([source_img, dest_img, im_out]):
                plt.subplot(1, 3, i+1)
                plt.imshow(image, 'gray')
                plt.title(titles[i])
            plt.show()

        except Exception as e:
            logger.warning(e)
            logger.info(f'could not warp image with homography matrix, matching is too bad')


def find_homography_wxbs(pts1, pts2):
    (H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return H


def h_euc_distance(h_true, h_found):
    try:
        dist = np.linalg.norm(h_true-h_found)
        logger.info(f'euclidean distance between predicted homography matrix and true homography matrix if {round(dist,3)}')
        return dist
    except Exception as e:
        logger.info('Could not calculate euc dist')
        logger.warning(e)

if __name__ == '__main__':

    img1 = read_2_gray_img('images/other/tower1.jpg')
    img2 = read_2_gray_img('images/other/tower3.jpg')

    orb = cv2.ORB_create()
    mser = cv2.MSER_create()
    brisk = cv2.BRISK_create()

    logger.info('MSER RESULTS:')

    e1 = cv2.getTickCount()  # measure time

    kp_mser_1 = det_mser(img1, mser, True)
    kp_mser_2 = det_mser(img2, mser, True)

    kp1, des1 = des_orb(kp_mser_1, img1, orb)
    kp2, des2 = des_orb(kp_mser_2, img2, orb)
    good_matches = match_two(kp1, des1, kp2, des2, img1, img2,
                             BF=False, dist_ratio=0.7, plot=True)

    H, _ = find_homography_ransac(kp1, kp2, good_matches)

    warp_with_homography(H, img1, img2, 'MSER')

    e2 = cv2.getTickCount()
    t = (e2 - e1) / cv2.getTickFrequency()
    logger.info(f'It took {t} seconds')

    logger.info('ORB RESULTS:')

    e1 = cv2.getTickCount()  # measure time

    kp_orb_1 = det_orb(img1, orb, True)
    kp_orb_2 = det_orb(img2, orb, True)

    kp1, des1 = des_orb(kp_orb_1, img1, orb)
    kp2, des2 = des_orb(kp_orb_2, img2, orb)

    good_matches = match_two(kp1, des1, kp2, des2, img1, img2,
                             BF=False, dist_ratio=0.7, plot=True)

    H, _ = find_homography_ransac(kp1, kp2, good_matches)

    warp_with_homography(H, img1, img2, 'ORB')

    e2 = cv2.getTickCount()
    t = (e2 - e1) / cv2.getTickFrequency()
    logger.info(f'It took {t} seconds')

    logger.info('BRISK RESULTS:')

    e1 = cv2.getTickCount()  # measure time

    kp_brisk_1 = det_brisk(img1, orb, True)
    kp_brisk_2 = det_brisk(img2, orb, True)

    kp1, des1 = des_brisk(kp_brisk_1, img1, brisk)
    kp2, des2 = des_brisk(kp_brisk_2, img2, brisk)

    good_matches = match_two(kp1, des1, kp2, des2, img1, img2,
                             BF=False, dist_ratio=0.7, plot=True)

    H, _ = find_homography_ransac(kp1, kp2, good_matches)

    warp_with_homography(H, img1, img2, 'BRISK')

    e2 = cv2.getTickCount()
    t = (e2 - e1) / cv2.getTickFrequency()
    logger.info(f'It took {t} seconds')
