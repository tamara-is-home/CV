import cv2
import numpy as np
from matplotlib import pyplot as plt

# thank you https://github.com/Mars-Rover-Localization/PyASIFT/blob/main/asift.py


def affine_skew(tilt, phi, img, mask=None):
    h, w = img.shape[:2]

    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255

    A = np.float32([[1, 0, 0], [0, 1, 0]])

    # Rotate image
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c, -s], [s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32(np.dot(corners, A.T))
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1, -1, 2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Tilt image (resizing after rotation)
    if tilt != 1.0:
        s = 0.8 * np.sqrt(tilt * tilt)
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt

    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)

    Ai = cv2.invertAffineTransform(A)

    return img, mask, Ai


def plot_transformed(img_o, img_a):
    plt.subplot(1, 2, 1)
    plt.imshow(img_o)
    plt.title('Original image')
    plt.subplot(1, 2, 2)
    plt.imshow(img_a)
    plt.title('Original image after affine transformation')
    plt.show()


if __name__ == '__main__':
    img1 = cv2.imread('images/other/chess1.jpg', cv2.COLOR_BGR2GRAY)
    img2, _, _ = affine_skew(img=img1, phi=60.2, tilt=1.1)
    plot_transformed(img1, img2)
