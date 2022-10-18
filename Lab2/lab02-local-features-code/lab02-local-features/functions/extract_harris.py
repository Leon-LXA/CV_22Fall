import numpy as np
import scipy.ndimage
from scipy import signal
import cv2
from datetime import datetime


# Harris corner detector
def extract_harris(img, sigma=1.0, k=0.05, thresh=1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # Compute image gradients
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    operator_y = 0.5 * np.array([[-1],
                                 [0],
                                 [1]])
    operator_x = 0.5 * np.array([[-1, 0, 1]])
    # Ix and Iy matrix
    Ix_mat = signal.convolve2d(img, operator_x, mode='same')
    Iy_mat = signal.convolve2d(img, operator_y, mode='same')

    # Compute local auto-correlation matrix
    # TODO: compute the auto-correlation matrix here
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    Mp_unsmooth11 = Ix_mat * Ix_mat
    Mp_unsmooth12 = Ix_mat * Iy_mat
    Mp_unsmooth21 = Mp_unsmooth12
    Mp_unsmooth22 = Iy_mat * Iy_mat

    Mp_11 = cv2.GaussianBlur(Mp_unsmooth11, (3, 3), sigma, cv2.BORDER_REPLICATE)
    Mp_12 = cv2.GaussianBlur(Mp_unsmooth12, (3, 3), sigma, cv2.BORDER_REPLICATE)
    Mp_21 = Mp_12
    Mp_22 = cv2.GaussianBlur(Mp_unsmooth22, (3, 3), sigma, cv2.BORDER_REPLICATE)

    # Compute Harris response function
    # TODO: compute the Harris response function C here
    C = (Mp_11 * Mp_22 - Mp_12 * Mp_21) - k * (Mp_11 + Mp_22) ** 2

    # Detection with threshold
    # TODO: detection and find the corners here
    # For the local maximum check, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    H = scipy.ndimage.maximum_filter(C, size=3)
    # ret, C = cv2.threshold(C, thresh, 255, cv2.THRESH_TOZERO)
    # time_1 = datetime.now()

    corners_idx = np.argwhere((C == H) & (C > thresh))
    corners = np.zeros((len(corners_idx), 2))
    corners[:, 0] = corners_idx[:, 1]
    corners[:, 1] = corners_idx[:, 0]
    corners = np.delete(corners, np.where((corners[:, 0] == 0) | (corners[:, 0] == img.shape[1]-1)
                                          | (corners[:, 1] == 0) | (corners[:, 1] == img.shape[0]-1))[0], axis=0)

    # corners = np.empty((0, 2), int)
    # for i in range(1, len(C)-1):
    #     for j in range(1, len(C[0])-1):
    #         if C[i][j] > thresh and C[i][j] == H[i][j]:
    #             corners = np.row_stack((corners, np.array([j, i])))

    # time_lap = (datetime.now() - time_1)
    # print(time_lap)
    return corners.astype(int), C
