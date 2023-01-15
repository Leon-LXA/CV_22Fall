import numpy as np
import cv2
import math

# First we create a color space containing hist_bin^3 bins. And we also need to
# adjust position inputs to make sure they lie in the frame. For every pixel in
# this small region, we judge which bin the RGB pixel should be in and sum them
# up for the final histogram and then normalize it.
def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin) -> np.array:
    hist = np.zeros(hist_bin ** 3)
    divide = np.linspace(0, 255, hist_bin + 1)
    xmin = np.maximum(xmin, 0)
    ymin = np.maximum(ymin, 0)
    xmax = np.minimum(xmax, frame.shape[1])
    ymax = np.minimum(ymax, frame.shape[0])

    # hist[i][j][k] = np.sum(frame[:,:] == [i,j,k])
    for x in range(math.ceil(xmin), math.floor(xmax)):
        for y in range(math.ceil(ymin), math.floor(ymax)):
            R = frame[y, x, 0]
            G = frame[y, x, 1]
            B = frame[y, x, 2]
            for i in range(hist_bin):
                if divide[i] <= R < divide[i + 1]:
                    R_index = i
                if divide[i] <= G < divide[i + 1]:
                    G_index = i
                if divide[i] <= B < divide[i + 1]:
                    B_index = i
            RGB_index = (hist_bin ** 2) * R_index + hist_bin * G_index + B_index
            hist[RGB_index] = hist[RGB_index] + 1

    return hist/sum(hist)
