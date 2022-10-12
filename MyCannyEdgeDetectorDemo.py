# Name - Jatin
# Entry Number - 2020CSB1090

import numpy as np
from math import log10, sqrt
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros
import skimage
from skimage.color import *
from skimage import feature
from matplotlib.image import imread
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity


def peak_signal_to_noise_ratio(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def myCannyEdgeDetector(inputImg, lowThresholdRatio, highThresholdRatio):

    # gaussian filter Noise Reduction
    gx, gy = np.meshgrid(np.arange(-7/2+1, 7/2+1), np.arange(-7/2+1, 7/2+1))

    NORMALISED = 1 / (2.0 * np.pi * 1**2)

    Kernal = np.exp(-(gx**2+gy**2) / (2.0*1**2)) / \
        NORMALISED

    size_kernal, gaussi_filter = Kernal.shape[0], np.zeros_like(
        img, dtype=float)

    for i in range(img.shape[0]-(size_kernal-1)):
        for j in range(img.shape[1]-(size_kernal-1)):
            window = img[i:i+size_kernal, j:j+size_kernal] * Kernal
            gaussi_filter[i, j] = np.sum(window)

    # plt.imshow(gaussi_filter, cmap='gray')
    # plt.show()

    # sobel filter
    sobel_x = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    sobel_y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    [rows, columns] = np.shape(gaussi_filter)
    sobel_filtered_image = np.zeros(shape=(rows, columns))

    theta = np.zeros(shape=(rows, columns))

    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(sobel_x, gaussi_filter[i:i + 3, j:j + 3]))
            gy = np.sum(np.multiply(sobel_y, gaussi_filter[i:i + 3, j:j + 3]))
            sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)
            if (gx == 0):
                theta[i+1, j+1] = 90
            else:
                theta[i+1, j+1] = ((np.arctan(gy/gx))/np.pi) * 180

    sobel_filtered_image = sobel_filtered_image/np.max(sobel_filtered_image)
    # plt.imshow(sobel_filtered_image, cmap='gray')
    # plt.show()

    # Nonmaximum suppression
    nms = np.copy(sobel_filtered_image)

    for i in range(theta.shape[0]-(2)):
        for j in range(theta.shape[1]-(2)):
            if theta[i][j] < 0:
                theta[i][j] += 180

            if (theta[i, j] <= 22.5 or theta[i, j] > 157.5):
                if (sobel_filtered_image[i, j] <= sobel_filtered_image[i-1, j]) and (sobel_filtered_image[i, j] <= sobel_filtered_image[i+1, j]):
                    nms[i, j] = 0
            if (theta[i, j] > 22.5 and theta[i, j] <= 67.5):
                if (sobel_filtered_image[i, j] <= sobel_filtered_image[i-1, j-1]) and (sobel_filtered_image[i, j] <= sobel_filtered_image[i+1, j+1]):
                    nms[i, j] = 0
            if (theta[i, j] > 67.5 and theta[i, j] <= 112.5):
                if (sobel_filtered_image[i, j] <= sobel_filtered_image[i+1, j+1]) and (sobel_filtered_image[i, j] <= sobel_filtered_image[i-1, j-1]):
                    nms[i, j] = 0
            if (theta[i, j] > 112.5 and theta[i, j] <= 157.5):
                if (sobel_filtered_image[i, j] <= sobel_filtered_image[i+1, j-1]) and (sobel_filtered_image[i, j] <= sobel_filtered_image[i-1, j+1]):
                    nms[i, j] = 0

    nms = nms/np.max(nms)
    # plt.imshow(nms, cmap='gray')
    # plt.show()

    # Double THRESHING
    Threash = np.copy(nms)
    h = int(Threash.shape[0])
    w = int(Threash.shape[1])
    highThreshold = np.max(Threash) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    for i in range(1, h-1):
        for j in range(1, w-1):
            if (Threash[i, j] > highThreshold):
                Threash[i, j] = 1
            elif (Threash[i, j] < lowThreshold):
                Threash[i, j] = 0
            else:
                if ((Threash[i-1, j-1] > highThreshold) or
                    (Threash[i-1, j] > highThreshold) or
                    (Threash[i-1, j+1] > highThreshold) or
                    (Threash[i, j-1] > highThreshold) or
                    (Threash[i, j+1] > highThreshold) or
                    (Threash[i+1, j-1] > highThreshold) or
                    (Threash[i+1, j] > highThreshold) or
                        (Threash[i+1, j+1] > highThreshold)):
                    Threash[i, j] = 1

    return Threash


img = skimage.io.imread('7.jpg')
img = rgb2gray(img)
outputImg = myCannyEdgeDetector(img, 0.2, 0.4)

plt.figure()
plt.imshow(outputImg, cmap='gray')
plt.title("My canny Edge Detector")
plt.show()
true_canny = feature.canny(img)
plt.imshow(true_canny, cmap='gray')
plt.title("Inbuilt_Function")
plt.show()
psnr = peak_signal_to_noise_ratio(true_canny, outputImg)
print("Peak Signal To Noise Ratio: ", end=' ')
print(psnr)
ssim1 = structural_similarity(outputImg, true_canny, data_range=255)
print("Structural Similarity Index Metric: ", end=' ')
print(ssim1)
