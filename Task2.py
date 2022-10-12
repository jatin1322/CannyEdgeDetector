# Name - Jatin
# Entry Number - 2020CSB1090

import numpy as np
from matplotlib.image import imread
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros
import matplotlib.pyplot as plt
from itertools import product
import skimage
from skimage.color import *
import sys


# kernel
conv_kernel = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])

# laplacian function


def laplace(image):
    gray = rgb2gray(image)

    y, x = gray.shape
    y = y - 2
    x = x - 2
    new_image = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            new_image[i][j] = np.sum(gray[i:i+3, j:j+3]*conv_kernel)

    return new_image


# sample images for dataset

images = ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg',
          '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg', '10.jpg']
images_var = []
for i in range(11):
    img = skimage.io.imread(images[i])
    imgg = laplace(img)
    images_var.append(imgg.var())

if (len(sys.argv) == 1):
    filename = "5.jpg"
else:
    filename = sys.argv[1]
inputimg = skimage.io.imread(filename)
inputvar = laplace(inputimg).var()
minvalue = min(images_var)
maxvalue = max(images_var)
if (inputvar >= maxvalue):
    print("Image is not blurry")
    print("And, Its Probability is 0")
elif (inputvar <= minvalue):
    print("Image is blurry")
    print("And, Its Probability is 1")
else:
    avrg = sum(images_var)/len(images_var)
    maxi = maxvalue-minvalue
    weight = maxvalue-inputvar
    if (inputvar < avrg):
        print("Image is blurry")
        print("And, its probability is ")
        print(weight/maxi)
    else:
        print("Image is not blurry")
        print("And, its probability is ")
        print(weight/maxi)
print("Variance after applying laplace filter on image is=")
print(inputvar)
print(" ")
print("If you want to change image you can pass that in as a argument i have attached diffrent images named like 0.jpg 1.jpg ...... 10.jpg          like python3 Task2.py 0.jpg")
plt.imshow(inputimg)
plt.title("Input Image is")
plt.show()
