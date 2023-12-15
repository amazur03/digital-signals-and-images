import numpy as np
from skimage import io
from scipy.ndimage import convolve
from matplotlib import pyplot as plt

image = io.imread(r"demosaicking/CFA/Bayer/circle.jpg")

# Define filters
laplace_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
gaussian_kernel_2x2 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
scharpening_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
gaussian_kernel_5x5 = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]]) / 273

# Convolve funtions to detect edges, blur, scharpening
result_laplace = np.dstack([
    convolve(image[:, :, channel], laplace_filter, mode="constant", cval=0.0)
    for channel in range(3)
])

result_gauss = np.dstack([
    convolve(image[:, :, channel], gaussian_kernel_2x2, mode="constant", cval=0.0)
    for channel in range(3)
])

result_scharpening = np.dstack([
    convolve(image[:, :, channel], scharpening_filter, mode="constant", cval=0.0)
    for channel in range(3)
])

result_gauss_5x5 = np.dstack([
    convolve(image[:, :, channel], gaussian_kernel_5x5, mode="constant", cval=0.0)
    for channel in range(3)
])


# Schow results
plt.imshow(result_gauss)
plt.title("gaussian kernel 2x2")
plt.show()
plt.imshow(result_gauss_5x5)
plt.title("gaussian kernel 5x5")
plt.show()
plt.imshow(result_scharpening)
plt.title("scharpening kernel")
plt.show()
plt.imshow(result_laplace)
plt.title("laplace kernel")
plt.show()