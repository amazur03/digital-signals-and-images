import numpy as np
from skimage import io
from scipy.ndimage import convolve
from matplotlib import pyplot as plt

image = io.imread(r"CFA/Bayer/circle.jpg")

# Define filters
laplace_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
gauss_filter = np.array([[1, 2, 1], [1, 4, 1], [1, 2, 1]]) / 16
scharpening_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# Convolve funtions to detect edges, blur, scharpening
result_laplace = np.dstack([
    convolve(image[:, :, channel], laplace_filter, mode="constant", cval=0.0)
    for channel in range(3)
])

result_gauss = np.dstack([
    convolve(image[:, :, channel], gauss_filter, mode="constant", cval=0.0)
    for channel in range(3)
])

result_scharpening = np.dstack([
    convolve(image[:, :, channel], scharpening_filter, mode="constant", cval=0.0)
    for channel in range(3)
])

# Schow results
plt.imshow(result_laplace)
plt.show()
plt.imshow(result_gauss)
plt.show()
plt.imshow(result_scharpening)
plt.show()