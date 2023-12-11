import numpy as np
import cv2
from sklearn import metrics

grey_image = cv2.imread("interpolation/image.jpg", cv2.IMREAD_GRAYSCALE)

# Define your custom kernel
custom_kernel = np.array([[1, 1],
                        [1, 1]]) * 0.25

# Pooling with avarege of pool region * kernel
def average_pooling(image, kernel, stride):
    kernel_height, kernel_width = kernel.shape
    result = np.zeros((image.shape[0] // stride, image.shape[1] // stride), dtype=np.uint8)
    for i in range(0, image.shape[0], stride):
        for j in range(0, image.shape[1], stride):
            pool_region = image[i:i+kernel_height, j:j+kernel_width]
            pool_mean = np.sum(pool_region * kernel)
            result[i // stride, j // stride] = pool_mean

    return result

# Pooling with max of pool region * kernel
def max_pooling(image, kernel, stride):
    kernel_height, kernel_width = kernel.shape
    result = np.zeros((image.shape[0] // stride, image.shape[1] // stride), dtype=np.uint8)
    for i in range(0, image.shape[0], stride):
        for j in range(0, image.shape[1], stride):
            pool_region = image[i:i+kernel_height, j:j+kernel_width]
            pool_max = np.max(pool_region * kernel * 4)
            result[i // stride, j // stride] = pool_max

    return result

# Kernels from 1d interpolation
def linear_kernel2d(x, offset, width):
    x = x - offset
    x = x / width
    return (1 - np.abs(x)) * (np.abs(x) < 1)

def sinc_kernel2d(x, offset, width, alpha = np.inf):
    x = x - offset
    x = x / width
    return (x >= -alpha) * (x < alpha) * np.sinc(x)

# Function from 1d interpolation
def conv1d_interpolate(x_measure, y_measure, x_interpolate, kernel: callable):
    width = x_measure[1] - x_measure[0]
    kernels = np.asarray([kernel(x_interpolate, offset=offset, width=width) for offset in x_measure])
    return y_measure @ kernels

# Function to interpolate image
def image_interpolate(image, kernel: callable, ratio):

    def row_column_interpolate(row):
        x_measure = np.arange(len(row))
        x_interpolate = np.linspace(0, len(row), ratio * len(row), endpoint=False)
        return conv1d_interpolate(x_measure, row, x_interpolate, kernel)

    interpolated = np.apply_along_axis(row_column_interpolate, 1, image)
    return np.apply_along_axis(row_column_interpolate, 0, interpolated)

# Number of pixels to step by, size of imput image must equal mod stride = 0
stride = 2
ratio = 2

average_pooled_image = average_pooling(grey_image, custom_kernel, stride)
max_pooled_image = max_pooling(grey_image, custom_kernel, stride)
interpolated_image = image_interpolate(average_pooled_image, linear_kernel2d, ratio)

# Save rusults
cv2.imwrite("interpolation/resultavg.jpg", average_pooled_image)
cv2.imwrite("interpolation/resultmax.jpg", max_pooled_image)
cv2.imwrite("interpolation/interpolated_image.jpg", interpolated_image)

# Print Metrics mean squared error
print(f"MSE: {metrics.mean_squared_error(grey_image, interpolated_image):.8f}")