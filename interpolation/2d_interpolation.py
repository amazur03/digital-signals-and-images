import numpy as np
import cv2

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


# Number of pixels to step by, size of imput image must equal mod stride = 0
stride = 2

average_pooled_image = average_pooling(grey_image, custom_kernel, stride)
max_pooled_image = max_pooling(grey_image, custom_kernel, stride)

# Save rusults
cv2.imwrite("interpolation/resultavg.jpg", average_pooled_image)
cv2.imwrite("interpolation/resultmax.jpg", max_pooled_image)