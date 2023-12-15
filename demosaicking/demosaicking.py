import numpy as np
from skimage import io
from matplotlib import pyplot as plt

image = io.imread(r"demosaicking/CFA/Bayer/panda.jpg")

# Bayer filter
demosaicking_convolution_mask = np.dstack([
    np.ones([2, 2]),        # R
    0.5 * np.ones([2, 2]),  # G
    np.ones([2, 2]),        # B
]) 

# Convolve with Bayer filter
def demosaic(image, kernel, stride):
    kernel_height, kernel_width, _ = kernel.shape
    result = np.zeros((image.shape[0] // stride, image.shape[1] // stride, image.shape[2]), dtype=np.uint8)

    for i in range(0, image.shape[0], stride):
        for j in range(0, image.shape[1], stride):
            for k in range(image.shape[2]):
                pool_region = image[i:i+kernel_height, j:j+kernel_width, k]
                kernel_region = kernel[:, :, k]
                pool_mean = np.sum(pool_region * kernel_region)
                result[i // stride, j // stride, k] = pool_mean

    return result

# Show results
result = demosaic(image, demosaicking_convolution_mask, 1)
plt.imshow(result)
plt.show()