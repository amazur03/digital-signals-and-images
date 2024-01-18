import numpy as np
from skimage import io

# Function to apply a function to each channel of an RGB image
def apply_rgb(func: callable, image, *args, **kwargs):
    return np.dstack([func(image[:, :, channel], *args, **kwargs) for channel in range(3)])

# Class for 2D Fourier Transform
class FourierTransform2D():
    def forward(self, variables):
        # Perform 2D Fourier transform
        return np.fft.fft2(variables)

    def backward(self, variables):
        # Perform inverse 2D Fourier transform
        return np.abs(np.fft.ifft2(variables))

# Function to compress and decompress image using Fourier Transform
def compress_and_decompress(image, transform: FourierTransform2D, compression: float):

    transformed = transform.forward(image)
    coefficients = np.sort(np.abs(transformed.reshape(-1)))
    threshold = coefficients[int(compression * len(coefficients))]
    indices = np.abs(transformed) > threshold
    decompressed = transformed * indices
    return transform.backward(decompressed)

# Read input image
image = io.imread(r"demosaicking/CFA/Bayer/panda.jpg")

# Apply compression and decompression to each channel of RGB image
decompressed_image = apply_rgb(compress_and_decompress, image, transform=FourierTransform2D(), compression=0.1)

# Display decompressed image after clipping pixel values to [0, 255]
_ = io.imshow(np.clip(decompressed_image.astype(int), 0, 255))
io.show()
