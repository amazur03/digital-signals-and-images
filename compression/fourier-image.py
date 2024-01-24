import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from skimage import io

# Load an image (replace 'your_image_path' with the actual path to your image)
image = io.imread(r"demosaicking/CFA/Bayer/panda.jpg")

# Separate RGB channels
red_channel, green_channel, blue_channel = [image[:, :, i] for i in range(3)]

# Apply Fourier transform and denoise each channel separately using list comprehensions
def denoise_channel(channel):
    channel_fft = fftpack.fft2(channel)

    cutoff_freq = 0.05
    rows, cols = channel.shape
    cutoff_row = int(rows * cutoff_freq)
    cutoff_col = int(cols * cutoff_freq)

    channel_fft[:cutoff_row, :] = 0
    channel_fft[:, :cutoff_col] = 0
    channel_fft[-cutoff_row:, :] = 0
    channel_fft[:, -cutoff_col:] = 0

    # Perform inverse Fourier transform to obtain denoised channel
    return np.abs(fftpack.ifft2(channel_fft))

# Apply denoising to each channel using list comprehensions
denoised_red, denoised_green, denoised_blue = [denoise_channel(channel) for channel in [red_channel, green_channel, blue_channel]]

# Combine denoised channels into an RGB image
denoised_image = np.stack([denoised_red, denoised_green, denoised_blue], axis=-1)

# Display the original and denoised images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(np.uint8(denoised_image))
plt.title('Denoised Image')

plt.show()