import numpy as np
from scipy import fft
from matplotlib import pyplot as plt
from scipy import stats

# Generate a signal with two sine waves and Gaussian noise
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y = np.sin(x) + np.sin(4 * x) + stats.norm.rvs(size=1000, loc=0, scale=0.1)

# Plot the original function
plt.plot(x, y)
plt.title("Original function")
plt.show()

# Perform Fourier transform on the signal
transformed = fft.fft(y)
freqs = fft.fftfreq(len(y))

# Plot the frequency spectrum of the transformed signal
plt.plot(np.sort(freqs)[600:1000], np.abs(transformed)[600:1000])
plt.show()

# Set a threshold to remove small magnitude frequency components
threshold = 0.5 * np.abs(transformed).max()
index = (np.abs(transformed) <= threshold)
transformed[index] = 0 + 0j

# Perform inverse Fourier transform to obtain denoised signal
denoised = fft.ifft(transformed)

# Plot the denoised signal
plt.plot(np.real(denoised))
plt.title("Denoised Signal")
plt.show()
