import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


# Definition of the kernels
def linear_kernel(x, offset, width):
    x = x - offset
    x = x / width
    return (1 - np.abs(x)) * (np.abs(x) < 1)

def sample_hold_kernel(x, offset, width):
    x = x - offset
    return (x >= 0) * (x < width)

def sinc_kernel(x, offset, width, alpha=np.inf):
    x = x - offset
    x = x / width
    return (x >= -alpha) * (x < alpha) * np.sinc(x)

# Convolution function
def conv1d(x_measure, y_measure, x_interpolate, kernel, **kwargs):
    width = x_measure[1] - x_measure[0]
    kernels = np.asarray([kernel(x_interpolate, offset=offset, width=width, **kwargs) for offset in x_measure])
    return y_measure @ kernels

# Display results function
def display_results(x_original, y_original, x_interp, y_interp, kernel_name, mse):
    mse_formatted = "{:.10f}".format(mse)
    label = f'Convolution Result with {kernel_name} Kernel (MSE={mse_formatted})'
    plt.scatter(x_original, y_original, label='Original sin(x)', s=3)
    plt.scatter(x_interp, y_interp, label=label, s=3)
    plt.title(f'Convolution Result with {kernel_name} Kernel, increased to {len(x_interp)} points')
    plt.legend()
    plt.grid(True)

# Constants
NUM_SUBPLOTS = 4

# Discrete sine function with N points in the interval [-pi, pi]
num_points = 100
x_original = np.linspace(-np.pi, np.pi, num_points)
y_original = np.sin(x_original)

# Interpolation of the sine function with N_interpolate points
N_interpolate_values = [200, 500, 1000]
x_interp_values = [np.linspace(-np.pi, np.pi, N_interpolate) for N_interpolate in N_interpolate_values]
y_true = [np.sin(x_interp) for x_interp in x_interp_values]

# Loop over different kernels
kernel_functions = [linear_kernel, sample_hold_kernel, sinc_kernel]
kernel_names = ['linear', 'sample hold', "sinc kernel"]

# Iterate over kernel functions and their names
for kernel, kernel_name in zip(kernel_functions, kernel_names):
    # Perform interpolation and calculate MSE values for each interpolation point
    result_convolution_values = [
        conv1d(x_original, y_original, x_interp, kernel) 
        for x_interp in x_interp_values
    ]

    mse_values = [metrics.mean_squared_error(y_true=y_true_i, y_pred=result_i) 
                  for y_true_i, result_i in zip(y_true, result_convolution_values)]

    # Create a plot for the original data
    plt.figure(figsize=(12, 16))
    plt.subplot(NUM_SUBPLOTS, 1, 1)
    plt.scatter(x_original, y_original, label='Original sin(x)', s=3)
    plt.title(f'Original sin(x) with {num_points} points')
    plt.legend()
    plt.grid(True)

    # Create subplots for different interpolation values
    for i, (N_interp, result, mse) in enumerate(zip(N_interpolate_values, result_convolution_values, mse_values), start=1):
        plt.subplot(NUM_SUBPLOTS, 1, i + 1)
        display_results(x_original, y_original, x_interp_values[i - 1], result, kernel_name, mse)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
