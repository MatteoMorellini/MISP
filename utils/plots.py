import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List

def plotting_imgs(vanilla_image: np.ndarray, blurred_image: np.ndarray, fourier_image:np.ndarray, filter: str) -> None:
    """
    Plots three images in a single row:
    1. The original image.
    2. The image after applying the specified filter (blurred image).
    3. The Fourier spectrum of the blurred image.

    Args:
    vanilla_image (ndarray): The original image to be displayed.
    blurred_image (ndarray): The blurred image obtained after applying the filter.
    fourier_image (ndarray): The Fourier spectrum (magnitude) of the blurred image.
    filter (str): The name of the filter used, displayed in the subplot titles.

    Returns:
    None: This function only displays the images, does not return any value.
    """
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
    
    # Plot the original image
    ax1.set_title("Image")
    ax1.imshow(vanilla_image, cmap='gray')
    ax1.axis('off')
    
    # Plot the blurred image with filter name in title
    ax2.set_title(f"{filter} blurred image")
    ax2.imshow(blurred_image, cmap='gray')
    ax2.axis('off')
    
    # Plot the Fourier spectrum of the blurred image
    ax3.set_title(f"Fourier spectrum of\n{filter} blurred image")
    ax3.imshow(fourier_image + 1, cmap="inferno")  # Added +1 for better visibility
    ax3.axis('off')
    
    # Adjust the layout and show the plots
    plt.tight_layout()
    plt.show()

def plotting_imgs_noise(vanilla_image: np.ndarray, blurred_image: np.ndarray, noisy_image: np.ndarray, filter: str, sigma: str) -> None:
    """
    Plots three images in a single row:
    1. The original image.
    2. The image after applying the specified filter (blurred image).
    3. The blurred image with noise added, showing the specified variance in the title.

    Args:
    vanilla_image (ndarray): The original image to be displayed.
    blurred_image (ndarray): The blurred image obtained after applying the filter.
    noisy_image (ndarray): The image after noise has been added to the blurred image.
    filter (str): The name of the filter used, displayed in the subplot titles.
    sigma (str): The variance of the noise added to the blurred image, displayed in the subplot title.

    Returns:
    None: This function only displays the images, does not return any value.
    """
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
    
    # Plot the original image
    ax1.set_title("Image")
    ax1.imshow(vanilla_image, cmap='gray')
    ax1.axis('off')
    
    # Plot the blurred image with filter name in title
    ax2.set_title(f"{filter} blurred image")
    ax2.imshow(blurred_image, cmap='gray')
    ax2.axis('off')
    
    # Plot the blurred image with noise and variance in title
    ax3.set_title(f"Blurred image with noise\n (variance: {sigma})")
    ax3.imshow(noisy_image + 1, cmap="gray")  # Added +1 for better visibility
    ax3.axis('off')
    
    # Adjust the layout and show the plots
    plt.tight_layout()
    plt.show()

def compute_deconvolution_multiple_mu(regularization: str, vanilla_image: np.ndarray, blurred_image: np.ndarray, \
                                      blur_kernel: np.ndarray, window: np.ndarray = None, iter: int = 20, log_range: \
                                      Tuple[int, int] = (-6, 1)):
    """
    Computes the deconvolution of a blurred image for multiple values of mu (regularization parameter).

    This function applies different regularization methods (H1 or L2) to the blurred image and calculates the 
    deconvolution for a range of mu values. 
    The result is a list of deconvolved images and their corresponding Peak Signal-to-Noise Ratio (PSNR) values.

    Args:
    regularization (str): The type of regularization to apply. Options are 'H1' or 'L2'.
    vanilla_image (ndarray): The original image (ground truth) to calculate PSNR.
    blurred_image (ndarray): The blurred image that is to be deconvolved.
    blur_kernel (ndarray): The blur kernel used for deconvolution.
    window (ndarray, optional): A window for normalization (to scale the deconvolved image). Default is None.
    iter (int, optional): The number of iterations (i.e., the number of different mu values). Default is 20.
    log_range (tuple, optional): Exponent range for mu values, defined as a tuple (start, stop) for logspace. Default is (-6, 1).

    Returns:
    tuple: A tuple containing:
        - deconvolved_images (list of ndarray): List of deconvolved images for each mu value.
        - psnr_values (list of float): List of PSNR values for each deconvolved image.
        - mu_values (ndarray): The array of mu values used for deconvolution.
    """
    deconvolved_images = []
    psnr_values = []
    start, stop = log_range
    mu_values = np.logspace(start, stop, num=iter)  # Generate mu values
    
    for mu in mu_values:
        # Apply the specified regularization method
        if regularization == "H1":
            deconvolved = h1_deconvolution(blurred_image, blur_kernel, mu)
        elif regularization == "L2":
            deconvolved = l2_deconvolution(blurred_image, blur_kernel, mu)
        else:
            raise ValueError("Unsupported regularization method. Use 'H1' or 'L2'.")
        
        # Normalize the deconvolved image if a window is provided
        if window is not None:
            epsilon = 1e-10  # Small value to avoid division by zero
            deconvolved = np.clip(deconvolved / (window + epsilon), 0, 1)
        
        # Append the deconvolved image and PSNR value
        deconvolved_images.append(deconvolved)
        psnr_values.append(psnr(deconvolved, vanilla_image))

    return deconvolved_images, psnr_values, mu_values

def plotting_deconvoluted_images_multiple_mu(deconvolved_images: List[np.ndarray], psnr_values: List[float], \
                                             mu_values: np.ndarray) -> None:
    """
    Plots deconvolved images for multiple values of mu and displays their corresponding PSNR values.

    This function creates a grid of subplots where each subplot displays a deconvolved image for a specific mu value, 
    along with the corresponding PSNR value. The images are arranged in a grid with a specified number of columns.

    Args:
    deconvolved_images (list of ndarray): List of deconvolved images, one for each mu value.
    psnr_values (list of float): List of PSNR values corresponding to each deconvolved image.
    mu_values (ndarray): Array of mu values used for deconvolution.

    Returns:
    None: This function only displays the images, does not return any value.
    """
    # Set the number of columns for the plot grid
    cols = 5
    # Calculate the number of rows needed to display all images
    rows = (len(mu_values) + cols - 1) // cols
    
    # Create the figure with appropriate size
    plt.figure(figsize=(10, 2 * rows))
    
    # Iterate over the deconvolved images, PSNR values, and mu values
    for i, (mu, image, psnr_value) in enumerate(zip(mu_values, deconvolved_images, psnr_values)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"µ={mu:.1e}\nPSNR={psnr_value:.2f} dB")  # Display the mu and PSNR value
        plt.axis('off')  # Hide the axis
    
    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()

def plot_pixels_histogram(vanilla_image: np.ndarray, deblurred_image: np.ndarray, ax: plt.Axes, label: str, mu: float):
    """
    Plots the histogram of pixel intensities for the original and deblurred images.

    This function visualizes the distribution of pixel intensities (values between 0 and 1) 
    for both the original and deblurred images. It plots the histograms on the same axes to 
    compare their intensity distributions. The `mu` parameter is included in the title to 
    indicate the specific regularization parameter used during the deblurring process.

    Args:
    vanilla_image (np.ndarray): The original image as a NumPy array (values typically between 0 and 1).
    deblurred_image (np.ndarray): The deblurred image as a NumPy array (values typically between 0 and 1).
    ax (plt.Axes): The Matplotlib axes on which the histogram will be plotted.
    label (str): A string label that describes the type of image processing (e.g., "H1", "L2").
    mu (float): The regularization parameter used during the deblurring process (for example, used in H1 or 
    L2 regularization).

    Returns:
    None: The function modifies the provided `ax` object in-place by plotting the histogram.

    Example:
    >>> fig, ax = plt.subplots()
    >>> plot_pixels_histogram(original_image, deblurred_image, ax, 'L2', 1e-3)
    """
    # Plot the histograms for both the original and deblurred images
    ax.hist([vanilla_image.flatten(), deblurred_image.flatten()], bins=256, range=[0, 1])
    
    # Add a legend and title to the plot
    ax.legend(['Original', 'Deblurred'])
    ax.set_title(f'Histogram of pixel intensities for type {label}\nwith µ={mu:.1e}')
