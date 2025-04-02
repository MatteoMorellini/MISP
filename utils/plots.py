import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from .deconvolutions import l2_deconvolution, h1_deconvolution
from .psnr import psnr

def plot_blurred_image(blurred_image: np.ndarray, fourier_image:np.ndarray, blur_kernel: np.ndarray, kernel: str) -> None:
    """
    Plots three images in a single row:
    1. The image after applying the specified kernel (blurred image).
    2. The Fourier spectrum of the blurred image.

    Args:
    blurred_image (ndarray): The blurred image obtained after applying the kernel.
    fourier_image (ndarray): The Fourier spectrum (magnitude) of the blurred image.
    kernel (str): The name of the kernel used, displayed in the subplot titles.

    Returns:
    None: This function only displays the images, does not return any value.
    """
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
    
    # Plot the blurred image with kernel name in title
    ax1.set_title(f"{kernel} blurred image")
    ax1.imshow(blurred_image, cmap='gray')
    ax1.axis('off')
    
    # Plot the Fourier spectrum of the blurred image
    ax2.set_title(f"Fourier spectrum of\n{kernel} blurred image")
    ax2.imshow(fourier_image + 1, cmap="inferno")  # Added +1 for better visibility
    ax2.axis('off')

    ax3.set_title(f"{kernel} Blur Kernel")
    ax3.imshow(blur_kernel,  cmap='gray')
    ax3.axis('off')

    
    # Adjust the layout and show the plots
    plt.tight_layout()
    plt.show()

def plot_image_blurred_noise(vanilla_image: np.ndarray, blurred_image: np.ndarray, noise_image: np.ndarray) -> None:
    """
    Plots three images in a single row:
    1. The original image.
    2. The image after applying the specified kernel (blurred image).
    3. The blurred image with noise added, showing the specified variance in the title.

    Args:
    vanilla_image (ndarray): The original image to be displayed.
    blurred_image (ndarray): The blurred image obtained after applying the kernel.
    noisy_image (ndarray): The image after noise has been added to the blurred image.
    kernel (str): The name of the kernel used, displayed in the subplot titles.
    sigma (str): The variance of the noise added to the blurred image, displayed in the subplot title.

    Returns:
    None: This function only displays the images, does not return any value.
    """
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 6))
    
    # Plot the original image
    ax1.set_title("Image")
    ax1.imshow(vanilla_image, cmap='gray')
    ax1.axis('off')
    
    # Plot the blurred image with kernel name in title
    ax2.set_title(f"Blurred image")
    ax2.imshow(blurred_image, cmap='gray')
    ax2.axis('off')
    
    # Plot the blurred image with noise and variance in title
    ax3.set_title(f"Blurred image with noise")
    ax3.imshow(noise_image, cmap="gray")
    ax3.axis('off')
    
    # Adjust the layout and show the plots
    plt.tight_layout()
    plt.show()

def get_inverse(regularization: str, vanilla_image: np.ndarray, blurred_image: np.ndarray,  \
                    blur_kernel: np.ndarray, window: np.ndarray = None, iter: int = 8, log_range: \
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
    iter (int, optional): The number of iterations (i.e., the number of different mu values). Default is 7.
    log_range (tuple, optional): Exponent range for mu values, defined as a tuple (start, stop) for logspace. Default is (-6, 1).

    Returns:
    tuple: A tuple containing:
        - PSNRs (list of float): List of PSNR values for each deconvolved image.
        - inverted_images (list of ndarray): List of deconvolved images for each mu value.
        - MUs (ndarray): The array of mu values used for deconvolution.
    """
    inverted_images = []
    PSNRs = []
    start, stop = log_range
    deconvolved = None
    MUs = np.logspace(start, stop, num=iter)  # Generate mu values

    assert regularization in ["H1", "L2"], "Regularization method must be 'H1' or 'L2'."
    
    for mu in MUs:
        # Apply the specified regularization method
        if regularization == "H1":
            deconvolved = h1_deconvolution(blurred_image, blur_kernel, mu)
        elif regularization == "L2":
            deconvolved = l2_deconvolution(blurred_image, blur_kernel, mu)
        
        
        # Normalize the deconvolved image if a window is provided
        if window is not None:
            epsilon = 1e-10  # Small value to avoid division by zero
            deconvolved = np.clip(deconvolved / (window + epsilon), 0, 1)
        
        # Append the deconvolved image and PSNR value
        inverted_images.append(deconvolved)
        PSNRs.append(psnr(deconvolved, vanilla_image))

    return inverted_images, PSNRs, MUs

def plot_inverse_images(deconvolved_images: List[np.ndarray], psnr_values: List[float], \
                                             mu_values: np.ndarray) -> None:
    """
    Plots deconvolved images for several values of mu and displays their corresponding PSNR values.

    This function creates a grid of subplots where each subplot displays a deconvolved image for a specific mu value, 
    along with the corresponding PSNR value. The images are arranged in a grid with a specified number of columns.

    Args:
    deconvolved_images (list of ndarray): List of deconvolved images, one for each mu value.
    psnr_values (list of float): List of PSNR values corresponding to each deconvolved image.
    mu_values (ndarray): Array of mu values used for deconvolution.

    Returns:
    None: This function only displays the images, does not return any value.
    """
    # Define grid parameters
    num_columns = 4
    num_rows = -(-len(mu_values) // num_columns) 

    # Initialize the figure
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(8, 2 * num_rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]  # Ensure axes is iterable

    # Iterate over images and display them
    for idx, (mu, img, psnr) in enumerate(zip(mu_values, deconvolved_images, psnr_values)):
        ax = axes[idx]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"µ={mu:.1e}\nPSNR={psnr:.2f} dB")
        ax.axis('off')

    # Hide unused subplots if any
    for ax in axes[len(mu_values):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    

def plot_psnr_vs_mu(MUs, PSNRs, title="Relation between µ and PSNR"):
    """
    Plots the relationship between the regularization parameter (µ) and PSNR.

    This function takes a set of regularization parameter values (MUs) and their 
    corresponding PSNR values (PSNRs) and plots them on a log-scaled x-axis.

    Parameters:
    -----------
    MUs : array-like
        A sequence of regularization parameter values (µ).
    PSNRs : array-like
        A sequence of Peak Signal-to-Noise Ratio (PSNR) values in decibels.
    title : str, optional
        The title of the plot (default is "Relation between µ and PSNR").

    Notes:
    ------
    - The x-axis is set to a logarithmic scale to better visualize the effect of µ.
    - A grid is added to improve readability.

    Example:
    --------
    >>> MUs = [0.001, 0.01, 0.1, 1, 10]
    >>> PSNRs = [25, 28, 30, 29, 27]
    >>> plot_psnr_vs_mu(MUs, PSNRs)
    """
    plt.figure(figsize=(8, 6))
    plt.plot(MUs, PSNRs)
    plt.xscale('log')
    plt.xlabel("Regularization parameter µ on a log scale")
    plt.ylabel("PSNR in decibel")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

def plot_psnr_comparison(mu_values_h1, psnr_values_h1, mu_values_l2, psnr_values_l2):
    """
    Plots the PSNR comparison for H¹ and L² regularization methods.

    This function visualizes how the Peak Signal-to-Noise Ratio (PSNR) changes with 
    different regularization parameters (µ) for both H¹ and L² deconvolution methods. 
    The x-axis is set to a logarithmic scale for better visualization.

    Parameters:
    -----------
    mu_values_h1 : array-like
        A sequence of regularization parameter values (µ) for H¹ regularization.
    psnr_values_h1 : array-like
        A sequence of PSNR values (in decibels) corresponding to H¹ regularization.
    mu_values_l2 : array-like
        A sequence of regularization parameter values (µ) for L² regularization.
    psnr_values_l2 : array-like
        A sequence of PSNR values (in decibels) corresponding to L² regularization.

    Notes:
    ------
    - H¹ regularization results are plotted in **green**.
    - L² regularization results are plotted in **red**.
    - The x-axis is log-scaled for better visualization of µ's impact.

    Example:
    --------
    >>> mu_h1 = [0.001, 0.01, 0.1, 1, 10]
    >>> psnr_h1 = [26, 28, 30, 29, 27]
    >>> mu_l2 = [0.001, 0.01, 0.1, 1, 10]
    >>> psnr_l2 = [24, 27, 29, 28, 25]
    >>> plot_psnr_comparison(mu_h1, psnr_h1, mu_l2, psnr_l2)
    """
    plt.figure(figsize=(8, 6))

    # Plot H1 regularization results in green
    plt.plot(mu_values_h1, psnr_values_h1, marker='o', linestyle='-', color='g', label="H1 Regularization")

    # Plot L2 regularization results in red
    plt.plot(mu_values_l2, psnr_values_l2, marker='o', linestyle='-', color='r', label="L2 Regularization")

    plt.xscale('log')
    plt.xlabel("Regularization Parameter µ (log scale)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs. µ for H¹ and L² Deconvolution")
    plt.grid(True, which="both", linestyle="--", linewidth=1)
    plt.legend()
    plt.show()


def plot_psnr_difference(mu_values, psnr_values_h1, psnr_values_l2):
    """
    Plots the PSNR difference between H1 and L2 regularization methods.

    This function computes and visualizes the difference in Peak Signal-to-Noise Ratio (PSNR) 
    between H1 and L2 deconvolution across different regularization parameter values (µ). 
    A reference line at 0 dB is included to indicate no difference.

    Parameters:
    -----------
    mu_values : array-like
        A sequence of regularization parameter values (µ).
    psnr_values_h1 : array-like
        A sequence of PSNR values (in decibels) for H1 regularization.
    psnr_values_l2 : array-like
        A sequence of PSNR values (in decibels) for L2 regularization.

    Notes:
    ------
    - The difference is calculated as **PSNR(H1) - PSNR(L2)**.
    - A horizontal dashed line at **0 dB** serves as a reference.
    - The x-axis is logarithmic to highlight variations in µ.

    Example:
    --------
    >>> import numpy as np
    >>> mu_vals = [0.001, 0.01, 0.1, 1, 10]
    >>> psnr_h1 = [26, 28, 30, 29, 27]
    >>> psnr_l2 = [24, 27, 29, 28, 25]
    >>> plot_psnr_difference(mu_vals, psnr_h1, psnr_l2)
    """
    psnr_diff = np.array(psnr_values_h1) - np.array(psnr_values_l2)
    plt.figure(figsize=(8, 6))
    plt.plot(mu_values, psnr_diff, linestyle='-', color='blue', label="H1 - L2 PSNR Difference")
    plt.xscale('log')
    plt.xlabel("Regularization Parameter µ (log scale)")
    plt.ylabel("PSNR Difference (H1 - L2) [dB]")
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Reference line at 0 dB
    plt.grid(True, which="both", linestyle="--", linewidth=1)
    plt.legend()
    plt.show()
