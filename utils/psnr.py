import numpy as np

def psnr(target: np.ndarray, ref: np.ndarray) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR is a metric used to evaluate the quality of a compressed or reconstructed image compared 
    to the original image. It is calculated based on the Mean Squared Error (MSE) between the target 
    (processed) image and the reference (original) image. Higher PSNR values indicate better quality 
    and less distortion.

    Args:
    target (np.ndarray): The target image (processed or noisy image) as a NumPy array.
    ref (np.ndarray): The reference image (original image) as a NumPy array.

    Returns:
    float: The PSNR value in decibels (dB). Returns infinity if the MSE is zero (i.e., the images are identical).

    Example:
    >>> target_image = np.array([[0.5, 0.5], [0.5, 0.5]])
    >>> ref_image = np.array([[1.0, 1.0], [1.0, 1.0]])
    >>> psnr_value = psnr(target_image, ref_image)
    >>> print(psnr_value)
    """
    mse = np.mean((target - ref) ** 2)  # Calculate Mean Squared Error (MSE)
    if mse == 0:
        return float('inf')  # If MSE is zero, images are identical, so PSNR is infinite
    max_pixel = 1.0  # Assuming the pixel values are in the range [0, 1]
    return 20 * np.log10(max_pixel / np.sqrt(mse))  # Calculate and return PSNR
