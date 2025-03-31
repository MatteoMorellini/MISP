import numpy as np
import cv2
from scipy.special import j1


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generates a 2D Gaussian kernel.

    This function creates a Gaussian kernel of a specified size and standard deviation (sigma). 
    The kernel is normalized so that the sum of all its elements is 1, ensuring that it can be 
    used for filtering operations (e.g., convolution).

    Args:
    size (int): The size of the kernel (must be an odd integer). Defines the width and height of the kernel.
    sigma (float): The standard deviation of the Gaussian distribution. Controls the spread of the kernel.

    Returns:
    np.ndarray: A 2D NumPy array representing the Gaussian kernel, normalized to sum to 1.

    Example:
    >>> kernel = gaussian_kernel(5, 1.0)
    >>> print(kernel)
    """
    ax = np.linspace(-(size // 2), size // 2, size)  # Create a 1D axis for the kernel
    xx, yy = np.meshgrid(ax, ax)  # Create 2D meshgrid
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))  # Apply Gaussian function
    return kernel / np.sum(kernel)  # Normalize the kernel

def linear_motion_blur_kernel(kernel_size: int, angle: float) -> np.ndarray:
    """
    Generates a linear motion blur kernel.

    This function creates a square kernel representing a linear motion blur. The blur is applied along
    a straight line, and the direction of the blur is determined by the specified angle. The kernel is
    then normalized so that the sum of all elements equals 1. The kernel is rotated according to the specified angle.

    Args:
    kernel_size (int): The size of the kernel (must be an odd integer). Defines the width and height of the kernel.
    angle (float): The angle (in degrees) of the motion blur. This determines the direction of the blur.

    Returns:
    np.ndarray: A 2D NumPy array representing the linear motion blur kernel, normalized to sum to 1.

    Example:
    >>> kernel = linear_motion_blur_kernel(5, 45)
    >>> print(kernel)
    """
    kernel = np.zeros((kernel_size, kernel_size))  # Initialize kernel with zeros
    center = kernel_size // 2  # Find the center of the kernel
    kernel[center, :] = np.ones(kernel_size)  # Create a horizontal line in the center
    kernel /= kernel_size  # Normalize the kernel so that the sum of elements equals 1
    
    # Create the rotation matrix to rotate the kernel by the given angle
    rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1) 
    
    # Apply the rotation to the kernel
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
    
    return kernel

def out_of_focus_blur_kernel(radius: int) -> np.ndarray:
    """
    Generates an out-of-focus blur kernel, often referred to as a disk blur.

    This function creates a square kernel where a circular region of ones is placed in the center.
    The radius of the circle determines the extent of the blur, and the kernel is normalized so that 
    the sum of all its elements equals 1. This type of kernel can be used to simulate an out-of-focus 
    blur in image processing.

    Args:
    radius (int): The radius of the circular blur region. The kernel size will be `2 * radius + 1`.

    Returns:
    np.ndarray: A 2D NumPy array representing the out-of-focus blur kernel, normalized to sum to 1.

    Example:
    >>> kernel = out_of_focus_blur_kernel(3)
    >>> print(kernel)
    """
    kernel_size = 2 * radius + 1  # Calculate the size of the kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)  # Initialize kernel with zeros
    cv2.circle(kernel, (radius, radius), radius, 1, -1)  # Draw a filled circle at the center
    kernel /= np.sum(kernel)  # Normalize the kernel so that the sum of its elements equals 1
    return kernel

def airy_disk_kernel(radius: float, kernel_size: int) -> np.ndarray:
    """
    Generates an Airy disk kernel, simulating diffraction-limited blur.

    This function creates a square kernel based on the Airy pattern, which models the diffraction 
    of light through a circular aperture. The radius controls the size of the central bright disk, 
    and the kernel is normalized so that the sum of all its elements equals 1. The Airy disk kernel 
    is often used to simulate optical blurring due to diffraction in imaging systems.

    Args:
    radius (float): The radius of the central bright disk in the Airy pattern. This influences 
                    the size of the blur.
    kernel_size (int): The size of the square kernel. It should be large enough to capture the 
                        Airy pattern, typically an odd number for symmetry.

    Returns:
    np.ndarray: A 2D NumPy array representing the Airy disk kernel, normalized to sum to 1.

    Example:
    >>> kernel = airy_disk_kernel(5.0, 21)
    >>> print(kernel)
    """
    # Create a grid of (x, y) coordinates
    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    y = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    X, Y = np.meshgrid(x, y)

    # Calculate the radial distance from the center
    R = np.sqrt(X**2 + Y**2)

    # Calculate the Airy pattern (using the first-order Bessel function of the first kind, j1)
    epsilon = 1e-10  # Small value to avoid division by zero
    Z = 2 * j1(np.pi * R / (radius + epsilon)) / (np.pi * R / (radius + epsilon))

    # Set the center value where R is zero
    Z[kernel_size // 2, kernel_size // 2] = 1.0  # Avoid division by zero at the center
    kernel = Z**2  # Square the Airy pattern to simulate the blur

    # Normalize the kernel so that the sum of all its elements equals 1
    kernel /= np.sum(kernel)
    
    return kernel
