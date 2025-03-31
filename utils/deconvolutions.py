import numpy as np
from numpy.fft import fft2, ifft2

def l2_deconvolution(noisy_blurred: np.ndarray, blur_kernel: np.ndarray, mu: float) -> np.ndarray:
    """
    Performs L2 deconvolution to recover an image from its noisy blurred version.

    This function applies L2 regularization during the deconvolution process, which aims to reduce the 
    noise introduced in the deconvolution step by adding a regularization term controlled by the parameter `mu`.
    The process is carried out in the Fourier domain, where the deconvolution is computed as the element-wise 
    multiplication of the Fourier transforms of the noisy blurred image and the conjugate of the blur kernel, 
    divided by the magnitude of the kernel's Fourier transform squared plus `mu`. This method is robust for 
    handling noise in the blurred image.

    Args:
    noisy_blurred (np.ndarray): The noisy blurred image as a NumPy array, where pixel values are typically in the range [0, 1].
    blur_kernel (np.ndarray): The blur kernel, representing the blurring process, as a NumPy array.
    mu (float): The regularization parameter. A small positive value that controls the strength of the regularization. 
                Higher values of `mu` reduce the effect of the kernel but can help to suppress noise.

    Returns:
    np.ndarray: The deconvolved image as a NumPy array, which is an estimation of the original image.
    
    Example:
    >>> deblurred_image = l2_deconvolution(noisy_image, kernel, mu=1e-3)
    """
    G = fft2(noisy_blurred)
    H = fft2(blur_kernel, s=noisy_blurred.shape)

    denom = np.abs(H)**2 + mu
    denom = np.where(denom == 0, 1e-10, denom)  

    F_hat = (np.conj(H) * G) / denom

    f_deconvolved = np.real(ifft2(F_hat))
    
    return f_deconvolved

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

def h1_deconvolution(noisy_blurred: np.ndarray, blur_kernel: np.ndarray, mu: float) -> np.ndarray:
    """
    Performs H1 deconvolution to recover an image from its noisy blurred version.

    This function applies H1 regularization during the deconvolution process. The method combines Fourier 
    domain deconvolution with a regularization term that depends on the frequency of the components. 
    The regularization term helps in suppressing noise and preventing instability in the deconvolution process.

    Args:
    noisy_blurred (np.ndarray): The noisy blurred image as a NumPy array, with pixel values typically in the range [0, 1].
    blur_kernel (np.ndarray): The blur kernel, representing the blurring process, as a NumPy array.
    mu (float): The regularization parameter. This parameter controls the strength of the regularization. 
                Higher values of `mu` reduce the impact of the kernel but can suppress noise more effectively.

    Returns:
    np.ndarray: The deconvolved image as a NumPy array, which is an estimation of the original image.

    Example:
    >>> deblurred_image = h1_deconvolution(noisy_image, kernel, mu=1e-3)
    """
    m, n = noisy_blurred.shape

    # Compute the Fourier transforms of the noisy blurred image and the blur kernel
    G = fft2(noisy_blurred)
    H = fft2(blur_kernel, s=noisy_blurred.shape)

    # Create frequency grids
    u = fftfreq(m).reshape(-1, 1)
    v = fftfreq(n).reshape(1, -1)

    # Compute |omega|^2 = u^2 + v^2
    omega_squared = u**2 + v**2

    # Compute the denominator in the Fourier domain, including the regularization term
    denom = np.abs(H)**2 + mu * (1 + omega_squared)
    denom = np.where(denom == 0, 1e-10, denom)  # Avoid division by zero

    # Compute the deconvolved image in the Fourier domain
    F_hat = (np.conj(H) * G) / denom

    # Inverse Fourier transform to get the deconvolved image in the spatial domain
    f_deconvolved = np.real(ifft2(F_hat))
    
    return f_deconvolved
