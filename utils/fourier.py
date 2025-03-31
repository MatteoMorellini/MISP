from numpy.fft import fft2, fftshift
import numpy as np

def ft(img):
    """
    Computes the logarithmic Fourier transform spectrum of an image.

    This function performs the following steps:
    1. Computes the 2D Fourier transform of the input image using `fft2`.
    2. Computes the magnitude (spectrum) of the Fourier transform.
    3. Shifts the zero-frequency component to the center of the spectrum using `fftshift`.
    4. Applies a logarithmic transformation to enhance visibility of the spectrum.

    Args:
        img (numpy.ndarray): Input image array (2D).

    Returns:
        numpy.ndarray: Logarithmic Fourier spectrum with the zero-frequency component shifted to the center.
    """
    fourier = fft2(img)
    fourier_spectrum = np.abs(fourier)
    # In the default output from np.fft.fft2(), the zero-frequency component (the average value of the image) is located 
    # in the top-left corner of the Fourier matrix
    fourier_spectrum_shifted = fftshift(fourier_spectrum)
    # shift the zero-frequency component (DC component) to the center of the image (or frequency matrix). After this 
    # shift, the low-frequency components are positioned at the center of the Fourier space, and the high-frequency 
    # components move to the edges. This is useful for visualization purposes.
    log_fourier_spectrum_shifted = np.log(fourier_spectrum_shifted)
    return log_fourier_spectrum_shifted
