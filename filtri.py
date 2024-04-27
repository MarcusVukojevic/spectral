import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import io, img_as_float, restoration
import pywt
from scipy.signal import wiener

def median_filtering(img_path, kernel_size=5):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    denoised_image = cv2.medianBlur(image, kernel_size)
    residual = image - denoised_image
    return residual

def bilateral_filtering(img_path, diameter=9, sigma_color=75, sigma_space=75):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    denoised_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    residual = image - denoised_image
    return residual

def adaptive_noise_filter(image_path):
    # Read the image
    image = cv2.imread(image_path, 0)  # 0 to read image in grayscale

    # Estimate the local mean and variance
    mean = cv2.blur(image, (3, 3))
    mean_sq = cv2.blur(image**2, (3, 3))
    variance = mean_sq - mean**2

    # Estimate the noise variance
    noise_variance = np.mean(np.var(image))

    # Apply the adaptive Wiener filter
    with np.errstate(divide='ignore', invalid='ignore'):
        result = mean + np.where(variance == 0, 0, 
                                 (variance - noise_variance) / variance) * (image - mean)
        # Where variance is zero, output mean instead of dividing by zero
        result[variance == 0] = mean[variance == 0]

    # The estimated noise is the difference between the original and the result
    noise = image - result

    return noise.astype(np.uint8)


def gaussian_filter(image_path):

    image = cv2.imread(image_path, 0)

    denoised_image = cv2.GaussianBlur(image, (5, 5), 1)

    # Compute the residual (noise)
    residual = image - denoised_image

    return np.array(residual)


def wavelet_denoise(image, wavelet='db1', level=1):
    # Convert image to float
    image = cv2.imread(image, 0)
    image = image.astype(float)
    
    # Decompose the image into wavelet components
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # Thresholding function
    def threshold(coeffs, median):
        return pywt.threshold(coeffs, median, mode='soft')
    
    # Apply a threshold to the detail coefficients
    coeffs_thresh = [coeffs[0]] + [(threshold(cH, np.median(np.abs(cH))), 
                                    threshold(cV, np.median(np.abs(cV))), 
                                    threshold(cD, np.median(np.abs(cD)))) 
                                   for cH, cV, cD in coeffs[1:]]
    
    # Reconstruct the image from the thresholded coefficients
    denoised_image = pywt.waverec2(coeffs_thresh, wavelet)
    
    # Clip values to be in the range [0, 1]
    denoised_image = np.clip(denoised_image, 0, 1)
    

    residual = image - denoised_image
    return residual


def non_local_means_filter(image_path):
    
    image = io.imread(image_path, as_gray=True)
    

    sigma_est = np.mean(restoration.estimate_sigma(image))
    denoised_image = restoration.denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True,
                                                patch_size=5, patch_distance=6)
    
    residual = image - denoised_image
    return residual


def wiener_filter(image_path):
    image = cv2.imread(image_path, 0)
    image = image.astype(float)
    filtered_image = wiener(image, (5, 5))  

    residual = image - filtered_image
    return residual
