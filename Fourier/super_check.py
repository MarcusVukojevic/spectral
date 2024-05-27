import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def scale_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    scaled_image = ((image - min_val) / (max_val - min_val)) * 255
    scaled_image = scaled_image.astype(np.uint8)
    return scaled_image



directory = "../../TestSet"
subfolders = [subfolder for subfolder in os.listdir(directory) if os.path.isdir(os.path.join(directory, subfolder))]
counter = 0

for subfolder in subfolders:
    counter+=1
    if(counter!=1):
        continue
    subfolder_path = os.path.join(directory, subfolder)

    images = []
    for filename in os.listdir(subfolder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(subfolder_path, filename)
            image = Image.open(image_path).convert('L')  # convert image to grayscale
            image_array = np.array(image)

            # Gaussian filter
            image_filtered = cv2.GaussianBlur(image_array, (5, 5), 0)
            residual = image_array-image_filtered
            #residual_normalized = scale_image(residual)

            f_righe = np.fft.fft(residual, axis=1)
            f_colonne_righe = np.fft.fft(f_righe, axis=0)
            f_shift = np.fft.fftshift(f_colonne_righe)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift))

            f_colonne = np.fft.fft(residual, axis=0)
            f_righe_colonne = np.fft.fft(f_colonne, axis=1)
            f_shift2 = np.fft.fftshift(f_righe_colonne)
            magnitude_spectrum2 = 20 * np.log(np.abs(f_shift2))

            """
            plt.imshow(magnitude_spectrum)
            plt.axis('off')
            plt.savefig('magnitude_spectrum.png')

            plt.imshow(magnitude_spectrum2)
            plt.axis('off')
            plt.savefig('magnitude_spectrum2.png')
            """

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 4, 1)
            plt.imshow(residual)
            plt.title('residual')
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.imshow(magnitude_spectrum)
            plt.title('magnitude_spectrum')
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.imshow(magnitude_spectrum2)
            plt.title('magnitude_spectrum2')
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.imshow(magnitude_spectrum2)
            plt.title('ignore')
            plt.axis('off')

            plt.show()





