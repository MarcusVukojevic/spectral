import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


image_path = "biggan_img.png"

image = Image.open(image_path).convert('L')  # convert image to grayscale
image_array = np.array(image)


f_righe = np.fft.fft(image_array, axis=1)
f_righe_colonne = np.fft.fft(f_righe, axis=0)
f_shift = np.fft.fftshift(f_righe_colonne)

magnitude_spectrum = 20 * np.log(np.abs(f_shift))

f_colonne = np.fft.fft(image_array, axis=0)
f_colonne_righe = np.fft.fft(f_colonne, axis=1)
f_shift2 = np.fft.fftshift(f_colonne_righe)
magnitude_spectrum2 = 20 * np.log(np.abs(f_shift2))


"""
print(np.abs(f_shift.shape))
posizioni_non_zero = np.nonzero(f_shift)
f_shift[posizioni_non_zero] = 255
for posizione in zip(posizioni_non_zero[0], posizioni_non_zero[1]):
    print(posizione)
"""


plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(image_array)
plt.title('image_array')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(np.abs(f_righe))
plt.title('f_righe')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(np.abs(f_colonne))
plt.title('f_colonne')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(np.abs(f_shift))
plt.title('f_righe_colonne')
plt.axis('off')

plt.show()