import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Caricamento dell'immagine
image_path = "biggan_img.png"  # Assicurati che il percorso sia corretto
image = Image.open(image_path)
image_array = np.array(image)

# Separazione dei canali di colore
red_channel = image_array[:, :, 0]
green_channel = image_array[:, :, 1]
blue_channel = image_array[:, :, 2]

# Calcolo della FFT per ciascun canale di colore
f_transform_red = np.fft.fft2(red_channel)
f_transform_green = np.fft.fft2(green_channel)
f_transform_blue = np.fft.fft2(blue_channel)

# Shift della FFT
f_shift_red = np.fft.fftshift(f_transform_red)
f_shift_green = np.fft.fftshift(f_transform_green)
f_shift_blue = np.fft.fftshift(f_transform_blue)

# Calcolo della magnitudine della FFT per ciascun canale
magnitude_spectrum_red = 20 * np.log(np.abs(f_shift_red))
magnitude_spectrum_green = 20 * np.log(np.abs(f_shift_green))
magnitude_spectrum_blue = 20 * np.log(np.abs(f_shift_blue))

######################################################### normalization
# Trova il valore massimo e minimo tra i tre spettri di magnitudine
max_value = max(np.max(magnitude_spectrum_red), np.max(magnitude_spectrum_green), np.max(magnitude_spectrum_blue))
min_value = min(np.min(magnitude_spectrum_red), np.min(magnitude_spectrum_green), np.min(magnitude_spectrum_blue))

# Normalizza i valori sulla scala 0-255
magnitude_spectrum_red_normalized = np.interp(magnitude_spectrum_red, (min_value, max_value), (0, 255))
magnitude_spectrum_green_normalized = np.interp(magnitude_spectrum_green, (min_value, max_value), (0, 255))
magnitude_spectrum_blue_normalized = np.interp(magnitude_spectrum_blue, (min_value, max_value), (0, 255))

# Converti i risultati in interi
magnitude_spectrum_red_uint8 = magnitude_spectrum_red_normalized.astype(np.uint8)
magnitude_spectrum_green_uint8 = magnitude_spectrum_green_normalized.astype(np.uint8)
magnitude_spectrum_blue_uint8 = magnitude_spectrum_blue_normalized.astype(np.uint8)

######################################################### remapping
# Rimappatura dei valori
remapped_values_red = np.zeros_like(magnitude_spectrum_red_uint8)
remapped_values_green = np.zeros_like(magnitude_spectrum_green_uint8)
remapped_values_blue = np.zeros_like(magnitude_spectrum_blue_uint8)

# Definizione degli intervalli e dei valori corrispondenti
intervals = [0, 51, 101, 151, 201, 256]
values = [0, 50, 100, 150, 255]

# Utilizzo di np.digitize per trovare gli indici dei bin corrispondenti
bins_red = np.digitize(magnitude_spectrum_red_uint8, intervals)
bins_green = np.digitize(magnitude_spectrum_green_uint8, intervals)
bins_blue = np.digitize(magnitude_spectrum_blue_uint8, intervals)

# Assegnazione dei valori rimappati basati sui bin
for i in range(len(values)):
    remapped_values_red[bins_red == i] = values[i]
    remapped_values_green[bins_green == i] = values[i]
    remapped_values_blue[bins_blue == i] = values[i]


######################################################### threshold + normalization
if(1):
    invert = False
    threshold_low = 175
    threshold_high = 225
    magnitude_spectrum_red[magnitude_spectrum_red < threshold_low] = threshold_low
    magnitude_spectrum_green[magnitude_spectrum_green < threshold_low] = threshold_low
    magnitude_spectrum_blue[magnitude_spectrum_blue < threshold_low] = threshold_low

    magnitude_spectrum_red[magnitude_spectrum_red > threshold_high] = threshold_high
    magnitude_spectrum_green[magnitude_spectrum_green > threshold_high] = threshold_high
    magnitude_spectrum_blue[magnitude_spectrum_blue > threshold_high] = threshold_high

    # Trova il valore massimo e minimo tra i tre spettri di magnitudine
    max_value = max(np.max(magnitude_spectrum_red), np.max(magnitude_spectrum_green), np.max(magnitude_spectrum_blue))
    min_value = min(np.min(magnitude_spectrum_red), np.min(magnitude_spectrum_green), np.min(magnitude_spectrum_blue))

    # Normalizza i valori sulla scala 0-255
    magnitude_spectrum_red_normalized = np.interp(magnitude_spectrum_red, (min_value, max_value), (0, 255))
    magnitude_spectrum_green_normalized = np.interp(magnitude_spectrum_green, (min_value, max_value), (0, 255))
    magnitude_spectrum_blue_normalized = np.interp(magnitude_spectrum_blue, (min_value, max_value), (0, 255))

    if(invert):
        magnitude_spectrum_red_normalized = 255 - magnitude_spectrum_red_normalized
        magnitude_spectrum_green_normalized = 255 - magnitude_spectrum_green_normalized
        magnitude_spectrum_blue_normalized = 255 - magnitude_spectrum_blue_normalized

    # Converti i risultati in interi
    magnitude_spectrum_red_uint8 = magnitude_spectrum_red_normalized.astype(np.uint8)
    magnitude_spectrum_green_uint8 = magnitude_spectrum_green_normalized.astype(np.uint8)
    magnitude_spectrum_blue_uint8 = magnitude_spectrum_blue_normalized.astype(np.uint8)


######################################################### print
# Visualizzazione della magnitudine della FFT per ciascun canale
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(magnitude_spectrum_red_uint8, cmap='gray')
plt.title('FFT - Red Channel')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(magnitude_spectrum_green_uint8, cmap='gray')
plt.title('FFT - Green Channel')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(magnitude_spectrum_blue_uint8, cmap='gray')
plt.title('FFT - Blue Channel')
plt.axis('off')

plt.show()

##########################################################
# Calcola gli elementi unici e le loro frequenze all'interno dell'array
unique_values, counts = np.unique(magnitude_spectrum_red_uint8, return_counts=True)

# Crea una lista dei valori da 0 a 255
all_values = np.arange(256)

# Inizializza un array di contatori per tutti i valori possibili da 0 a 255 con contatore iniziale 0
counters = np.zeros(256, dtype=int)

# Assegna i contatori calcolati agli elementi della lista dei valori
counters[unique_values] = counts
#print(counters)
###################################################################################################################


if(0): #check compression fft
    f_transform_red_rows = np.fft.fft(red_channel, axis=1)
    f_transform_red_rows[:, 129:] = 0
    f_transform_red_rows[:, 0] = f_transform_red_rows[:, 0]/2
    f_transform_red_rows[:, 128] = f_transform_red_rows[:, 128]/2
    inv_fft_red = np.fft.ifft(f_transform_red_rows, axis=1).real*2

    reconstructed_image = np.copy(image_array)
    reconstructed_image[:,:,0] = inv_fft_red



    # Visualizzazione dell'immagine originale e dell'immagine ricostruita
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image_array)
    plt.title('Immagine Originale')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image)
    plt.title('Immagine Ricostruita')
    plt.axis('off')

    plt.show()

elif(0):
    f_transform_red_rows = np.fft.fft(red_channel,axis=1)
    f_transform_red_rows[:, 0] = f_transform_red_rows[:, 0]/2
    f_transform_red_rows[:, 128] = f_transform_red_rows[:, 128]/2
    f_transform_red_rows[:, 129:] = 0
    f_transform_red_cols = np.fft.fft(f_transform_red_rows,axis=0)
    f_transform_red_cols[0, :] = f_transform_red_cols[0, :]/2
    f_transform_red_cols[128, :] = f_transform_red_cols[128, :]/2
    f_transform_red_cols[129:, :] = 0
    inv_fft_red_cols = np.fft.ifft(f_transform_red_cols, axis=0).real*2
    inv_fft_red = np.fft.ifft(inv_fft_red_cols, axis=1).real*2


    print(f_transform_red_rows)
    print(inv_fft_red_cols)

    values_equal_rounded = np.round(red_channel.astype(float), decimals=10) == np.round(inv_fft_red, decimals=10)

    print("rounded values are equal -> ", values_equal_rounded.all())

# demonstration impossibility to direct reconstract the image
if(0): #even
#array_start = np.array([1.5+0.5j, 2.7+0.7j, 3.9+0.3j, 2.4+5.4j, 6.4+2.2j, 3.5+3.5j, 2.2+1.1j, 1.9+3.8j])
    array_start = np.array([1.5, 2.7, 3.9, 2.4, 6.4, 3.5, 2.2, 1.9])
    f_rows = np.fft.fft(array_start)
    f_rows_tmp = f_rows.copy()
    array_end_tmp = np.fft.ifft(f_rows)
    f_rows[0] = f_rows[0]/2
    f_rows[4] = f_rows[4]/2
    f_rows[5:] = 0
    array_end = np.fft.ifft(f_rows).real*2

    print(array_start)
    print(f_rows_tmp)
    print(f_rows)
    print(array_end)

elif(0): #odd
    #array_start = np.array([1.5+0.5j, 2.7+0.7j, 3.9+0.3j, 2.4+5.4j, 6.4+2.2j, 3.5+3.5j, 2.2+1.1j])
    array_start = np.array([1.5, 0.0, -1.5, 0.0, 1.5, 0.0, -1.5])
    f_rows = np.fft.fft(array_start)
    f_rows_tmp = f_rows.copy()
    array_end_tmp = np.fft.ifft(f_rows)
    f_rows[0] = f_rows[0]/2
    f_rows[4:] = 0
    array_end = np.fft.ifft(f_rows)*2

    print(array_start)
    print(f_rows_tmp)
    print(f_rows)
    print(array_end_tmp)
    print(array_end)


# demonstration impossibility to indirect reconstract the image
if(0): #even (same for odd)
    array_start = np.array([1.5+0.5j, 2.7+0.7j, 3.9+0.3j, 2.4+5.4j, 6.4+2.2j, 3.5+3.5j, 2.2+1.1j, 1.9+3.8j])
    #array_start = np.array([1.5, 2.7, 3.9, 2.4, 6.4, 3.5, 2.2, 1.9])
    f_rows = np.fft.fft(array_start)
    f_rows_tmp = f_rows.copy()
    array_end_tmp = np.fft.ifft(f_rows)
    f_rows[5:] = 0
    f_restore = f_rows.copy()
    f_restore[5:].real = f_rows[1:4][::-1].real
    f_restore[5:].imag = -f_rows[1:4][::-1].imag
    array_end = np.fft.ifft(f_rows)

    print(array_start)
    print(f_rows_tmp.real)
    print(f_restore)
    print(array_end)
