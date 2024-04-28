import os
from filtri import median_filtering, bilateral_filtering, adaptive_noise_filter, gaussian_filter, wavelet_denoise, non_local_means_filter, wiener_filter
import numpy as np
import matplotlib.pyplot as plt 
from fingerprint import remove_cross, find_fingerprint, bandpass_filter, normalization_fft_2

cartella_input = os.listdir("TestSet")#"biggan_512"
cartella_target = "best_plots_altri_filtri_reverse/"


for cartella in cartella_input:
    if cartella == ".DS_Store":
        continue

    lista_immagini = os.listdir(f"TestSet/{cartella}")


    median = []
    bilateral = []
    adaptive = []
    gaussian = []
    wave = []
    non_local = []
    wiener = []


    # mi restituisce direttamente il residuo per ogni filtro
    for i in lista_immagini:
        immagine = f"TestSet/{cartella}/{i}"
        # Median filter
        rumore_median = median_filtering(immagine)
        rumore_median_senza_dc = rumore_median - np.mean(rumore_median)
        median.append(bandpass_filter(rumore_median_senza_dc))

        # Bilateral filter
        rumore_bilateral = bilateral_filtering(immagine)
        rumore_bilateral_senza_dc = rumore_bilateral - np.mean(rumore_bilateral)
        bilateral.append(bandpass_filter(rumore_bilateral_senza_dc))

        # Adaptive filter
        rumore_adaptive = adaptive_noise_filter(immagine)
        rumore_adaptive_senza_dc = rumore_adaptive - np.mean(rumore_adaptive)
        adaptive.append(bandpass_filter(rumore_adaptive_senza_dc))

        # Gaussian filter
        rumore_gaussian = gaussian_filter(immagine)
        rumore_gaussian_senza_dc = rumore_gaussian - np.mean(rumore_gaussian)
        gaussian.append(bandpass_filter(rumore_gaussian_senza_dc))

        # Wavelet denoise
        rumore_wave = wavelet_denoise(immagine)
        rumore_wave_senza_dc = rumore_wave - np.mean(rumore_wave)
        wave.append(bandpass_filter(rumore_wave_senza_dc))

        # Non-local means filter
        rumore_non_local = non_local_means_filter(immagine)
        rumore_non_local_senza_dc = rumore_non_local - np.mean(rumore_non_local)
        non_local.append(bandpass_filter(rumore_non_local_senza_dc))

        # Wiener filter
        rumore_wiener = wiener_filter(immagine)
        rumore_wiener_senza_dc = rumore_wiener - np.mean(rumore_wiener)
        wiener.append(bandpass_filter(rumore_wiener_senza_dc))


    media_median = np.mean(median, axis=0)
    media_bilateral = np.mean(bilateral, axis=0)
    media_adaptive = np.mean(adaptive, axis=0)
    media_gaussian = np.mean(gaussian, axis=0)
    media_wave = np.mean(wave, axis=0)
    media_non_local = np.mean(non_local, axis=0)
    media_wiener = np.mean(wiener, axis=0)


    lista_medie = [media_median, media_bilateral, media_adaptive, media_gaussian, media_wave, media_non_local, media_wiener]
    titoli = ["Median filter", "Bilateral Filter", "Adaptive Noise filter", "Gaussian Filter", "Wavelet Filter", "Non Local Means filter", "Wiener Filter"]


    for i in zip(lista_medie, titoli):
        #f_transform = np.fft.fft2(i[0])
        #f_shift = np.fft.fftshift(f_transform)
        #magnitude_spectrum = 20 * np.log(np.abs(f_shift))
        finale = normalization_fft_2(i[0])
        plt.imshow(finale)
        plt.title(f'{i[1]}'), plt.xticks([]), plt.yticks([])
        plt.savefig(f'{cartella_target}{cartella}_{i[1]}.png')