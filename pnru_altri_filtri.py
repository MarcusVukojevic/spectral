import os
from filtri import median_filtering, bilateral_filtering, adaptive_noise_filter, gaussian_filter, wavelet_denoise, non_local_means_filter, wiener_filter
import numpy as np
import matplotlib.pyplot as plt 


cartella_input = os.listdir("TestSet")#"biggan_512"
cartella_target = "plots_altri_filtri/"


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
        median.append(median_filtering(immagine))
        bilateral.append(bilateral_filtering(immagine))
        adaptive.append(adaptive_noise_filter(immagine))
        gaussian.append(gaussian_filter(immagine))
        wave.append(wavelet_denoise(immagine))
        non_local.append(non_local_means_filter(immagine))
        wiener.append(wiener_filter(immagine))


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
        f_transform = np.fft.fft2(i[0])
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift))
        plt.imshow(magnitude_spectrum)
        plt.title(f'{i[1]}'), plt.xticks([]), plt.yticks([])
        plt.savefig(f'{cartella_target}{i[1]}_{cartella}.png')