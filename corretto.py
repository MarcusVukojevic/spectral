import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from fingerprint import remove_cross, find_fingerprint, bandpass_filter, normalization_fft_2
from copy import deepcopy
import torch



'''
lista_folders = [ "TestSet/dalle-mini_valid",  "TestSet/dalle_2", "TestSet/eg3d", "TestSet/progan_lsun", "TestSet/stable_diffusion_256", "TestSet/biggan_256", \
                  "TestSet/taming-transformers_class2image_ImageNet", "TestSet/latent-diffusion_noise2image_FFHQ",  "TestSet/glide_text2img_valid", "TestSet/stylegan3_r_ffhqu_1024x1024" \
                  "TestSet/stylegan3_t_ffhqu_1024x1024" ,  "TestSet/stylegan2_afhqv2_512x512",  "TestSet/biggan_512"]
'''
lista_folders = os.listdir("TestSet")
lista_folders = ["biggan_256"]

for j in lista_folders:
    if os.path.isdir(os.path.join("TestSet/", j)) and j != "dalle_2":
        print(j)
        
        cartella_ori = f"TestSet/{j}"
        cartella_noise = f"rumore_estratto/{j}_dn_drunet_gray"


        lista_immagini_1 = os.listdir(cartella_ori)
        lista_immagini_2 = os.listdir(cartella_noise)

        # Estrai i nomi dei file dalle liste
        nomi_file_1 = [file.split(".")[0] for file in lista_immagini_1]
        nomi_file_2 = [file.split(".")[0] for file in lista_immagini_2]

        # Ordina i nomi dei file
        nomi_file_1.sort()
        nomi_file_2.sort()


        # Riordina le liste delle immagini in base all'ordinamento dei nomi dei file
        lista_immagini_1_ordinate = [file + ".png" for file in nomi_file_1]
        lista_immagini_2_ordinate = [file + ".png" for file in nomi_file_2]

        lista_immagini_rumore = []

        seconda = len(lista_immagini_1_ordinate)
        if len(lista_immagini_1_ordinate) > len(lista_immagini_2_ordinate):
            seconda = len(lista_immagini_2_ordinate)

        print("Creo il residuo")
        for i in range(seconda):
            if i == 0:
                continue
            image_original = Image.open(f"{cartella_ori}/{lista_immagini_1_ordinate[i]}").convert("L")
            image_array_original = np.array(image_original)
            

            image_denoised = Image.open(f"{cartella_noise}/{lista_immagini_2_ordinate[i]}").convert("L")
            image_array_denoised = np.array(image_denoised)


            rumore = image_array_original - image_array_denoised
            
            # Plot the first array
            """
            plt.subplot(1, 3, 1)  # Create subplot for the first array
            plt.imshow(image_original, cmap='gray')  # Plot the first array
            plt.title('Original')  # Set title for the first array

            # Plot the second array
            plt.subplot(1, 3, 2)  # Create subplot for the second array
            plt.imshow(image_array_denoised, cmap='gray')  # Plot the second array
            plt.title('Denoised')  # Set title for the second array

            plt.subplot(1, 3, 3)  # Create subplot for the second array
            plt.imshow(rumore, cmap='gray')  # Plot the second array
            plt.title('Noise')  # Set title for the second array

            plt.show()
            """
            #print(rumore.shape)
            rumore_senza_dc = rumore - np.mean(rumore)
            # Aggiungi il rumore alla lista dei rumori
            lista_immagini_rumore.append(bandpass_filter(rumore_senza_dc))
            
        
        print("Calcolo media")
        media_rumore_complessiva = np.mean(lista_immagini_rumore, axis=0)
        #print(media_rumore_complessiva.shape)
        f_transform = np.fft.fft2(media_rumore_complessiva)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift))

        #finale = find_fingerprint(remove_cross(magnitude_spectrum), 50)[0]
        finale = normalization_fft_2(media_rumore_complessiva)
        
        plt.imshow(finale)
        plt.title(f'{j}'), plt.xticks([]), plt.yticks([])
        plt.savefig(f'{j}.png')
        