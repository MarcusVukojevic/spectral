import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def apply_gaussian_blur(image):
    # Applica un filtro gaussiano per ridurre il rumore
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image

def apply_fourier_transform(image_path):
    # Load the image
    image = Image.open(image_path).convert('L')  # convert image to grayscale
    image_array = np.array(image)

    # Gaussian filter
    image_filtered = apply_gaussian_blur(image_array)
    residual = image_array-image_filtered
    residual_normalized = scale_image(residual)

    # Apply Fourier Transform
    f_transform = np.fft.fft2(residual_normalized)
    f_shift = np.fft.fftshift(f_transform)
    
    # Get magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    
    return magnitude_spectrum


def load_images_and_apply_fourier(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            magnitude_spectrum = apply_fourier_transform(image_path)
            images.append(magnitude_spectrum)
    return images

def create_average_image(images):
    num_images = len(images)
    average_image = np.zeros_like(images[0], dtype=np.float64)
    for image in images:
        average_image += image / num_images
    return average_image

def scale_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    scaled_image = ((image - min_val) / (max_val - min_val)) * 255
    scaled_image = scaled_image.astype(np.uint8)
    return scaled_image


def scale_image2(image):
    height, width = image.shape
    center_x = width // 2
    center_y = height // 2
    
    hyperbola = (np.abs(np.arange(height)[:, None] - center_y) * np.abs(np.arange(width)[None, :] - center_x))
    
    mask = hyperbola <= height*width/128

    min_val = np.min(image)
    max_val = np.max(image)
    scaled_image = ((image - min_val) / (max_val - min_val)) * 255
    scaled_image[mask] *= 0.6
    unscaled_pixels = scaled_image[~mask] * 1.5
    scaled_image[~mask] = np.minimum(unscaled_pixels, 255)
    scaled_image = scaled_image.astype(np.uint8)
    return scaled_image

def remove_cross(image):
    height, lenght = image.shape
    # hyperbola parameters
    center_x = lenght // 2
    center_y = height // 2
    radius = height*lenght/1500
    # create an array with coordinates x e y
    yy, xx = np.ogrid[:height, :lenght]
    # calculate the area of the cross
    cross_area = abs((xx - center_x) * (yy - center_y)) <= radius
    # set pixels to 0
    image[cross_area] = np.min(image)
    #circle_area = (xx - center_x)**2 + (yy - center_y)**2 <= radius*15
    #image[circle_area] = 0
    return image

def threshold_image(image, threshold=215):
    # Copia l'immagine originale per evitare modifiche indesiderate
    thresholded_image = np.copy(image)
    # Imposta a 0 tutti i pixel sotto il valore di soglia
    thresholded_image[thresholded_image < threshold] = 0
    return thresholded_image


current_directory = os.getcwd()

# Definisci la cartella di destinazione dei risultati
results_directory = os.path.join(current_directory, "risultati2")

# Se la cartella "risultati" non esiste, creala
if not os.path.exists(results_directory):
    os.makedirs(results_directory)




# Definire la cartella contenente le immagini
directory = "../../TestSet"

subfolders = [subfolder for subfolder in os.listdir(directory) if os.path.isdir(os.path.join(directory, subfolder))]
counter = 0
# Itera su ogni sottocartella
for subfolder in subfolders:
    counter+=1
    # Percorso completo della sottocartella
    subfolder_path = os.path.join(directory, subfolder)

    # Carica le immagini e applica la trasformata di Fourier a ciascuna sottocartella
    images = load_images_and_apply_fourier(subfolder_path)

    # Crea un'immagine media
    average_image = create_average_image(images)

    # Rimuovi le croci dall'immagine media
    image_crossless = remove_cross(average_image.copy())

    # Ridimensiona l'immagine
    scale_img = scale_image2(image_crossless)

    # Applica la soglia all'immagine
    threshold_img = threshold_image(scale_img, 130) #ris = 130

    # Visualizzare l'immagine media
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(average_image)
    plt.title('Average image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(scale_img)
    plt.title('Scaled image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(threshold_img)
    plt.title('Thresholded image')
    plt.axis('off')

    plt.tight_layout()
    #plt.savefig(os.path.join(results_directory, f"{subfolder}_magnitude.png"))
    print(f"------> {subfolder} completed ----- ({counter}/{len(subfolders)})")
    #plt.show()

    #plt.imshow(average_image)
    #plt.tight_layout()
    #plt.axis('off')
    #plt.savefig("name.png")

"""
counter = 0
for img in images:
    counter+=1
    img_crossless = remove_cross(img.copy())
    scale = scale_image(img_crossless)
    threshold = threshold_image(scale)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f'Immagine #{counter}')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(scale)
    plt.title('Immagine scalata')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(threshold)
    plt.title('Immagine sogliata')
    plt.axis('off')

    plt.show()
"""