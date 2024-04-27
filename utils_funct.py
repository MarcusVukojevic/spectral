import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# MI FA LA FFT SULLE RIGHE SENZA SHIFT
def fft_righe(immagine):
    fft2_rows = np.fft.fft(immagine, axis=1)
    return fft2_rows

# MI FA LA FFT SULLE COLONNE SENZA SHIFT
def fft_colonne(immagine):
    fft2_columns = np.fft.fft(immagine, axis=0)
    return fft2_columns

# MI FA LA FFT SULLE RIGHE CON SHIFT
def fft_righe_shift(righe):
    wela_shift = np.fft.fftshift(fft_righe(righe), axes=(1,))
    magnitude_spectrum = 20 * np.log(np.abs(wela_shift))
    for i in range(len(magnitude_spectrum)):
        for j in range(len(magnitude_spectrum[i])):
            if(magnitude_spectrum[i][j] < 0):
                magnitude_spectrum[i][j] = 0

    return magnitude_spectrum

# MI FA LA FFT SULLE COLONNE CON SHIFT
def fft_colonne_shift(righe):
    wela_shift = np.fft.fftshift(fft_colonne(righe), axes=(0,))
    magnitude_spectrum = 20 * np.log(np.abs(wela_shift))

    for i in range(len(magnitude_spectrum)):
        for j in range(len(magnitude_spectrum[i])):
            if(magnitude_spectrum[i][j] < 0):
                magnitude_spectrum[i][j] = 0
    return magnitude_spectrum

# PRENDO UN ARRAY E FACCIO LA FFT + SHIFT --> SOLO MAGN
def fft_da_array(img):
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    
    
    for i in range(len(magnitude_spectrum)):
        for j in range(len(magnitude_spectrum[i])):
            if(magnitude_spectrum[i][j] < 0):
                magnitude_spectrum[i][j] = 0

    return magnitude_spectrum


# PRENDO IL PATH DI UN'IMMAGINE E FACCIO LA FFT + SHIFT --> SOLO MAGN

def fft_da_immagine(img):
    image = Image.open(img).convert('L')
    image_array = np.array(image)

    f_transform = np.fft.fft2(image_array)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = (np.abs(f_shift))
    
    for i in range(len(magnitude_spectrum)):
        for j in range(len(magnitude_spectrum[i])):
            if(magnitude_spectrum[i][j] < 0):
                magnitude_spectrum[i][j] = 0
    return magnitude_spectrum


# PRENDO UN ARRAY E FACCIO LA FFT + SHIFT + MAGNITUDE + PHASE --> mi restituisce un'immagine rgb a colori con magnitude e fase combinati

def fft_da_array_colorized(img):
    f_transform = np.fft.fftshift(np.fft.fft2(img))
    magnitude = np.abs(f_transform)
    phase = np.angle(f_transform)

    hue = (phase + np.pi) / (2 * np.pi)  # Normalize phase to 0-1
    saturation = np.ones_like(hue)  # Full saturation
    value = magnitude / 255  # Use scaled magnitude for value
    hsv_image = np.stack([hue, saturation, value], axis=-1)
    rgb_image = colors.hsv_to_rgb(hsv_image)

    return rgb_image

# PRENDO UN'IMMAGINE E FACCIO LA FFT + SHIFT + MAGNITUDE + PHASE --> mi restituisce un'immagine rgb a colori con magnitude e fase combinati

def fft_da_array_colorized_da_immagine(img):
    image = Image.open(img).convert('L')
    image_array = np.array(image)
    
    f_transform = np.fft.fftshift(np.fft.fft2(image_array))
    magnitude = np.abs(f_transform)
    phase = np.angle(f_transform)

    hue = (phase + np.pi) / (2 * np.pi)  # Normalize phase to 0-1
    saturation = np.ones_like(hue)  # Full saturation
    value = magnitude / 255  # Use scaled magnitude for value
    hsv_image = np.stack([hue, saturation, value], axis=-1)
    rgb_image = colors.hsv_to_rgb(hsv_image)

    return rgb_image


# PLOTTO LE IMMAGINI DATE UNA LISTA DI IMMAGINI E TITOLI

def plot_images(lista_immagini, lista_titoli):

    plt.figure(figsize=(12, 6))
    
    for i in range(len(lista_immagini)):
        plt.subplot(1, len(lista_immagini), i + 1)
        plt.imshow(lista_immagini[i], cmap='gray', vmin=0, vmax=255)
        plt.title(f'{lista_titoli[i]}')
        plt.xticks([]), plt.yticks([])
        
    plt.show()


def scala_a_255(magnitude):
    magnitude -= magnitude.min()
    magnitude = magnitude / magnitude.max()
    magnitude *= 255

    return magnitude.astype(np.uint8)



# suca

def create_image_with_moving_block(size, block_size, move_distance):
    # Create an image with a black background
    image = np.zeros((size, size), dtype=np.uint8)

    # Calculate the end position of the block, ensuring it stays within the image boundaries
    end_x = min(size, block_size + move_distance)
    end_y = min(size, block_size + move_distance)

    # Place the white block
    image[move_distance:end_y, move_distance:end_x] = 255

    return image


# Function to apply Fourier Transform to an image
def apply_fourier_transform(image_path):
    # Load the image
    image = Image.open(image_path).convert('L')  # convert image to grayscale
    image_array = np.array(image)

    # Apply Fourier Transform
    f_transform = np.fft.fft2(image_array)
    f_shift = np.fft.fftshift(f_transform)

    # Get magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))

    return magnitude_spectrum
