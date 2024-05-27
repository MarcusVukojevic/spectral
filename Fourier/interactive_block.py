import pygame
import numpy as np
import sys
import matplotlib.pyplot as plt

# Inizializzazione di pygame
pygame.init()

# Dimensioni della finestra
width, height = 1200, 400

# Creazione della finestra
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Move the square with the arrows, +/- to change dimensions, WASD to change 1 dimension, QE to change mode, ZX to change color")

# Posizione iniziale del quadrato
square_y = height // 2
square_x = square_y
square_lx = 50
square_ly = 50
mode = True
color = 255

# Colore del quadrato
WHITE = (255, 255, 255)

# Funzione per calcolare la trasformata di Fourier lungo le righe dell'immagine
def fft_rows(image):
    fft_rows = np.fft.fft(image.astype(float), axis=0)
    fft_rows = np.abs(fft_rows)
    return fft_rows

# Funzione per calcolare la trasformata di Fourier lungo le colonne dell'immagine
def fft_columns(image):
    fft_columns = np.fft.fft(image.astype(float), axis=1)
    fft_columns = np.abs(fft_columns)
    return fft_columns

# Funzione per calcolare la trasformata di Fourier 2D dell'immagine
def fft2(image):
    fft_image = np.fft.fft2(image.astype(float))
    fft_image = np.abs(fft_image)
    return fft_image

def apply_fourier_transform(image):

    image_array = np.array(image)

    # Apply Fourier Transform
    f_transform = np.fft.fft2(image_array)
    f_shift = np.fft.fftshift(f_transform)
    
    # Get magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))

    return magnitude_spectrum
###np.fft.fftshift

# Funzione per creare l'immagine con il quadrato interattivo
def create_image_with_square(color):
    image = np.zeros((height, width), dtype=np.uint8)
    image[int(square_y-square_ly/2):int(square_y+square_ly/2), int(square_x-square_lx/2):int(square_x+square_lx/2)] = color
    return image

def plot_images(lista_immagini, lista_titoli):

    plt.figure(figsize=(12, 6))

    for i in range(len(lista_immagini)):
        plt.subplot(1, len(lista_immagini), i + 1)
        plt.imshow(lista_immagini[i], cmap='gray', vmin=0, vmax=255)
        plt.title(f'{lista_titoli[i]}')
        plt.xticks([]), plt.yticks([])
    plt.show()



running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                square_y -= 10
            elif event.key == pygame.K_RIGHT:
                square_y += 10
            elif event.key == pygame.K_UP:
                square_x -= 10
            elif event.key == pygame.K_DOWN:
                square_x += 10
            elif event.key == pygame.K_PLUS:
                square_ly += 10
                square_lx += 10
            elif event.key == pygame.K_MINUS:
                square_ly -= 10
                square_lx -= 10
            elif event.key == pygame.K_w:
                square_ly += 10
            elif event.key == pygame.K_s:
                square_ly -= 10
            elif event.key == pygame.K_d:
                square_lx += 10
            elif event.key == pygame.K_a:
                square_lx -= 10
            elif event.key == pygame.K_q:
                mode = True
            elif event.key == pygame.K_e:
                mode = False
            elif event.key == pygame.K_x:
                if(color<255):
                    color += 1
            elif event.key == pygame.K_z:
                if(color>0):
                    color -= 1

    image1 = create_image_with_square(color)
    image2 = fft_columns(image1)
    #image3 = np.fft.fftshift(fft_columns(image1),axes=0)
    if(mode):
        image3 = fft2(image1)
    else:
        image3 = apply_fourier_transform(image1)
    
    
    # Riempimento dello sfondo con il colore nero

    screen.fill((0, 0, 0))

    surface1 = pygame.surfarray.make_surface(image1)
    screen.blit(surface1, (0, 0))

    surface2 = pygame.surfarray.make_surface(image2)
    screen.blit(surface2, (width // 3, 0))

    surface3 = pygame.surfarray.make_surface(image3)
    screen.blit(surface3, (width // 3*2, 0))

    # Aggiorna la finestra
    pygame.display.flip()


pygame.quit()
sys.exit()
