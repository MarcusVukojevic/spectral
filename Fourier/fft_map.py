import pygame
import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image

# Inizializzazione di pygame
pygame.init()

# Dimensioni della finestra
width, height = 255*2, 255

# Creazione della finestra
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Move")

par = 0


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                if(par>0):
                    par-=1
            elif event.key == pygame.K_RIGHT:
                if(par<255):
                    par+=1
            elif event.key == pygame.K_UP:
                if(par<246):
                    par+=10
            elif event.key == pygame.K_DOWN:
                if(par>9):
                    par-=10

    image = Image.open("magnitude_spectrum_color.png").convert('L')
    image_array = np.array(image)

    image2 = image_array.copy()
    image2[image2 < par * 10] = 0
    #image1 = create_image_with_square(color)
    #image2 = fft_columns(image_array)

    # Riempimento dello sfondo con il colore nero

    screen.fill((0, 0, 0))

    surface1 = pygame.surfarray.make_surface(image_array)
    screen.blit(surface1, (0, 0))

    surface2 = pygame.surfarray.make_surface(image2)
    screen.blit(surface2, (width // 2, 0))

    # Aggiorna la finestra
    pygame.display.flip()


pygame.quit()
sys.exit()
