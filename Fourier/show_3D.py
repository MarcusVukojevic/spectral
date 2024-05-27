
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

image = plt.imread('magnitude_spectrum_color.png')

if len(image.shape) == 3:
    image = np.mean(image, axis=2)

height, width = image.shape

h_center = height // 2
w_center = width // 2

h_start = h_center - 10
h_end = h_center + 10
w_start = w_center - 10
w_end = w_center + 10

matrix = image[h_start:h_end, w_start:w_end]

x, y = np.meshgrid(np.arange(20), np.arange(20))

"""
# Genera una matrice di esempio 20x20
matrix = np.random.rand(20, 20) * 10  # Valori casuali tra 0 e 10

# Definisci le coordinate x e y per la matrice
x = np.arange(20)
y = np.arange(20)
x, y = np.meshgrid(x, y)
"""


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Rappresenta ogni valore della matrice come un parallelepipedo
for i in range(20):
    for j in range(20):
        dx = dy = 1  # Dimensioni della base del parallelepipedo
        dz = matrix[i, j]  # Altezza del parallelepipedo
        if j == 17 and i == 3:
            ax.bar3d(i, j, 0, dx, dy, dz+0.1, color='g', zsort='average')
        elif j == 17 or i == 3:
            ax.bar3d(i, j, 0, dx, dy, dz, color='r', zsort='average')
        else:
            ax.bar3d(i, j, 0, dx, dy, dz, color='b', zsort='average')

ax.set_xlabel('Rows')
ax.set_ylabel('Columns')
ax.set_zlabel('Value')
plt.title('FFT')
plt.show()


""" # 3D Fourier
import numpy as np
import matplotlib.pyplot as plt

# Carica l'immagine in scala di grigi
image = plt.imread('magnitude_spectrum_color.png')

if len(image.shape) == 3:
    image = np.mean(image, axis=2)
print("image.shape= ",image.shape)
# Ottenere le dimensioni dell'immagine

height, width = image.shape


h_center = height // 2
w_center = width // 2

# Definisci i limiti della regione rettangolare
h_start = h_center - 128
h_end = h_center + 128
w_start = w_center - 128
w_end = w_center + 128

# Estrai il quadrato centrale dall'immagine
image = image[h_start:h_end, w_start:w_end]

X, Y = np.meshgrid(np.arange(256), np.arange(256))
# Crea un array per rappresentare l'altezza lungo l'asse z

# Visualizzazione in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, image, cmap='gray')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Altezza')

ax.view_init(elev=20, azim=45)

plt.show()
"""