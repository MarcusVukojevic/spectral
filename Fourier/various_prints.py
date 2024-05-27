import numpy as np
import matplotlib.pyplot as plt

if(0):
    # Definizione della matrice
    matrice = np.array([
        [10, 0, -10, 0, 10, 0, -10, 0],
        [5, 0, -5, 0, 5, 0, -5, 0],
        [10, 0, -10, 0, 10, 0, -10, 0],
        [5, 0, -5, 0, 5, 0, -5, 0],
        [10, 0, -10, 0, 10, 0, -10, 0],
        [5, 0, -5, 0, 5, 0, -5, 0],
        [10, 0, -10, 0, 10, 0, -10, 0],
        [5, 0, -5, 0, 5, 0, -5, 0]
    ])


    # Calcolo della FFT sulle righe
    fft_righe = np.fft.fft(matrice, axis=1)

    # Calcolo della FFT sulle colonne
    fft_colonne = np.fft.fft(matrice, axis=0)

    # Stampa della FFT sulle righe come matrice
    print("FFT on rows:")
    print(np.real(fft_righe))

    # Stampa della FFT sulle colonne come matrice
    print("\nFFT on columns:")
    print(np.real(fft_colonne))



###disegnare seni e coseni

### Definire i parametri dell'onda sinusoidale
frequenza = 0.5  # Frequenza in Hz
frequenza2 = 3/5
frequenza3 = 7/5
ampiezza = 10  # Ampiezza dell'onda
ampiezza2 = 5
ampiezza3 = 10/2**(1/2)

### Generare i valori x (tempo) da 0 a 2 secondi con un passo di 0.001 secondi
tempo = np.arange(0, 8, 0.001)

### Calcolare i valori y (ampiezza dell'onda sinusoidale) utilizzando la funzione sinusoidale
#ampiezza_onda = ampiezza * np.cos(2 * np.pi * frequenza * tempo + np.pi/3)
ampiezza_onda2 = ampiezza2 * np.cos(2 * np.pi * frequenza * tempo)
#ampiezza_onda3 = ampiezza3 * np.cos(2 * np.pi * frequenza * tempo - np.pi/4)
#ampiezza_onda2 = ampiezza * np.cos(2*np.pi*tempo*frequenza2)
#ampiezza_onda3 = ampiezza * np.cos(2*np.pi*tempo*frequenza3)
### Plot dell'onda sinusoidale
#plt.plot(tempo, ampiezza_onda)
plt.plot(tempo, ampiezza_onda2)
#plt.plot(tempo, ampiezza_onda3)
plt.title('Signals with frequencies within and outside the range')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

quit()
#np.random.randint(1, 11)
numero_campionamenti = 20
periodo_pixel = 20/3 #20/k

# Calcola il numero di pixel per un periodo
periodo = 2 * np.pi / periodo_pixel

# Inizializza un array vuoto per i campionamenti
y = np.empty(numero_campionamenti)

# Genera i campionamenti utilizzando un ciclo for
for i in range(numero_campionamenti):
    x = i * periodo #+ np.pi/2
    y[i] = np.cos(x)#np.random.randint(1, 11)

#print(y)

f_righe = np.fft.fft(y, axis=0)
#print(f_righe)
#for i in range(11, 20):
#    f_righe[i]=0
#f_righe[0] = f_righe[0]/2
#f_righe[10] = f_righe[10]/2
y2 = np.fft.ifft(f_righe, axis=0)

epsilon = 1e-10  # Piccola costante per evitare il logaritmo di zero
magnitude_spectrum = 20 * np.log(np.abs(f_righe) + epsilon)
#magnitude_spectrum = np.clip(magnitude_spectrum, a_min=0, a_max=None)
#print(magnitude_spectrum)
print("y\t\t\t\tf_riga\t\t\t\t\ty2")
for i in range(numero_campionamenti):
    if(i==0):
        print("------------------constant----------------------------------------------------------")
    print("{:.6f}\t\t\t{:.6f}\t\t\t{:.6f}".format(y[i], f_righe[i], y2[i]))
    if(i==0):
        print("------------------positive----------------------------------------------------------")
    if((numero_campionamenti%2==0)and(i==numero_campionamenti/2-1)):
        print("------------------Nyquist-----------------------------------------------------------")
    if(i==numero_campionamenti/2 or i==(numero_campionamenti-1)/2):
        print("------------------negative----------------------------------------------------------")

#quit()
fft = np.empty((numero_campionamenti,numero_campionamenti*2), dtype=complex)
for j in range(1,numero_campionamenti*2+1):       
    periodo_pixel = numero_campionamenti/j #20/j
    periodo = 2 * np.pi / periodo_pixel #periodo campionamento

    y = np.empty(numero_campionamenti)
    for i in range(numero_campionamenti):
        x = i * periodo + np.pi/2
        y[i] = np.sin(x)

    f_righe = np.fft.fft(y, axis=0)
    fft[:, j - 1] = f_righe



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Rappresenta ogni valore della matrice come un parallelepipedo
for i in range(numero_campionamenti):
    for j in range(numero_campionamenti*2):
        dx = dy = 1  # Dimensioni della base del parallelepipedo
        dz = fft[i, j]  # Altezza del parallelepipedo
        if i == 0:
            ax.bar3d(i, j, 0, dx, dy, dz+0.1, color='g', zsort='average')
        elif i == 10:
            ax.bar3d(i, j, 0, dx, dy, dz+0.1, color='y', zsort='average')
        elif i > 10:
            ax.bar3d(i, j, 0, dx, dy, dz, color='b', zsort='average')
        else:
            ax.bar3d(i, j, 0, dx, dy, dz, color='r', zsort='average')

ax.set_xlabel('Discrete Frequencies')
ax.set_ylabel('K (signal\'s period = N/K)')
ax.set_zlabel('Value')
plt.title('FFT')
plt.show()

"""
first_column = fft[:, numero_campionamenti*2]
first_column[0] = 5

mid_index = len(first_column) // 2

# Crea il plot con colori diversi per ciascun gruppo
for i, value in enumerate(first_column):
    if i == 0:
        plt.bar(i, value, color='g')
    elif 1 <= i < mid_index:
        plt.bar(i, value, color='r')
    elif i == mid_index:
        plt.bar(i, value, color='y')
    else:
        plt.bar(i, value, color='b')

plt.xlabel('Row Index')
plt.ylabel('Value')
plt.title('FFT row')
plt.legend()
plt.show()
"""
