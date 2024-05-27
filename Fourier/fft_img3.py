import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def find_max_coordinates(image, x, y, window_size):

    half_window = window_size // 2
    max_val = None
    max_coordinates = []

    for i in range(x - half_window, x + half_window + 1):
        for j in range(y - half_window, y + half_window + 1):
            if 0 <= i < len(image) and 0 <= j < len(image[0]):
                if max_val is None or image[i][j] > max_val:
                    max_val = image[i][j]
                    max_coordinates = [(i, j, max_val)]
                elif image[i][j] == max_val:
                    max_coordinates.append((i, j, max_val))
    return max_coordinates

def is_within_window(point1, point2, window_size):
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) <= window_size and abs(y1 - y2) <= window_size

def update_output_list(new_point, output_list, window_size):
    x_new, y_new, val_new = new_point
    
    to_add = True
    
    to_remove = []
    
    for i, point in enumerate(output_list):
        x, y, val = point
        
        if is_within_window((x_new, y_new), (x, y), window_size):
            if val > val_new:
                to_add = False
                break
            elif val < val_new:
                to_remove.append(i)
    
    # Rimuovi i punti che sono stati superati dal nuovo punto
    for index in reversed(to_remove):
        del output_list[index]
    
    # Se il flag è True, aggiungi il nuovo punto alla lista di output
    if to_add:
        output_list.append(new_point)



def keep_max_local_points_only(image, max_points):
    # Converti la lista di max_points in un set di tuple (x, y)
    max_points_set = {(point[0], point[1]) for point in max_points}

    rows, cols = image.shape
    result_image = np.zeros_like(image)
    # Aggiungi i vicini dei punti di massimo locale al set dei punti di massimo locale
    neighbour_points = set()
    for point in max_points_set:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                new_row, new_col = point[0] + dr, point[1] + dc
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    neighbour_points.add((new_row, new_col))
    max_points_set.update(neighbour_points)
    # Imposta a 0 i pixel che non sono presenti nei punti di massimo locale o nei loro vicini
    for row in range(rows):
        for col in range(cols):
            if (row, col) not in max_points_set:
                result_image[row, col] = 0
            else:
                result_image[row, col] = 255 #image[row, col]*2
    return result_image


#-----------------------------------------------------------------------

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

def find_max(scale_img,window_size):
    max_points = []
    rows, cols = scale_img.shape
    i = int((window_size-1)/2)
    while(i<rows+window_size):
        j = int((window_size-1)/2)
        while(j<cols+window_size):
            max_points_tmp = find_max_coordinates(scale_img,i,j,window_size)
            for new_point in max_points_tmp:
                update_output_list(new_point, max_points, window_size)
            j+=window_size
        i+=window_size
    return max_points
#-------------------------------------------------------------

current_directory = os.getcwd()

# Definisci la cartella di destinazione dei risultati
results_directory = os.path.join(current_directory, "massimi_x_modello")

# Se la cartella non esiste, creala
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

#directory = "../../TestSet"
directory = "dalle_mini_tmp"
#directory = "test"
window_size = 7

formatted_subfolders = []
subfolders = [subfolder for subfolder in os.listdir(directory) if os.path.isdir(os.path.join(directory, subfolder))]
counter = 0
# Itera su ogni sottocartella
for subfolder in subfolders:
    counter+=1
    #if(counter!=1):
    #    continue
    #print(f"Processing {counter}: {subfolder}")


    # Percorso completo della sottocartella
    subfolder_path = os.path.join(directory, subfolder)

    # Carica le immagini e applica la trasformata di Fourier a ciascuna sottocartella
    images = load_images_and_apply_fourier(subfolder_path)

    # Crea un'immagine media
    average_image = create_average_image(images)

    # Rimuovi le croci dall'immagine media
    #image_crossless = remove_cross(average_image.copy())

    # Ridimensiona l'immagine
    scale_img = scale_image(average_image)

    # Applica la soglia all'immagine
    #threshold_img = threshold_image(scale_img, 130)
    max_points = find_max(scale_img,window_size)

    result_image = keep_max_local_points_only(np.copy(scale_img), max_points)
    #rescale_image = scale_image(result_image)

"""
### generate max_points for models and test per window's lenght

    # Visualizza l'immagine con la mappa dei colori specificata
    plt.imshow(result_image) #, cmap=plt.cm.gray 
    plt.axis('off')
    plt.tight_layout()
    #plt.savefig(os.path.join(results_directory, f"{subfolder}_max_point_w{window_size}.png"))
    #print(f"------> {subfolder} completed ----- ({counter}/{len(subfolders)})")
    #plt.show()

    formatted_subfolder = f"{subfolder} = {max_points}\n"
    formatted_subfolders.append(formatted_subfolder)

# Scrivi tutte le stringhe formattate nel file di testo
output_file = os.path.join(os.getcwd(), f"max_points_images_w{window_size}.txt")
with open(output_file, 'w') as file:
    file.writelines(formatted_subfolders)
"""