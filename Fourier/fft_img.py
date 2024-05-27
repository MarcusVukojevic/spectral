import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

# Call
#image_path = "Sudoku.jpg"
image_path = "biggan_000_302875.png"
#image_path = "square.png"
fft = apply_fourier_transform(image_path)

plt.imshow(fft)
plt.axis('off')  # Remove axis
plt.title('FFT')
plt.show()
#plt.imshow(image, cmap='gray')

def remove_cross(image):
    height, lenght = image.shape

    # hyperbola parameters
    center_x = lenght // 2
    center_y = height // 2
    radius = 400

    # create an array with coordinates x e y
    yy, xx = np.ogrid[:height, :lenght]
    # calculate the area of the cross
    cross_area = abs((xx - center_x) * (yy - center_y)) <= radius
    # set pixels to 0
    image[cross_area] = 0
    return image

def find_fingerprint(image,step):
    pixel_totali = image.size
    pixel_zero_prec = 0
    fingerprint = []
    for i in range(255):
        image[image < i ] = 0
        pixel_zero = np.count_nonzero(image == 0)
        if(pixel_zero>pixel_zero_prec+step):
            pixel_zero_prec = pixel_zero
            if((pixel_zero > pixel_totali*0.995) & (pixel_zero < pixel_totali*0.998)):
                fingerprint.append(image.copy())
    return fingerprint




# Copia dell'immagine originale
modified_image = np.copy(fft)
modified_image = remove_cross(modified_image)
fingerprints = find_fingerprint(modified_image,50)

for img in fingerprints:
    plt.imshow(img)       
    plt.axis('off')
    plt.show()
    
"""
pixel_zero_prec = 0
step = 50
for i in range(255):
    # Impostazione a 0 dei pixel sotto il valore del contatore * 10
    modified_image[modified_image < i ] = 0
    pixel_zero = np.count_nonzero(modified_image == 0)
    if(pixel_zero<=pixel_zero_prec+step):
        continue
    pixel_zero_prec=pixel_zero
    pixel_totali = modified_image.size
    if(pixel_zero > pixel_totali*0.998):
        break
    if(pixel_zero > pixel_totali*0.995):
        print("Numero di pixel a 0:", pixel_zero)
        print("Numero di pixel totali:", pixel_totali)
    # Visualizzazione dell'immagine

        plt.imshow(modified_image)
        plt.title(f'Valore del contatore: {i}')
        plt.axis('off')
        plt.savefig(f"z_image_{i}.png")
        plt.show()
"""
