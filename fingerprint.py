import numpy as np
from scipy import ndimage
from copy import deepcopy
import torch

def remove_cross(image):
    # hyperbola parameters
    height, lenght = image.shape
    center_x = lenght // 2
    center_y = height // 2
    radius = 400

    # create an array with coordinates x e y
    yy, xx = np.ogrid[:height, :lenght]
    # calculate the area of the cross
    cross_area = abs((xx - center_x) * (yy - center_y)) <= radius
    # set pixels to 0
    image[cross_area] = 0
    circle_area = (xx-center_x)**2 + (yy-center_y)**2 <= radius*10
    image[circle_area] = 0
    return image


def find_fingerprint(image,step):
    # parameters
    pixel_totali = image.size
    pixel_zero_prec = 0
    fingerprint = []
    # range in which search the fingerprint (percentage)
    range_min = 0.995
    range_max = 0.998

    for i in range(255):
        image[image < i ] = 0
        pixel_zero = np.count_nonzero(image == 0)
        if(pixel_zero>pixel_zero_prec+step):
            pixel_zero_prec = pixel_zero
            if((pixel_zero > pixel_totali*range_min) & (pixel_zero < pixel_totali*range_max)):
                fingerprint.append(image.copy())
    return fingerprint


def bandpass_filter(image):
    # Kernel per il filtro passa-basso (media)
    """
    low_threshold = 10
    high_threshold = 9
    lowpass_kernel = np.ones((low_threshold, low_threshold)) / (low_threshold * low_threshold)
    
    # Applicazione del filtro passa-basso
    lowpass_image = ndimage.convolve(image, lowpass_kernel)
    """
    # Kernel per il filtro passa-alto
    highpass_kernel = np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]])
    
    # Applicazione del filtro passa-alto all'immagine originale
    highpass_image = ndimage.convolve(image, highpass_kernel)
    
    # Combinazione dei risultati per ottenere l'effetto passa-banda
    #bandpass_image = lowpass_image + highpass_image
    
    return highpass_image


def normalization_fft_2(pic):
    im = np.float32(deepcopy(np.asarray(pic))) / 255.0

    # Gestione delle immagini in scala di grigi e a colori
    if len(im.shape) == 2:  # Immagine in scala di grigi
        channels = 1
    else:  # Immagine a colori
        channels = im.shape[2]

    for i in range(channels):
        if channels == 1:  # Per immagini in scala di grigi
            img = im
        else:  # Per immagini a colori
            img = im[:, :, i]

        fft_img = np.fft.fft2(img)
        fft_img = np.fft.fftshift(fft_img)
        fft_img = np.log(np.abs(fft_img) + 1e-3)
        fft_min = np.percentile(fft_img, 5)
        fft_max = np.percentile(fft_img, 95)
        if (fft_max - fft_min) > 0:
            fft_img = (fft_img - fft_min) / (fft_max - fft_min)
        else:
            #print('Intervallo di normalizzazione non valido.')
            fft_img = (fft_img - fft_min) / (np.finfo(float).eps)
        
        fft_img = (fft_img - 0.5) * 2
        fft_img[fft_img < -1] = -1
        fft_img[fft_img > 1] = 1
        
        if channels == 1:
            im = fft_img
        else:
            im[:, :, i] = fft_img

    return im


def normalization_fft(pic):

    im = np.float32(deepcopy(np.asarray(pic))) / 255.0

    for i in range(im.shape[2]):
        img = im[:, :, i]
        fft_img = np.fft.fft2(img)
        fft_img = np.log(np.abs(fft_img) + 1e-3)
        fft_min = np.percentile(fft_img, 5)
        fft_max = np.percentile(fft_img, 95)
        if (fft_max - fft_min) <= 0:
            print('ma cosa...')
            fft_img = (fft_img - fft_min) / ((fft_max - fft_min)+np.finfo(float).eps)
        else:
            fft_img = (fft_img - fft_min) / (fft_max - fft_min)
        fft_img = (fft_img - 0.5) * 2
        fft_img[fft_img < -1] = -1
        fft_img[fft_img > 1] = 1
        im[:, :, i] = fft_img

    return im

def normalization_residue3(pic, flag_tanh=False):

    x = np.float32(deepcopy(np.asarray(pic))) / 32
    wV = (-1 * x[1:-3, 2:-2, :] + 3 * x[2:-2, 2:-2, :] - 3 * x[3:-1, 2:-2, :] + 1 * x[4:, 2:-2, :])
    wH = (-1 * x[2:-2, 1:-3, :] + 3 * x[2:-2, 2:-2, :] - 3 * x[2:-2, 3:-1, :] + 1 * x[2:-2, 4:, :])
    ress = np.concatenate((wV, wH), -1)
    if flag_tanh:
        ress = np.tanh(ress)
    

    selected_channels = ress[:3, :, :] 
    ress = torch.from_numpy(selected_channels).permute(1, 2, 0).contiguous()

    return ress