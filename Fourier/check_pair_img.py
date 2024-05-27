from PIL import Image

def check_images(img1_path, img2_path):
    # Loading images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Check dimensions
    if img1.size != img2.size:
        print("Different dimensions")
    else:
        # Check pixel by pixel
        lenght, height = img1.size
        check = True
        for y in range(height):
            for x in range(lenght):
                pixel_img1 = img1.getpixel((x, y))
                pixel_img2 = img2.getpixel((x, y))
                if pixel_img1 != pixel_img2:
                    print(f"The pixel at posizion ({x}, {y}) is different:")
                    print(f"Image 1: {pixel_img1}")
                    print(f"Image 2: {pixel_img2}")
                    check = False
        if(check):
            print("Same image (pixels)")

# Call
img1_path = "FFT_square_gray.png"
img2_path = "FFT_square_RGB.png"
check_images(img1_path, img2_path)
