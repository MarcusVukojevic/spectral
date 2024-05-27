from PIL import Image

# Loading image
#image_path = "biggan_000_302875.png"
image_path = "square.png"
image = Image.open(image_path)

# Print dimensions
lenght, height = image.size
print("Dimensioni:", lenght, "x", height)

# Print format
if image.mode == "RGB":
    print("Format: RGB")
elif image.mode == "L":
    print("Format: gray")
else:
    print("Format: ", image.mode)
