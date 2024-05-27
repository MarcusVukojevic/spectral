from PIL import Image, ImageDraw

def create_square(dim_x, dim_y, suqare_side, tipo, file_name):
    # Create black image
    img = Image.new('RGB', (dim_x, dim_y), color='black')

    # Compute square coordinate
    left = (dim_x - suqare_side) // 2
    top = (dim_y - suqare_side) // 2
    right = (dim_x + suqare_side) // 2
    bottom = (dim_y + suqare_side) // 2

    # Draw a white square at the center
    draw = ImageDraw.Draw(img)
    draw.rectangle((left, top, right, bottom), fill='white')

    # Convert the image into gray scale if specified
    if tipo == 'grayscale':
        img = img.convert('L')

    # Save image with nome specified
    img.save(file_name)

# Call
create_square(256,256,256/3,"RGB","square.png")