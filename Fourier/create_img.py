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

def create_sin(dim_x, dim_y, value, offset, file_name):
    # Create black image
    img = Image.new('RGB', (dim_x, dim_y), color='black')

    img = img.convert('L')

    # Get the drawing context
    draw = ImageDraw.Draw(img)

    # Loop through each pixel
    for y in range(dim_y):
        for x in range(dim_x):
            # Esempio di operazione su ogni pixel: disegna una linea bianca diagonale
            if(y==0):
                if(x%2==0):
                    draw.point((x, y), fill=offset+50)
                else:
                    draw.point((x, y), fill=(x%4-1)*value+50)
            else:
                if(x%2==0):
                    draw.point((x, y), fill=offset)
                else:
                    draw.point((x, y), fill=(x%4-1)*value)
    # Save image
    img.save(file_name)


def create_sin2(dim_x, dim_y, value, offset, file_name):
    # Create black image
    img = Image.new('RGB', (dim_x, dim_y), color='black')

    img = img.convert('L')

    # Get the drawing context
    draw = ImageDraw.Draw(img)

    # Loop through each pixel
    for y in range(dim_y):
        for x in range(dim_x):
            # Esempio di operazione su ogni pixel: disegna una linea bianca diagonale
            if(x%3==0):
                draw.point((x, y), fill=offset)
            elif(x%6==1 or x%6==2):
                draw.point((x, y), fill=value+offset)
            else:
                draw.point((x, y), fill=offset-value)

    # Save image
    img.save(file_name)

# Call
#create_square(256,256,256/3,"RGB","square.png")
create_sin(256,256,100,100,"sin.png")
#create_sin2(256,256,100,100,"sin2.png")