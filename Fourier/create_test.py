import os
import shutil

def copy_single_image(source_dir, dest_dir):
    # Iterate over subfolders in the source directory
    for subfolder in os.listdir(source_dir):
        # Construct full path of subfolder in source directory
        subfolder_path = os.path.join(source_dir, subfolder)
        
        # Check if the path is a directory
        if os.path.isdir(subfolder_path):
            # Construct full path of subfolder in destination directory
            dest_subfolder_path = os.path.join(dest_dir, subfolder)
            
            # Create the corresponding subfolder in the destination directory
            if not os.path.exists(dest_subfolder_path):
                os.makedirs(dest_subfolder_path)
            
            # Iterate over files in the subfolder
            for filename in os.listdir(subfolder_path):
                # Check if the file is an image (you may want to add more extensions)
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    # Construct full path of the image file
                    image_path = os.path.join(subfolder_path, filename)
                    # Copy the image file to the destination subfolder
                    shutil.copy(image_path, dest_subfolder_path)
                    # Only copy one image per subfolder, then break the loop
                    break

# Source directory containing subfolders with images
source_directory = "../../TestSet"

# Destination directory where the copied images will be stored
destination_directory = "test"

# Call the function to copy single images from each subfolder
copy_single_image(source_directory, destination_directory)
