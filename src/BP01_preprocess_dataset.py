import os
import shutil
from PIL import Image


def change_file_name(input_path, parent_folder, folder):
    """
    Rename all files in the folder to {parent_folder prefix}_{folder name}_{index}.PNG
    """
    folder_path = os.path.join(input_path, folder)
    files = os.listdir(folder_path)
    prefix = parent_folder.split('_')[0]  # Extract prefix from parent folder name

    for index, file in enumerate(files):
        new_name = f"{prefix}_{folder}_{index}.PNG"  # Create new file name
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_name))


def move_subfolders_to_parent(parent_path):
    """
    Move all files from subfolders to the parent folder and remove empty subfolders.
    """
    # List all subfolders in the parent directory
    for subfolder in os.listdir(parent_path):
        subfolder_path = os.path.join(parent_path, subfolder)

        # Check if the subfolder is a directory
        if os.path.isdir(subfolder_path):
            # List all items in the subfolder
            for img in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, img)
                destination_path = os.path.join(parent_path, img)

                # Move each item to the parent directory
                shutil.move(img_path, destination_path)
            # Remove the now-empty subfolder
            os.rmdir(subfolder_path)


def resize_image(input_path, folder, max_size=256):
    """
    Resize all images in the folder to have a maximum dimension of max_size, preserving aspect ratio.
    """
    folder_path = os.path.join(input_path, folder)
    # List all images in the folder
    for image_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, image_name)

        with Image.open(img_path) as img:
            # Get current size
            width, height = img.size

            # Determine new size maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(max_size * height / width)
            else:
                new_height = max_size
                new_width = int(max_size * width / height)

            # Resize the image
            img = img.resize((new_width, new_height), Image.LANCZOS)

            # Save the resized image
            img.save(img_path)


def copy_directory(source_path, destination_path):
    """
    Copy the directory from source_path to destination_path, overwriting if it exists.
    """
    if os.path.exists(destination_path):
        # Remove existing destination directory
        shutil.rmtree(destination_path)

    # Copy source directory to destination
    shutil.copytree(source_path, destination_path)


if __name__ == '__main__':
    # Define parent and input dataset paths
    parent_dir = '../data'
    input_dir = os.path.join(parent_dir, 'raw_images')

    # Step 1: Change file names
    output_dir = os.path.join(parent_dir, 'processed_images_V1')
    copy_directory(input_dir, output_dir)

    svp_types = os.listdir(output_dir)
    for svp_type in svp_types:  # Iterate over each type in raw_images
        svp_path = os.path.join(output_dir, svp_type)
        for protein_type in os.listdir(svp_path):  # Iterate over each subfolder
            change_file_name(svp_path, svp_type, protein_type)
    print(f'File names changed and saved to {output_dir}')

    # Step 2: Move subfolders' contents to parent folder
    prev_dir = output_dir
    output_dir = os.path.join(parent_dir, 'processed_images_V2')
    copy_directory(prev_dir, output_dir)

    for svp_type in svp_types:
        svp_path = os.path.join(output_dir, svp_type)
        move_subfolders_to_parent(os.path.join(output_dir, svp_type))
    print(f'Subfolders consolidated and saved to {output_dir}')

    # Step 3: Resize images
    prev_dir = output_dir
    output_dir = os.path.join(parent_dir, 'processed_images_V3')
    copy_directory(prev_dir, output_dir)

    for svp_type in svp_types:
        resize_image(output_dir, svp_type)
    print(f'Images resized and saved to {output_dir}')
