import os
import shutil
import numpy as np
from PIL import Image


def add_padding(input_path, output_path):
    """
    Add padding to images in the dataset to make them square, using the mean color of the image as the padding color.
    """
    # Create new directory for dataset with padding
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # List all sub-visible particle (svp) types in the dataset path
    svp_types = os.listdir(input_path)
    for svp in svp_types:
        file_path = os.path.join(input_path, svp)
        new_svp_dir = os.path.join(output_path, svp)
        os.makedirs(new_svp_dir)

        # List all images in the current svp type
        images = os.listdir(file_path)
        for image in images:
            image_path = os.path.join(file_path, image)
            im = Image.open(image_path)

            # Calculate mean of RGB channels and add as padding
            im_array = np.array(im)
            avg_r = int(np.mean(im_array[:, :, 0]))  # Red channel
            avg_g = int(np.mean(im_array[:, :, 1]))  # Green channel
            avg_b = int(np.mean(im_array[:, :, 2]))  # Blue channel

            width, height = len(im_array[0]), len(im_array)

            # Create new image with padding
            new_image = Image.new("RGB",
                                  (max(width, height), max(width, height)),
                                  (avg_r, avg_g, avg_b))
            # Paste the original image in the center of the new image
            if width > height:
                new_image.paste(im, (0, (width-height) // 2))
            else:
                new_image.paste(im, ((height-width) // 2, 0))

            # Save the new padded image
            save_location = os.path.join(output_path, svp, image)
            new_image.save(save_location)

        print('Finished with', svp)


if __name__ == '__main__':
    input_dir = "../data/processed_images_V3"
    output_dir = "../data/processed_images_V4"
    add_padding(input_dir, output_dir)
    print('All images have been padded and saved to', output_dir)
