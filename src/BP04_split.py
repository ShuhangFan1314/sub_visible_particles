import os
import shutil
import numpy as np


def split_and_copy_files(input_path, cls, output_path, train_ratio=0.8):
    """
    Split files into training and testing sets, and copy them to the respective directories.
    """
    src_dir = os.path.join(input_path, cls)
    all_file_names = os.listdir(src_dir)

    # Separate files that should only be in the test dataset
    test_specific_files = [f for f in all_file_names if f.startswith('mech_pembrolizumab') or f.startswith('heat_trastuzumab')]
    other_files = [f for f in all_file_names if f not in test_specific_files]

    # Shuffle and split the remaining files
    np.random.shuffle(other_files)
    num_train = int(train_ratio * len(all_file_names))
    train_files = np.array(other_files[:num_train])
    test_files = np.array(other_files[num_train:] + test_specific_files)  # Add the specific files to the test set

    # Prepare destination directories
    train_ds_dir = os.path.join(output_path, 'train_ds', cls)
    test_ds_dir = os.path.join(output_path, 'test_ds', cls)
    os.makedirs(train_ds_dir, exist_ok=True)
    os.makedirs(test_ds_dir, exist_ok=True)

    # Copy files to their respective directories
    copy_files(train_files, src_dir, train_ds_dir)
    copy_files(test_files, src_dir, test_ds_dir)

    # Print summary of the operation
    print_summary(cls, all_file_names, train_files, test_files)


def copy_files(file_names, source_dir, destination_dir):
    """
    Copy a list of files from the source directory to the destination directory.
    """
    for name in file_names.tolist():
        shutil.copy(os.path.join(source_dir, name), destination_dir)


def print_summary(cls, all_file_names, train_files, test_files):
    """
    Print a summary of the number of files in training and testing sets.
    """
    print(f"\n *****************************",
          f"\n Total images: {cls} {len(all_file_names)}",
          f'\n Training: {len(train_files)}',
          f'\n Testing: {len(test_files)}',
          f'\n *****************************')


if __name__ == '__main__':
    input_dir = '../data/processed_images_V4/'
    output_dir = '../data/processed_images_V5'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    classes_dir = os.listdir(input_dir)  # List all classes in the root directory
    for cls in classes_dir:
        split_and_copy_files(input_dir, cls, output_dir)

    print('All images are split and saved to', output_dir)
