import os
from PIL import Image
from torch.utils.data.dataset import Dataset


class SubParDataset(Dataset):
    def __init__(self, dataset_path_label, aug, is_grey=False):
        # List that will contain all image paths and labels
        self.all_images_labels = []

        # Add [image_path, label] to self.all_images_labels
        for path, label in dataset_path_label:
            image_files = os.listdir(path)
            for im in image_files:
                full_im_path = os.path.join(path, im)
                self.all_images_labels.append([full_im_path, label])

        # Check if greyscale image
        self.is_grey = is_grey
        self.aug = aug

        print('Dataset with', len(self.all_images_labels), 'samples')

    def __getitem__(self, index):
        im_path, label = self.all_images_labels[index]
        im_name = im_path.split('/')[-1]
        stress_name = im_name.split('_')[0]
        drug_name = im_name.split('_')[1]

        # Convert image type to RGB
        im = Image.open(im_path).convert('RGB')
        # Convert image to greyscale
        if self.is_grey:
            im = Image.open(im_path).convert('L').convert('RGB')

        im = self.aug(im)

        return im, int(label), stress_name, drug_name

    def __len__(self):
        return len(self.all_images_labels)
