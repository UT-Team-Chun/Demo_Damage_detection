import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
from torchvision import transforms
import warnings # Import warnings module

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (string): Directory path with all the images (.png or .jpg).
            mask_dir (string): Directory path with all the masks (.png or .jpg).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Find both .png and .jpg files
        image_files_png = glob.glob(os.path.join(image_dir, '*.png'))
        image_files_jpg = glob.glob(os.path.join(image_dir, '*.jpg'))
        all_image_files = sorted(image_files_png + image_files_jpg)

        mask_files_png = glob.glob(os.path.join(mask_dir, '*.png'))
        mask_files_jpg = glob.glob(os.path.join(mask_dir, '*.jpg'))
        all_mask_files = sorted(mask_files_png + mask_files_jpg)

        # Create dictionaries mapping basename to full path
        image_path_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in all_image_files}
        mask_path_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in all_mask_files}

        # Find common basenames
        img_basenames = set(image_path_dict.keys())
        mask_basenames = set(mask_path_dict.keys())
        common_basenames = sorted(list(img_basenames.intersection(mask_basenames)))

        # Build the final lists using the dictionaries
        self.image_files = [image_path_dict[base] for base in common_basenames]
        self.mask_files = [mask_path_dict[base] for base in common_basenames]


        # Add check for empty dataset
        if len(self.image_files) == 0:
             warnings.warn(f"No matching image and mask files (.png or .jpg) found in {image_dir} and {mask_dir}. "
                           f"Please check paths, extensions, and filenames.", UserWarning)
             # Or raise an error:
             # raise FileNotFoundError(f"No matching image and mask files found in {image_dir} and {mask_dir}. "
             #                         f"Please check paths, extensions, and filenames.")


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Assume mask is a single-channel grayscale image

        # Convert mask to binary (0 or 1)
        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32) # Set pixels > 0 to 1, otherwise 0
        mask = Image.fromarray(mask)

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            # Apply the same geometric transforms to the mask, but not color transforms like normalization
            # Note: This transform needs to handle both image and mask, or be applied separately
            # For simplicity, we only apply basic ToTensor and Normalize to the image
            # Mask is only converted to Tensor
            mask_transform = transforms.Compose([
                transforms.Resize(sample['image'].shape[1:]), # Ensure mask and image sizes match
                transforms.ToTensor()
            ])
            sample['mask'] = mask_transform(sample['mask'])


        return sample['image'], sample['mask']

# Example transforms (adjust as needed)
def get_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Common ImageNet mean and std
    ])
