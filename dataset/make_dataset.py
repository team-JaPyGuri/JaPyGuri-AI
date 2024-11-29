import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from dataset.augmentation import random_augmentation
from dataset.make_noise import *

class SourceDataset(data.Dataset):
    def __init__(self, csv_path='/mnt/hgh/JaPyGuri-AI/dataset/source_labeled/file_path.csv'):
        super().__init__()
        path = pd.read_csv(csv_path)

        data_path_arr = path['Image'].values.tolist()
        label_arr = path['Label'].tolist()

        self.x_path = data_path_arr
        self.y_path = label_arr

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1024, 1024)),
            # transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        self.x_data = Image.open(self.x_path[index])
        self.y_data = Image.open(self.y_path[index])
        self.y_data = self.y_data.convert('L')

        # 데이터 정규화
        self.x_data = self.transform(self.x_data)
        self.y_data = self.transform(self.y_data)

        self.x_data, self.y_data = random_augmentation(self.x_data, self.y_data)

        return self.x_data.float(), self.y_data.float()

    def __len__(self):
        return len(self.y_path)


class TargetDataset(data.Dataset):
    def __init__(self, csv_path='/mnt/hgh/JaPyGuri-AI/dataset/target_unlabeled/HandInfo.csv', noise_factor=0.25, patch_ratio=0.05, patch_cnt=20, patch_flag=True):
        super().__init__()
        dataFrame = pd.read_csv(csv_path)
        nail_data = dataFrame[dataFrame['aspectOfHand'].isin(['dorsal right', 'dorsal left'])]
        self.file_name = nail_data['imageName'].values.tolist()

        # Update directory paths based on environment
        dir_name = "/mnt/hgh/JaPyGuri-AI/dataset/target_unlabeled/Hands/Hands/"  # Server path
        # dir_name = "/Users/Han/Desktop/capstone/JaPyGuri-AI/dataset/target_unlabeled/Hands/Hands/"  # Local path
        self.file_paths = [os.path.join(dir_name, file_name) for file_name in self.file_name]

        self.noise_factor = noise_factor
        self.patch_ratio = patch_ratio
        self.patch_cnt = patch_cnt
        self.patch_flag = patch_flag

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1024, 1024)),
            # transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        self.x_data = Image.open(self.file_paths[index])
        self.x_data = self.transform(self.x_data)

        # Add noise
        if self.patch_flag:
            noisy_images = make_noise(self.x_data, self.noise_factor, self.patch_ratio, self.patch_cnt)
        else:
            noisy_images = add_salt_pepper_noise(self.x_data, noise_factor=self.noise_factor)

        noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
        self.x_data = torch.clamp(self.x_data, 0.0, 1.0)

        return noisy_images.float(), self.x_data.float()

    def __len__(self):
        return len(self.file_paths)


def make_source_loader(batch_size=1):
    dataset = SourceDataset()
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    # Split dataset into training and validation sets
    train_dataset = data.Subset(dataset, range(0, train_size))
    val_dataset = data.Subset(dataset, range(train_size, total_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def make_source_testloader(batch_size=1):
    dataset = SourceDataset()
    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return testloader


def make_target_loader(batch_size=8, noise_factor=0.25, patch_ratio=0.05, patch_cnt=20, patch_flag=True):
    dataset = TargetDataset(noise_factor=noise_factor, patch_ratio=patch_ratio, patch_cnt=patch_cnt, patch_flag=patch_flag)
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    # Randomly split dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def visualize_segmentation_data_matplotlib(loader):
    num_images = len(loader)
    """
    Visualize segmentation data using matplotlib: images, masks, and overlays side by side.
    """
    images_shown = 0
    for batch_idx, (images, masks) in enumerate(loader):
        for i in range(images.size(0)):
            if images_shown >= num_images:
                return

            # Convert image and mask to NumPy arrays
            image_np = images[i].squeeze().permute(1, 2, 0).numpy()
            mask_np = masks[i].squeeze().numpy()

            # Overlay the mask onto the image
            mask_colored = np.stack([mask_np, mask_np, mask_np], axis=-1)  # Convert grayscale mask to 3 channels
            overlay = np.clip(image_np * 0.6 + mask_colored * 0.4, 0, 1)

            # Plot images
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(image_np)
            axs[0].set_title("Image")
            axs[0].axis('off')

            axs[1].imshow(mask_np, cmap='jet', alpha=0.6)
            axs[1].set_title("Mask")
            axs[1].axis('off')

            axs[2].imshow(overlay)
            axs[2].set_title("Overlay")
            axs[2].axis('off')

            plt.show()
            images_shown += 1


if __name__ == "__main__":
    print("Starting dataset creation and visualization...")

    # Test the target loader with specific noise settings
    train, val = make_target_loader(batch_size=8, noise_factor=0.25, patch_ratio=0.05, patch_cnt=20)
    print(f'Train loader size: {len(train)}')
    print(f'Validation loader size: {len(val)}')

    # Visualize images in the validation set
    for noisy_batch, image_batch in val:
        print(f'Noisy (input) - shape: {noisy_batch.shape}, max: {torch.max(noisy_batch)}, min: {torch.min(noisy_batch)}')
        print(f'Original (target) - shape: {image_batch.shape}, max: {torch.max(image_batch)}, min: {torch.min(image_batch)}')

        for noisy_img, img in zip(noisy_batch, image_batch):
            visualize_images(img, noisy_img, figsize=(10, 5))
        break  # Only visualize the first batch
