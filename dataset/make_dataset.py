from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from torch.utils.data import random_split
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset.augmentation import random_augmentation
from dataset.make_noise import *


class SourceDataset(data.Dataset):
    def __init__(self, csv_path='/Users/Han/Desktop/capstone/JaPyGuri-AI/dataset/source_labeled/file_path.csv'):
        super().__init__()
        path = pd.read_csv(csv_path)

        data_path_arr = path['Image'].values.tolist()
        label_arr = path['Label'].tolist()

        self.x_path = data_path_arr
        self.y_path = label_arr

        self.transform = transforms.Compose([
            transforms.ToTensor(),
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
    def __init__(self, csv_path='/Users/Han/Desktop/capstone/JaPyGuri-AI/dataset/target_unlabeled/HandInfo.csv', patch_flag=True):
        super().__init__()
        dataFrame = pd.read_csv(csv_path)
        nail_data = dataFrame[dataFrame['aspectOfHand'].isin(['dorsal right', 'dorsal left'])]
        self.file_name = nail_data['imageName'].values.tolist()

        dir_name = "/Users/Han/Desktop/capstone/JaPyGuri-AI/dataset/target_unlabeled/Hands/Hands/"
        self.file_paths = [os.path.join(dir_name, file_name) for file_name in self.file_name]
        self.patch_flag = patch_flag

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(300),
            # transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        self.x_data = Image.open(self.file_paths[index])
        self.x_data = self.transform(self.x_data)

        # add noise

        if self.patch_flag is True:
            noisy_images = make_noise(self.x_data, noise_factor=0.4, patch_ratio=0.05, patch_cnt=20)
        else:
            noisy_images = add_salt_pepper_noise(self.x_data, noise_factor=0.4)

        noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
        self.x_data = torch.clamp(self.x_data, 0.0, 1.0)

        return noisy_images.float(), self.x_data.float()
    def __len__(self):
        return len(self.file_paths)



def make_source_loader(batch_size=1):
    dataset = SourceDataset()
    total_size = len(dataset)
    train_size = int(0.8 * total_size) + 1
    val_size = total_size - train_size

    # Randomly split dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataset = data.Subset(dataset, range(0, train_size))
    val_dataset = data.Subset(dataset, range(train_size, total_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def make_target_loader(batch_size=8, patch_flag=True):
    dataset = TargetDataset(patch_flag=patch_flag)
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    # Randomly split dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataset = data.Subset(dataset, range(0, train_size))
    val_dataset = data.Subset(dataset, range(train_size, total_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



def visualize_segmentation_data_matplotlib(loader):
    num_images = len(loader)
    """
    DataLoader의 세그멘테이션 데이터를 matplotlib을 사용하여 시각화합니다.
    이미지, 마스크, 그리고 오버레이를 나란히 표시합니다.
    """
    images_shown = 0
    for batch_idx, (images, masks) in enumerate(loader):
        for i in range(images.size(0)):
            if images_shown >= num_images:
                return

            # 이미지와 마스크를 NumPy 배열로 변환
            image_np = images[i].squeeze().permute(1, 2, 0).numpy()
            mask_np = masks[i].squeeze().numpy()

            print(f'image max : {torch.max(images)}, min : {torch.min(images)}')
            print(f'mask max : {torch.max(masks)}, min : {torch.min(masks)}')

            # 오버레이를 위해 컬러맵 적용
            mask_colored = np.stack([mask_np, mask_np, mask_np], axis=-1)  # 흑백 마스크를 3채널로 변환
            overlay = np.clip(image_np * 0.6 + mask_colored * 0.4, 0, 1)  # 이미지와 마스크를 오버레이

            # 시각화
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
    print("make_dataset.py")
    target_dataset = TargetDataset(patch_flag=False)
    print(len(target_dataset))

    train, val = make_target_loader(batch_size=16,patch_flag=True)
    print(len(train))
    print(len(val))
    from make_noise import visualize_images
    for noisy_batch, image_batch in val:
        print(f'noisy(input) - shape : {noisy_batch.shape}, max : {torch.max(noisy_batch)}, min : {torch.min(noisy_batch)}')
        print(f'origin(target) - shape : {image_batch.shape}, max : {torch.max(image_batch)}, min : {torch.min(image_batch)}')
        for noisy_img, img in zip(noisy_batch, image_batch):
            visualize_images(img, noisy_img, figsize=(10, 5))
        break
