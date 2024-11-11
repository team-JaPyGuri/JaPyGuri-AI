from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from dataset.augmentation import random_augmentation
import torch
import numpy as np
import torch.nn.functional as F


class SourceDataset(data.Dataset):
    def __init__(self, csv_path):
        super().__init__()
        path = pd.read_csv(csv_path)

        data_path_arr = path['Image'].values.tolist()
        label_arr = path['Label'].tolist()

        self.x_path = data_path_arr
        self.y_path = label_arr


        self.transform = transforms.Compose([
            transforms.ToTensor(),

        ])

    def __getitem__(self, index):
        self.x_data = Image.open(self.x_path[index])
        self.y_data = Image.open(self.y_path[index])
        self.y_data = self.y_data.convert('L')

        # 데이터 정규화
        self.x_data = self.transform(self.x_data)
        self.y_data = self.transform(self.y_data)
        _, h, w = self.x_data.shape
        # self.y_data = (self.y_data + 1) / 2  # [-1, 1] -> [0, 1]
        self.x_data, self.y_data = self.resize_and_pad(self.x_data, h, w), self.resize_and_pad(self.y_data, h, w)
        self.x_data, self.y_data = random_augmentation(self.x_data, self.y_data)

        return self.x_data.float(), self.y_data.float()
    
    def resize_and_pad(self, image, height, width):
        target_size = 256

        # 1. 가로와 세로 중 큰 축을 선택하여 그 값을 256으로 맞추기
        if width > height:
            new_width = target_size
            new_height = int(target_size * height / width)
        else:
            new_height = target_size
            new_width = int(target_size * width / height)

        # 2. 비율을 유지한 리사이즈 (이미지를 리사이즈)
        image = F.interpolate(image.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)

        # 3. 256x256 크기로 만들기 위해 나머지 부분을 0 패딩
        padding_top = (target_size - new_height) // 2
        padding_left = (target_size - new_width) // 2
        padding_bottom = target_size - new_height - padding_top
        padding_right = target_size - new_width - padding_left

        # 0 패딩을 추가하여 이미지 크기 맞추기
        image = F.pad(image, (padding_left, padding_right, padding_top, padding_bottom), mode='constant', value=0)
        # 4. 수정된 이미지를 리턴
        return image




    def __len__(self):
        return len(self.y_path)
    
    
    



from torch.utils.data import random_split

def make_loader(csv_path='/Users/Han/Desktop/capstone/JaPyGuri-AI/dataset/source_labeled/file_path.csv', batch_size=1):
    dataset = SourceDataset(csv_path)
    total_size = len(dataset)

    # Define split sizes
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    # Randomly split dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader for each subset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



if __name__ == "__main__":
    csv_path = '/Users/Han/Desktop/capstone/JaPyGuri-AI/dataset/source_labeled/file_path.csv'
    train_loader, val_loader = make_loader(csv_path, batch_size=8)

    import matplotlib.pyplot as plt

    for i in range(len(train_loader.dataset)):
        image, label = train_loader.dataset[i]

        print(image.shape, label.shape)
        print(f'image - max : {torch.max(image)}, min : {torch.min(image)}')
        print(f'label - max : {torch.max(label)}, min : {torch.min(label)}')

        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
        # 원본 이미지와 레이블
        revised = np.transpose(image.numpy(), (1, 2, 0))
        axes[0].imshow(np.transpose(image.numpy(), (1, 2, 0)))  # 채널 순서 조정 (H, W, C)
        axes[0].set_title("Original Image")
        axes[1].imshow(np.transpose(label.numpy(), (1, 2, 0)).squeeze(), cmap="gray")  # 레이블 (단일 채널, H, W)
        axes[1].set_title("Original Label")

    
        # 그래프 설정
        for ax in axes.flat:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    for i in range(len(val_loader.dataset)):
        image, label = val_loader.dataset[i]

        print(image.shape, label.shape)
        print(f'image - max : {torch.max(image)}, min : {torch.min(image)}')
        print(f'label - max : {torch.max(label)}, min : {torch.min(label)}')

        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
        # 원본 이미지와 레이블
        axes[0].imshow(np.transpose(image.numpy(), (1, 2, 0)))  # 채널 순서 조정 (H, W, C)
        axes[0].set_title("Original Image")
        axes[1].imshow(label.numpy().squeeze(), cmap="gray")  # 레이블 (단일 채널, H, W)
        axes[1].set_title("Original Label")

    
        # 그래프 설정
        for ax in axes.flat:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

