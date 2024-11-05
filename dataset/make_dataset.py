from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from dataset.augmentation import random_augmentation
import torch
import numpy as np

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
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        self.x_data = Image.open(self.x_path[index])
        self.y_data = Image.open(self.y_path[index])
        self.y_data = self.y_data.convert('L')

        # 데이터 정규화
        self.x_data = self.transform(self.x_data)
        self.y_data = self.transform(self.y_data)
        self.y_data = (self.y_data + 1) / 2

        self.x_data, self.y_data = random_augmentation(self.x_data, self.y_data)

        return self.x_data.float(), self.y_data.float()

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
    train_loader, val_loader = make_loader(csv_path, batch_size=1)

    import matplotlib.pyplot as plt

    for i in range(len(val_loader.dataset)):
        image, label = val_loader.dataset[i]

        print(image.shape, label.shape)

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

