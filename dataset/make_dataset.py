from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

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

        return self.x_data.float(), self.y_data.float()

    def __len__(self):
        return len(self.y_path)


def make_loader(csv_path='/Users/Han/Desktop/capstone/dataset/source_labeled/file_path.csv', batch_size=1):
    dataset = SourceDataset(csv_path)
    total_size = len(dataset)
    
    train_size = int(0.8 * total_size)
    
    train_dataset = data.Subset(dataset, range(0, train_size))
    val_dataset = data.Subset(dataset, range(train_size, total_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    print("make_dataset.py")
    csv_path = '/Users/Han/Desktop/capstone/dataset/source_labeled/file_path.csv'
    train_loader, val_loader = make_loader(csv_path, batch_size=1)

    # 각 로더의 배치 수 확인
    print(f"학습용 배치 수: {len(train_loader)}")
    print(f"검증용 배치 수: {len(val_loader)}")
