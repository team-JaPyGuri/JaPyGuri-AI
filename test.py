import numpy as np
import pandas as pd
import os
import glob as gb
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import cv2

from skimage.io import imread, imshow
from skimage.transform import resize

TRAIN_PATH ='/Users/Han/Desktop/capstone/JaPyGuri-AI/dataset/source_labeled/images'
print(os.path.exists(TRAIN_PATH))
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
train_ids = gb.glob(TRAIN_PATH + '/*.jpg')

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)): 
    img = imread(id_)[:,:,:IMG_CHANNELS] 
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True) 
    X_train[n] = img 

    
TRAIN_PATH ='/Users/Han/Desktop/capstone/JaPyGuri-AI/dataset/source_labeled/labels'
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1
train_ids = gb.glob(TRAIN_PATH + '/*.jpg')
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)): 
    mask = imread(id_)[:,:,:1] 
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True) 
    Y_train[n] = mask 


from model.unet import unet
from loss.diceBCEloss import DiceBCELoss
import torch

model = unet(3,1)
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
def dice(y_true, y_pred):
    num = y_true.size(0)
    eps = 1e-7

    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)

    intersection = (y_true_flat * y_pred_flat).sum(1)

    score = (2 * intersection) / (y_true_flat.sum(1) + y_pred_flat.sum(1) + eps)
    score = score.sum() / num
    return score




import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# 데이터셋을 PyTorch TensorDataset으로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)  # CHW 형식으로 변환
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).permute(0, 3, 1, 2)  # CHW 형식으로 변환

# 훈련 데이터와 검증 데이터를 나누기 (80% 훈련, 20% 검증)
train_size = int(0.8 * len(X_train_tensor))+1
val_size = len(X_train_tensor) - train_size
train_dataset, val_dataset = random_split(TensorDataset(X_train_tensor, Y_train_tensor), [train_size, val_size])

# DataLoader 준비
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 모델 학습 및 검증 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    # GPU 사용 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()  # 모델을 학습 모드로 설정
        running_loss = 0.0
        running_dice = 0.0
        
        # 훈련 루프
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Gradients 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(inputs)
            
            # 손실 계산
            loss = criterion(outputs, labels)
            dice_score = dice(labels, outputs)
            
            # 역전파 및 최적화
            loss.backward()
            optimizer.step()
            
            # 손실 및 Dice 계측
            running_loss += loss.item()
            running_dice += dice_score.item()
        
        # 에폭마다 훈련 손실 및 Dice 출력
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {running_loss/len(train_loader):.4f}, Train Dice Score: {running_dice/len(train_loader):.4f}')
        
        # 검증 루프
        model.eval()  # 모델을 평가 모드로 설정
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():  # 검증 단계에서는 gradient 계산을 하지 않음
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 순전파
                outputs = model(inputs)
                
                # 손실 계산
                loss = criterion(outputs, labels)
                dice_score = dice(labels, outputs)
                
                val_loss += loss.item()
                val_dice += dice_score.item()
        
        # 에폭마다 검증 손실 및 Dice 출력
        print(f'Epoch [{epoch+1}/{num_epochs}] - Val Loss: {val_loss/len(val_loader):.4f}, Val Dice Score: {val_dice/len(val_loader):.4f}')
        
        # 모델 체크포인트 저장 (선택 사항)
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')

# 모델 학습 및 검증
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
