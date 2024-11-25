import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.autoencoder import *
from dataset.make_dataset import *
import os
from torch.utils.data import ConcatDataset
from dataset.make_dataset import *
from torch.nn import MSELoss

# 모델 정의
enc = encoder(in_channel=3)
dec = decoder(out_channel=3)

# 모델 파라미터 파일 로드 (CPU로 로드하도록 수정)
path = "/Users/Han/Desktop/capstone/JaPyGuri-AI/e_49.pth"
enc.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
path = "/Users/Han/Desktop/capstone/JaPyGuri-AI/d_49.pth"
dec.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

# 데이터 로딩
train, val = make_target_loader(batch_size=16)

combined_dataset = ConcatDataset([train.dataset, val.dataset])
combined_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True)

enc.eval()
dec.eval()

criterion = MSELoss()

# 훈련셋 전체에 대해 예측 및 시각화
def visualize_predictions(encoder, decoder, data_loader, criterion, save_dir=None):
    # 배치별로 예측 수행
    for idx, (inputs, targets) in enumerate(data_loader):
        # 이미지를 모델에 넣고 예측을 받기
        with torch.no_grad():  # 평가 시에는 그래디언트 계산을 하지 않음
            latent, size = encoder(inputs)
            output = decoder(latent, size)  # 모델 예측

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # 입력 이미지 시각화 (배치 차원 제거, [C, H, W] -> [H, W])
        ax[0].imshow(inputs[0].permute(1, 2, 0).cpu().numpy())  # [H, W, C] 형태로 변환
        ax[0].set_title(f" Noisy Image {idx + 1}")
        ax[0].axis('off')

        # 예측된 마스크 시각화
        ax[1].imshow(output[0].permute(1, 2, 0).cpu().numpy())  # [H, W, C] 형태로 변환
        ax[1].set_title(f" Reconstructed Image {idx + 1}")
        ax[1].axis('off')

        # 실제 GT 마스크 시각화
        ax[2].imshow(targets[0].permute(1, 2, 0).cpu().numpy())  # [H, W, C] 형태로 변환
        ax[2].set_title(f" Ground Truth Mask {idx + 1}")
        ax[2].axis('off')

        plt.show()

        loss = criterion(output, targets)
        print(f'loss : {loss}')


        # 이미지 저장
        # save_path = os.path.join(save_dir, f"batch_{idx + 1}.png")  # 저장할 경로와 파일 이름
        # plt.savefig(save_path)  # 파일로 저장
        # plt.close()  # 플롯을 닫아서 메모리 절약

        # print(f"Saved {save_path}")

        # 원하는 만큼 배치 시각화를 진행 후 종료하려면, break를 주석처리하거나 원하는 수로 변경
        # break  # 전체 배치에 대해 시각화하려면 이 줄을 삭제하거나 주석처리하세요.

# 훈련셋과 검증셋에 대해 시각화 실행
visualize_predictions(enc, dec, combined_loader, criterion)
