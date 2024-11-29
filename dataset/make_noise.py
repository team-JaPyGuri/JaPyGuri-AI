import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import torchvision.transforms.functional as F

def add_salt_pepper_noise(image, noise_factor=0.4):
    
    # 이미지를 numpy 배열로 변환
    image = image.numpy()

    # 전체 픽셀 수
    total_pixels = image.size

    # Salt (흰색 점 추가)
    num_salt = int(total_pixels * noise_factor * 0.5)  # salt와 pepper의 비율은 동일하게 나눔
    salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    image[salt_coords[0], salt_coords[1], salt_coords[2]] = 1  # Salt는 1로 설정 (흰색)

    # Pepper (검은색 점 추가)
    num_pepper = int(total_pixels * noise_factor * 0.5)  # pepper는 salt와 같은 비율
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    image[pepper_coords[0], pepper_coords[1], pepper_coords[2]] = 0  # Pepper는 0으로 설정 (검은색)

    # 노이즈가 추가된 이미지를 다시 Tensor로 변환하고 [0, 1] 범위로 반환
    return torch.tensor(image).float()


def add_black_patches(image, patch_ratio=0.2, patch_count=5):
    
    # 이미지를 numpy 배열로 변환
    image = image.numpy()

    # 이미지 크기
    C, H, W = image.shape

    # 각 패치의 크기 계산 (H, W의 비율로 패치 크기 결정)
    patch_h = int(H * patch_ratio)  # 패치의 높이
    patch_w = int(W * patch_ratio)  # 패치의 너비

    for _ in range(patch_count):
        # 패치가 들어갈 랜덤 위치 생성 (패치가 이미지 밖으로 나가지 않도록 제한)
        top = np.random.randint(0, H - patch_h)
        left = np.random.randint(0, W - patch_w)

        # 해당 위치에 검정색 패치를 추가
        image[:, top:top+patch_h, left:left+patch_w] = 0  # 패치는 0으로 설정 (검은색)

    # 노이즈가 추가된 이미지를 다시 Tensor로 변환하고 [0, 1] 범위로 반환
    return torch.tensor(image).float()


def make_noise(img, noise_factor=0.4, patch_ratio=0.05, patch_cnt=20):
    noisy_img = torch.clone(img)
    noisy_img = add_salt_pepper_noise(noisy_img,noise_factor=noise_factor)
    if patch_cnt > 0 or patch_ratio > 0.0:
        noisy_img = add_black_patches(noisy_img, patch_ratio=patch_ratio, patch_count=patch_cnt)
    return noisy_img


def visualize_images(original_image, noisy_image, figsize=(10, 5)):

    # 시각화
    plt.figure(figsize=figsize)
    
    # 원본 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(torch.permute(original_image,(1,2,0)))
    plt.title("Original Image")
    plt.axis("off")
    
    # 노이즈 이미지
    plt.subplot(1, 2, 2)
    plt.imshow(torch.permute(noisy_image,(1,2,0)))
    plt.title("Noisy Image")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from make_dataset import make_target_loader

    train_loader, val_loader = make_target_loader(batch_size=8)

    for noisy, images in val_loader:
        for n, i in zip(noisy, images):
            print(n.shape)
            visualize_images(i, n)
        break  # 첫 번째 배치만 시각화
