import torch
import random
import torchvision.transforms.functional as F
import torchvision


def random_augmentation(image, label):
    """
    Apply a series of augmentations to an RGB image and its segmentation label.
    Each augmentation is applied with a 50% probability.
    """
    # Ensure the image is within [0, 1] before applying augmentations
    image = torch.clamp(image, 0.0, 1.0)

    # Random Horizontal Flip
    if random.random() > 0.5:
        image = F.hflip(image)
        label = F.hflip(label)
    
    # Random Vertical Flip
    if random.random() > 0.5:
        image = F.vflip(image)
        label = F.vflip(label)

    # Random Rotation (-30 to 30 degrees)
    if random.random() > 0.5:
        angle = random.uniform(-30, 30)
        image = F.rotate(image, angle)
        label = F.rotate(label, angle)

    # Random Color Jitter for image (brightness, contrast, saturation, and hue)
    if random.random() > 0.5:
        brightness_factor = random.uniform(1.0, 1.3)
        contrast_factor = random.uniform(0.8, 1.2)
        saturation_factor = random.uniform(0.8, 1.2)
        hue_factor = random.uniform(-0.1, 0.1)
        image = F.adjust_brightness(image, brightness_factor)
        image = F.adjust_contrast(image, contrast_factor)
        image = F.adjust_saturation(image, saturation_factor)
        image = F.adjust_hue(image, hue_factor)

    # Adding Gaussian Noise to image
    if random.random() > 0.5:
        noise = torch.randn_like(image) * 0.05
        image = image + noise
        image = torch.clamp(image, 0.0, 1.0)  # Ensure the image is within [0, 1]

    # Random Gaussian Blur for image
    if random.random() > 0.5:
        sigma = random.uniform(0.1, 2.0)
        image = F.gaussian_blur(image, kernel_size=3, sigma=sigma)

    # Ensure the image is within [0, 1] after all augmentations
    image = torch.clamp(image, 0.0, 1.0)

    return image, label
