import torch
import random
import torchvision.transforms.functional as F
import torchvision

def random_augmentation(image, label):
    """
    Apply a series of augmentations to an RGB image and its segmentation label.
    Each augmentation is applied with a 50% probability, with final clamping to keep values within [0,1].
    """

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
        image = torch.clamp(image, 0, 1)

    # Adding Gaussian Noise to image
    if random.random() > 0.5:
        noise = torch.randn_like(image) * 0.05
        image = image + noise
        image = torch.clamp(image, 0, 1)

    # Random Crop and Resize (applies to both image and label)
    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.0)
        target_size = [int(scale * image.shape[1]), int(scale * image.shape[2])]
        image = F.resize(image, target_size)
        label = F.resize(label, target_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        # Center Crop to match original size
        image = F.center_crop(image, (image.shape[1], image.shape[2]))
        label = F.center_crop(label, (label.shape[1], label.shape[2]))

    # Random Gaussian Blur for image
    if random.random() > 0.5:
        sigma = random.uniform(0.1, 2.0)
        image = F.gaussian_blur(image, kernel_size=3, sigma=sigma)
        image = torch.clamp(image, 0, 1)

    # Final clamp to ensure all values within [0, 1]
    image = torch.clamp(image, 0, 1)
    label = torch.clamp(label, 0, 1)  # Clamp label as well if it's float-based

    return image, label
