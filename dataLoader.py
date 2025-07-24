import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import pandas as pd
import numpy as np

# Root dataset path
DATASET_ROOT = '/home/phd/datasets/'

# Dataset-specific paths (adjust folder names as per your actual structure)
DATASET_PATHS = {
    'NIH': os.path.join(DATASET_ROOT, 'NIH_Chest_Xray'),
    'BRaTS': os.path.join(DATASET_ROOT, 'BRaTS'),
    'Camelyon16': os.path.join(DATASET_ROOT, 'Camelyon16'),
    'PANDA': os.path.join(DATASET_ROOT, 'PANDA'),
    'ISIC': os.path.join(DATASET_ROOT, 'ISIC2019'),
    'BreakHis': os.path.join(DATASET_ROOT, 'BreakHis')
}

# Supported datasets
SUPPORTED_DATASETS = list(DATASET_PATHS.keys())

# Class labels for each dataset
CLASS_LABELS = {
    'NIH': ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'],
    'BRaTS': ['Background', 'Non-Enhancing_Tumor', 'Edema', 'Enhancing_Tumor'],
    'Camelyon16': ['Normal_Tissue', 'Tumor_Tissue'],
    'PANDA': ['Gleason_3_3', 'Gleason_3_4', 'Gleason_4_3', 'Gleason_4_4', 'Gleason_5_5'],
    'ISIC': ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'],
    'BreakHis': ['SOB_B_A', 'SOB_B_F', 'SOB_B_TA', 'SOB_B_PT', 'SOB_B_DC', 'SOB_B_LC', 'SOB_B_MC', 'SOB_B_FL']
}

# Modality-specific image sizes and normalization
DATASET_CONFIG = {
    'NIH': {'size': (224, 224), 'mean': [0.505, 0.505, 0.505], 'std': [0.275, 0.275, 0.275]},
    'BRaTS': {'size': (240, 240), 'mean': [0.0] * 4, 'std': [1.0] * 4},  # 4-channel MRI
    'Camelyon16': {'size': (224, 224), 'mean': [0.775, 0.620, 0.750], 'std': [0.170, 0.215, 0.165]},
    'PANDA': {'size': (224, 224), 'mean': [0.640, 0.500, 0.670], 'std': [0.190, 0.225, 0.175]},
    'ISIC': {'size': (224, 224), 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
    'BreakHis': {'size': (224, 224), 'mean': [0.7, 0.55, 0.7], 'std': [0.2, 0.25, 0.2]}
}

class CustomImageFolder(Dataset):
    """Custom dataset loader to handle different structures and preprocessing."""
    def __init__(self, root, transform=None, loader=None):
        self.dataset = ImageFolder(root=root, transform=transform)
        self.loader = loader if loader else self.default_loader
        self.imgs = self.dataset.imgs
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def default_loader(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.dataset.transform:
            img = self.dataset.transform(img)
        return img, target, path  # Return image, label, and path

    def __len__(self):
        return len(self.imgs)


class BRaTSDataset(Dataset):
    """Specialized loader for BRaTS MRI data (multi-modal, 4 channels)."""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for case_dir in os.listdir(self.root):
            case_path = os.path.join(self.root, case_dir)
            if os.path.isdir(case_path):
                # Assume modalities: T1, T1c, T2, FLAIR
                modalities = ['T1.npy', 'T1c.npy', 'T2.npy', 'FLAIR.npy']
                image_paths = [os.path.join(case_path, m) for m in modalities]
                if all(os.path.exists(p) for p in image_paths):
                    # Dummy label for unsupervised learning
                    samples.append((image_paths, 0))
        return samples

    def __getitem__(self, index):
        img_paths, target = self.samples[index]
        # Load and stack 4D MRI volume (simulate single slice for simplicity)
        slices = []
        for path in img_paths:
            slice_2d = np.load(path)  # Assume shape (H, W)
            slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
            slices.append(slice_2d)
        image = np.stack(slices, axis=0)  # (4, H, W)

        if self.transform:
            # Apply transforms (e.g., resize, normalize)
            image = self.transform(image)
        return torch.from_numpy(image).float(), target

    def __len__(self):
        return len(self.samples)


def get_transforms(dataset_name, is_train=True):
    """Return dataset-specific transforms."""
    config = DATASET_CONFIG[dataset_name]
    size = config['size']
    mean = config['mean']
    std = config['std']

    if dataset_name == 'BRaTS':
        # For 4-channel MRI, we define a custom transform
        return transforms.Compose([
            transforms.Lambda(lambda x: x),  # Placeholder; resizing handled in dataset
        ])
    else:
        # For 3-channel RGB images
        t = []
        if is_train:
            t.append(transforms.RandomResizedCrop(size, scale=(0.8, 1.0)))
            t.append(transforms.RandomHorizontalFlip())
        else:
            t.append(transforms.Resize(size))
        t.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return transforms.Compose(t)


def load_dataset(dataset_name, domain_id, is_train=True, batch_size=64, num_workers=4):
    """
    Load a specific dataset and domain.

    Args:
        dataset_name (str): One of 'NIH', 'BRaTS', 'Camelyon16', 'PANDA', 'ISIC', 'BreakHis'
        domain_id (str): Domain identifier (e.g., 'D1', 'D2', 'NIH', 'MIMIC')
        is_train (bool): Whether to load training data
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader

    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from {SUPPORTED_DATASETS}")

    root_path = DATASET_PATHS[dataset_name]
    domain_path = os.path.join(root_path, domain_id)

    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"Domain path not found: {domain_path}")

    transform = get_transforms(dataset_name, is_train=is_train)

    if dataset_name == 'BRaTS':
        dataset = BRaTSDataset(root=domain_path, transform=transform)
    else:
        dataset = CustomImageFolder(root=domain_path, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )

    print(f"Loaded {len(dataset)} samples from {dataset_name} domain {domain_id}")
    return dataloader


# === Example Usage ===
if __name__ == "__main__":
    # Example: Load NIH Chest X-ray from domain D1
    try:
        dataloader = load_dataset('NIH', 'D1', is_train=True, batch_size=32)
        for batch_idx, (data, target, path) in enumerate(dataloader):
            print(f"Batch {batch_idx}: Data shape {data.shape}, Target {target}")
            if batch_idx == 0:
                break  # Just show first batch
    except Exception as e:
        print(f"Error loading dataset: {e}")

    # Example: Load BRaTS data
    try:
        brats_loader = load_dataset('BRaTS', 'SiteA', is_train=True, batch_size=16)
        for data, target in brats_loader:
            print(f"BRaTS Batch: Data shape {data.shape}, Target {target}")
            break
    except Exception as e:
        print(f"Error loading BRaTS: {e}")