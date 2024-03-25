# Importing from Library
import json
import os
import random
import torch
from torchvision.transforms import v2
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

# Importing from local
from config import WALDO_PATH, TRAIN_JSON_PATH, TEST_JSON_PATH, TRAIN_IMG_PATH, TEST_IMG_PATH


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.images = list(self.img_dir.glob('*.jpg'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)  # Apply transformations
        return image


class PrepareImage(object):
    def __init__(self, img_dir='travel_adventure', size=1000):
        super(PrepareImage, self).__init__()

        self.train_transform = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.5, hue=0.3),
            PutWaldo(mode='train'),
        ])

        self.test_transform = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            PutWaldo(mode='test'),
        ])

        self.img_dir = img_dir
        self.size = size
        self.train_metadata = []
        self.test_metadata = []

        self.process_images()

    def process_images(self):
        dataset = CustomImageDataset(img_dir=self.img_dir, transform=None)  # No transform here; applied later
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        train_indices = list(range(0, train_size))
        test_indices = list(range(train_size, train_size + test_size))

        for idx in train_indices:
            img = dataset[idx]
            transformed_img, meta_data = self.train_transform(img)
            self.save_image_and_metadata(transformed_img, meta_data, idx, 'train')

        for idx in test_indices:
            img = dataset[idx]  # Using the original dataset but applying test_transform
            transformed_img, meta_data = self.test_transform(img)
            self.save_image_and_metadata(transformed_img, meta_data, idx - train_size, 'test')  # Adjust idx for test

        self.write_metadata_to_json()

    def save_image_and_metadata(self, img, metadata, idx, mode):
        img_dir = TRAIN_IMG_PATH if mode == 'train' else TEST_IMG_PATH
        file_name = f'{mode}_image_{idx}.jpg'
        img_path = os.path.join(img_dir, file_name)

        img = F.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        img.save(img_path)

        metadata['image_path'] = img_path
        if mode == 'train':
            self.train_metadata.append(metadata)
        else:
            self.test_metadata.append(metadata)

    def write_metadata_to_json(self):
        with open(TRAIN_JSON_PATH, 'w') as f:
            json.dump(self.train_metadata, f, indent=4)
        with open(TEST_JSON_PATH, 'w') as f:
            json.dump(self.test_metadata, f, indent=4)


class PutWaldo:
    def __init__(self, mode='train'):
        self.mode = mode  # 'train' or 'test'

    def __call__(self, img):
        img, metadata = self.put_waldo(img)
        return img, metadata

    def put_waldo(self, img):
        is_present = 0
        img = img.convert('RGBA')

        with Image.open(WALDO_PATH) as waldo:
            waldo = waldo.convert('RGBA')
            waldo_width, waldo_height = waldo.size
            bg_width, bg_height = img.size

            max_width_pos = bg_width - waldo_width
            max_height_pos = bg_height - waldo_height

            if random.choice([True, False]):
                is_present = 1
                start_x = random.randint(0, max_width_pos)
                start_y = random.randint(0, max_height_pos)
                end_x = start_x + waldo_width
                end_y = start_y + waldo_height
                img.paste(waldo, (start_x, start_y), waldo)

        img = img.convert('RGB')
        metadata = {
            "is_present": is_present,
            "bbox": None if is_present == 0 else {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y
            }
        }
        return img, metadata
