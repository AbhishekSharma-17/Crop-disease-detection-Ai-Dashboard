# dataset.py

import os
import json
from PIL import Image
from torch.utils.data import Dataset

class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        # root_dir: Path to the directory containing the image data
        # transform: Transformations to be applied to the images (e.g., resizing, normalization)
        # split: Specifies the data split ('train', 'valid', or 'test')
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.images = []
        self.labels = []

        # Load annotations from the COCO JSON file
        annotation_file = os.path.join(self.root_dir, '_annotations.coco.json')
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        print(f"Loaded {len(annotations['images'])} images for split: {split}")
        print(f"Loaded {len(annotations['annotations'])} annotations for split: {split}")

        # Extract image paths and labels from annotations
        for annotation in annotations['annotations']:
            image_id = annotation['image_id']
            image_path = os.path.join(self.root_dir, annotations['images'][image_id]['file_name'])
            label = annotation['category_id']
            self.images.append(image_path)
            self.labels.append(label)

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve the image and label for the given index
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        # Apply transformations to the image if specified
        if self.transform:
            image = self.transform(image)

        return image, label