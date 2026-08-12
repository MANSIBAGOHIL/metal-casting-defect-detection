import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class MetalDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['crease', 'crescent_gap', 'inclusion', 'oil_spot', 
                       'punching_hole', 'rolled_pit', 'silk_spot', 
                       'waist_folding', 'water_spot', 'welding_line']
        
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        
        for class_name in self.classes:
            img_dir = os.path.join(self.root_dir, 'Images', class_name)
            ann_dir = os.path.join(self.root_dir, 'ann_json', class_name)
            
            for filename in os.listdir(img_dir):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(img_dir, filename)
                    json_filename = os.path.splitext(filename)[0] + '.json'
                    json_path = os.path.join(ann_dir, json_filename)
                    
                    if os.path.exists(json_path):
                        samples.append({
                            'image_path': img_path,
                            'json_path': json_path,
                            'class': class_name,
                            'class_idx': self.class_to_idx[class_name]
                        })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        with open(sample['json_path'], 'r') as f:
            annotation = json.load(f)
        
        # Extract bounding boxes
        boxes = []
        for obj in annotation['objects']:
            exterior = obj['points']['exterior']
            x1, y1 = exterior[0]
            x2, y2 = exterior[1]
            boxes.append([x1, y1, x2, y2])
        
        if not boxes:
            # If no objects found, use the whole image
            boxes = [[0, 0, annotation['size']['width'], annotation['size']['height']]]
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64) * sample['class_idx']
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target