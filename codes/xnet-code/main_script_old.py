import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Custom Dataset Class
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
            
            if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
                print(f"Warning: Path does not exist - {img_dir} or {ann_dir}")
                continue
                
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
        
        # Extract label
        label = sample['class_idx']
        
        if self.transform:
            image = self.transform(image)
        
        # Simplified return - just image and class label for classification
        return image, label

# XNet Model Class
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class XNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=10):
        super(XNet, self).__init__()
        
        # Encoder path
        self.encoder1 = DoubleConv(n_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Middle
        self.middle = DoubleConv(512, 1024)
        
        # Decoder path
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)
        
        # Cross connections (what makes this XNet)
        self.cross_conv1 = nn.Conv2d(64, 64, kernel_size=1)
        self.cross_conv2 = nn.Conv2d(128, 128, kernel_size=1)
        self.cross_conv3 = nn.Conv2d(256, 256, kernel_size=1)
        self.cross_conv4 = nn.Conv2d(512, 512, kernel_size=1)
        
        # Final classification layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, n_classes)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        x = self.pool(enc1)
        
        enc2 = self.encoder2(x)
        x = self.pool(enc2)
        
        enc3 = self.encoder3(x)
        x = self.pool(enc3)
        
        enc4 = self.encoder4(x)
        x = self.pool(enc4)
        
        # Middle
        x = self.middle(x)
        
        # Decoder with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, self.cross_conv4(enc4)], dim=1)
        x = self.decoder4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, self.cross_conv3(enc3)], dim=1)
        x = self.decoder3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, self.cross_conv2(enc2)], dim=1)
        x = self.decoder2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, self.cross_conv1(enc1)], dim=1)
        x = self.decoder1(x)
        
        # Classification head
        features = self.avg_pool(x)
        features = torch.flatten(features, 1)
        output = self.classifier(features)
        
        return output

# Training and evaluation functions
def train_model(model, dataloaders, criterion, optimizer, num_epochs=1000, device='cuda'):
    model = model.to(device)
    
    # Initialize variables to track best model
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_model.pth')
        
        print()
    
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(torch.load('best_model.pth'))
    return model, history

def visualize_results(history):
    # Plot loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Epoch')
    plt.savefig('training_results.png')
    plt.show()

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print(classification_report(all_labels, all_preds))

def main():
    # Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 512)),  # Resize to smaller dimensions for memory efficiency
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    # Fix the path - replace this with your actual path
    root_dir = r'F:\Projects\006_Mansiba_Gohil_36_Metal_Defect_Detection\classification'
    dataset = MetalDefectDataset(root_dir=root_dir, transform=transform)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Split into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = XNet(n_channels=3, n_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    model, history = train_model(
        model, dataloaders, criterion, optimizer, 
        num_epochs=1000, device=device)
    
    # Visualize training results
    visualize_results(history)
    
    # Evaluate on test set
    evaluate_model(model, test_loader, device=device)

if __name__ == '__main__':
    main()

'''Possible Issues

Large Test Dataset: If your test dataset is very large, the evaluation might take a long time
GPU Memory Issues: Converting large tensors between GPU and CPU could cause memory issues
Batch Size: If your batch size is too small, evaluation will take more iterations'''