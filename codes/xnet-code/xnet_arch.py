import torch
import torch.nn as nn
import torch.nn.functional as F

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