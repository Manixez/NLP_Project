"""
FASE 1: CNN Encoder
===================
Extract fitur visual dari gambar menggunakan pre-trained CNN
"""
import torch
import torch.nn as nn
import torchvision.models as models

import config


class EncoderCNN(nn.Module):
    """
    CNN Encoder untuk ekstraksi fitur gambar
    
    Input: Gambar (batch_size, 3, 224, 224)
    Output: Vektor Fitur (batch_size, encoded_size, encoded_size, feature_dim)
            Contoh VGG16: (batch_size, 14, 14, 512)
    """
    
    def __init__(self, encoded_size=14, fine_tune=False):
        """
        Args:
            encoded_size: Spatial size output (14 untuk VGG16/ResNet)
            fine_tune: True jika ingin fine-tune CNN, False freeze weights
        """
        super(EncoderCNN, self).__init__()
        
        self.encoded_size = encoded_size
        self.fine_tune = fine_tune
        
        # Load pre-trained model
        if config.CNN_MODEL == 'vgg16':
            vgg = models.vgg16(pretrained=True)
            # Ambil layer features (convolutional layers)
            self.cnn = nn.Sequential(*list(vgg.features.children())[:-1])
            self.feature_dim = 512  # VGG16 output channels
            
        elif config.CNN_MODEL == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            # Ambil semua layer kecuali avgpool dan fc
            modules = list(resnet.children())[:-2]
            self.cnn = nn.Sequential(*modules)
            self.feature_dim = 2048  # ResNet50 output channels
            
        elif config.CNN_MODEL == 'resnet101':
            resnet = models.resnet101(pretrained=True)
            modules = list(resnet.children())[:-2]
            self.cnn = nn.Sequential(*modules)
            self.feature_dim = 2048  # ResNet101 output channels
        
        else:
            raise ValueError(f"Unknown CNN model: {config.CNN_MODEL}")
        
        # Adaptive pooling untuk ensure output size (14, 14)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))
        
        # Freeze CNN weights jika tidak fine-tune
        if not fine_tune:
            for param in self.cnn.parameters():
                param.requires_grad = False
    
    def forward(self, images):
        """
        Forward pass
        
        Args:
            images: Tensor (batch_size, 3, 224, 224)
        
        Returns:
            features: Tensor (batch_size, encoded_size, encoded_size, feature_dim)
                     Contoh: (32, 14, 14, 512) untuk VGG16
        """
        with torch.set_grad_enabled(self.fine_tune):
            # Extract features from CNN
            features = self.cnn(images)  # (batch, channels, H, W)
            
            # Adaptive pooling to fixed size
            features = self.adaptive_pool(features)  # (batch, channels, encoded_size, encoded_size)
        
        # Permute: (batch, channels, H, W) -> (batch, H, W, channels)
        # Ini agar spatial features tetap terstruktur untuk attention (opsional di masa depan)
        features = features.permute(0, 2, 3, 1)  # (batch, encoded_size, encoded_size, feature_dim)
        
        return features
    
    def fine_tune(self, fine_tune=True):
        """
        Enable/disable fine-tuning
        
        Args:
            fine_tune: True untuk allow gradient updates
        """
        self.fine_tune = fine_tune
        for param in self.cnn.parameters():
            param.requires_grad = fine_tune


if __name__ == "__main__":
    print("Testing CNN Encoder...")
    
    # Create encoder
    encoder = EncoderCNN(encoded_size=14, fine_tune=False)
    encoder.eval()
    
    # Test dengan dummy batch
    dummy_images = torch.randn(4, 3, 224, 224)  # batch_size=4
    
    with torch.no_grad():
        features = encoder(dummy_images)
    
    print(f"\n✅ Encoder Test:")
    print(f"   Input shape: {dummy_images.shape}")
    print(f"   Output shape: {features.shape}")
    print(f"   Feature dim: {encoder.feature_dim}")
    print(f"   Encoded size: {encoder.encoded_size}")
    print(f"   Fine-tune: {encoder.fine_tune}")
    
    # Hitung total parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
