"""
FASE 0 & 1: Dataset Loader
===========================
Load gambar dan caption, preprocessing, dan split train/val
"""
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from sklearn.model_selection import train_test_split

import config


class CaptionDataset(Dataset):
    """
    PyTorch Dataset untuk Image Captioning
    
    Setiap item: (image_tensor, caption_indices, caption_length)
    """
    
    def __init__(self, image_paths, captions, vocab, transform=None):
        """
        Args:
            image_paths: List path ke gambar
            captions: List caption (raw text)
            vocab: Vocabulary object
            transform: Transformasi untuk gambar
        """
        self.image_paths = image_paths
        self.captions = captions
        self.vocab = vocab
        self.transform = transform
        
        if self.transform is None:
            self.transform = get_transform(train=True)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor (3, 224, 224)
            caption: Tensor (max_len,) - indices dengan padding
            length: int - panjang caption asli (termasuk <start> dan <end>)
        """
        # Load dan transform gambar
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Encode caption
        caption_text = self.captions[idx]
        caption_indices = self.vocab.encode_caption(caption_text)
        
        # Potong jika terlalu panjang
        if len(caption_indices) > config.MAX_CAPTION_LENGTH:
            caption_indices = caption_indices[:config.MAX_CAPTION_LENGTH - 1] + [self.vocab(config.END_TOKEN)]
        
        # Padding
        length = len(caption_indices)
        padded_caption = caption_indices + [self.vocab(config.PAD_TOKEN)] * (config.MAX_CAPTION_LENGTH - length)
        
        return (
            image,
            torch.tensor(padded_caption, dtype=torch.long),
            length
        )


def get_transform(train=True):
    """
    Transformasi untuk preprocessing gambar
    
    Args:
        train: True jika training (dengan augmentasi), False jika validasi
    """
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def collate_fn(batch):
    """
    Custom collate function untuk DataLoader
    Mengurutkan batch berdasarkan panjang caption (descending) untuk pack_padded_sequence
    
    Args:
        batch: List of (image, caption, length)
    
    Returns:
        images: Tensor (batch_size, 3, 224, 224)
        captions: Tensor (batch_size, max_len)
        lengths: List panjang caption
    """
    # Sort batch by caption length (descending)
    batch.sort(key=lambda x: x[2], reverse=True)
    
    images, captions, lengths = zip(*batch)
    
    images = torch.stack(images, 0)
    captions = torch.stack(captions, 0)
    
    return images, captions, list(lengths)


def load_metadata(metadata_path=None):
    """
    Load metadata CSV dan parse captions
    
    Format CSV yang didukung:
        1. file_name, text (multiple rows per image) ← DATASET KAMU
        2. image_name, caption_1, caption_2, caption_3 (fallback)
    
    Returns:
        image_paths: List path gambar (satu entry per caption)
        captions: List semua caption 
        all_captions_text: List semua caption untuk build vocabulary
    
    Contoh:
        Jika ada 8000 gambar dengan rata-rata 5 caption per gambar,
        maka akan return ~40,000 pairs (image_path, caption)
    """
    metadata_path = metadata_path or config.METADATA_FILE
    images_dir = config.IMAGES_DIR
    
    print(f"📂 Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    print(f"   CSV columns: {df.columns.tolist()}")
    print(f"   Total rows: {len(df)}")
    
    image_paths = []
    captions = []
    all_captions_text = []
    
    # Cek kolom yang ada
    if 'file_name' in df.columns and 'text' in df.columns:
        # ✅ FORMAT DATASET KAMU: file_name, text (multiple rows per image)
        print(f"   Format detected: file_name + text (multiple rows per image)")
        
        missing_images = 0
        for idx, row in df.iterrows():
            img_name = row['file_name']
            img_path = images_dir / img_name
            
            if not img_path.exists():
                missing_images += 1
                continue
            
            caption_text = str(row['text']).strip()
            
            if caption_text and caption_text != 'nan' and len(caption_text) > 0:
                image_paths.append(str(img_path))
                captions.append(caption_text)
                all_captions_text.append(caption_text)
        
        if missing_images > 0:
            print(f"   ⚠️  {missing_images} images tidak ditemukan (diabaikan)")
        
        # Hitung unique images
        unique_images = len(set(image_paths))
        avg_captions = len(captions) / unique_images if unique_images > 0 else 0
        print(f"   ✅ Loaded: {unique_images} unique images, {len(captions)} total captions")
        print(f"   📊 Average captions per image: {avg_captions:.1f}")
        if missing_images > 0:
            print(f"   ⚠️  {missing_images} images tidak ditemukan (diabaikan)")
        
        # Hitung unique images
        unique_images = len(set(image_paths))
        avg_captions = len(captions) / unique_images if unique_images > 0 else 0
        print(f"   ✅ Loaded: {unique_images} unique images, {len(captions)} total captions")
        print(f"   📊 Average captions per image: {avg_captions:.1f}")
    
    else:
        # ⚠️ FALLBACK: format lama (image_name, caption_1, caption_2, caption_3)
        print(f"   Format detected: image_name + multiple caption columns (fallback)")
        caption_cols = [col for col in df.columns if 'caption' in col.lower()]
        
        if not caption_cols:
            caption_cols = ['caption']
        
        for _, row in df.iterrows():
            img_name = row['image_name'] if 'image_name' in df.columns else row.iloc[0]
            img_path = images_dir / img_name
            
            if not img_path.exists():
                continue
            
            for cap_col in caption_cols:
                caption_text = str(row[cap_col]).strip()
                
                if caption_text and caption_text != 'nan':
                    image_paths.append(str(img_path))
                    captions.append(caption_text)
                    all_captions_text.append(caption_text)
        
        unique_images = len(set(image_paths))
        avg_captions = len(captions) / unique_images if unique_images > 0 else 0
        print(f"   ✅ Loaded: {unique_images} unique images, {len(captions)} total captions")
        print(f"   📊 Average captions per image: {avg_captions:.1f}")
    
    return image_paths, captions, all_captions_text


def prepare_dataloaders(vocab, batch_size=None, num_workers=None):
    """
    🎯 MAIN FUNCTION: Prepare train & validation DataLoaders
    
    Args:
        vocab: Vocabulary object
        batch_size: Batch size (default dari config)
        num_workers: Num workers (default dari config)
    
    Returns:
        train_loader, val_loader
    """
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    
    print("\n" + "="*60)
    print("📦 Preparing DataLoaders...")
    print("="*60)
    
    # Load metadata
    image_paths, captions, _ = load_metadata()
    
    # Split train/validation
    train_imgs, val_imgs, train_caps, val_caps = train_test_split(
        image_paths, captions,
        test_size=(1 - config.TRAIN_VAL_SPLIT),
        random_state=42
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training: {len(train_imgs)} samples")
    print(f"   Validation: {len(val_imgs)} samples")
    
    # Create datasets
    train_dataset = CaptionDataset(
        train_imgs, train_caps, vocab,
        transform=get_transform(train=True)
    )
    
    val_dataset = CaptionDataset(
        val_imgs, val_caps, vocab,
        transform=get_transform(train=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"\n✅ DataLoaders ready!")
    print(f"   Batch size: {batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing Dataset Loader...")
    
    # Load vocabulary (harus sudah dibuat dulu)
    from utils.vocabulary import load_vocabulary_and_embeddings
    
    try:
        vocab, _ = load_vocabulary_and_embeddings()
    except:
        print("⚠️  Vocabulary belum dibuat. Jalankan dulu: python utils/vocabulary.py")
        exit(1)
    
    # Test dataloader
    train_loader, val_loader = prepare_dataloaders(vocab, batch_size=4)
    
    # Test satu batch
    images, captions, lengths = next(iter(train_loader))
    print(f"\n🧪 Sample Batch:")
    print(f"   Images shape: {images.shape}")
    print(f"   Captions shape: {captions.shape}")
    print(f"   Lengths: {lengths}")
    
    # Decode caption pertama
    print(f"\n   Sample caption (decoded):")
    print(f"   {vocab.decode_caption(captions[0].tolist())}")
