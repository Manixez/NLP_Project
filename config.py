"""
Konfigurasi Hyperparameter untuk Image Captioning Pipeline
"""
import os
from pathlib import Path

# ==================== PATH CONFIGURATION ====================
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset paths (sudah di-copy langsung ke Dataset/)
DATASET_DIR = PROJECT_ROOT / "Dataset"
IMAGES_DIR = DATASET_DIR / "Images"
METADATA_FILE = DATASET_DIR / "metadata.csv"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "saved_models"
VOCAB_DIR = OUTPUT_DIR / "vocab"
LOGS_DIR = OUTPUT_DIR / "logs"

# FastText model path (download dari https://fasttext.cc/docs/en/crawl-vectors.html)
# Model: cc.id.300.bin (Indonesian)
FASTTEXT_MODEL_PATH = PROJECT_ROOT / "fasttext" / "cc.id.300.bin"

# ==================== DATA CONFIGURATION ====================
TRAIN_VAL_SPLIT = 0.9  # 90% training, 10% validation
MAX_CAPTION_LENGTH = 50  # Panjang maksimal caption (dalam kata)
MIN_WORD_FREQ = 2  # Kata harus muncul minimal 2x untuk masuk vocabulary

# Special tokens
PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'
UNK_TOKEN = '<unk>'

SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

# ==================== MODEL CONFIGURATION ====================
# Encoder (CNN)
CNN_MODEL = 'resnet101'  # Options: 'vgg16', 'resnet50', 'resnet101'
IMG_SIZE = (224, 224)  # Input size for CNN
ENCODED_IMAGE_SIZE = 14  # Spatial size (14x14 for VGG16, ResNet)

# Decoder (LSTM)
EMBEDDING_DIM = 300  # FastText dimension
HIDDEN_SIZE = 512  # LSTM hidden units
NUM_LAYERS = 1  # Number of LSTM layers
DROPOUT = 0.5  # Dropout probability

# ==================== TRAINING CONFIGURATION ====================
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
GRADIENT_CLIP = 5.0  # Gradient clipping threshold

# Teacher forcing
TEACHER_FORCING_RATIO = 1.0  # Always use teacher forcing during training

# Early stopping
PATIENCE = 5  # Stop if validation BLEU doesn't improve for N epochs

# ==================== DEVICE CONFIGURATION ====================
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4  # DataLoader workers

# ==================== EVALUATION CONFIGURATION ====================
BEAM_SIZE = 3  # Beam search width (set to 1 for greedy decoding)

# ==================== LOGGING ====================
PRINT_FREQ = 100  # Print training stats every N batches
SAVE_FREQ = 1  # Save model every N epochs

# ==================== CREATE DIRECTORIES ====================
def create_output_dirs():
    """Buat semua folder output yang diperlukan"""
    for directory in [OUTPUT_DIR, MODELS_DIR, VOCAB_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directories created at {OUTPUT_DIR}")

if __name__ == "__main__":
    create_output_dirs()
    print("\n📋 Configuration Summary:")
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Images: {IMAGES_DIR}")
    print(f"  Metadata: {METADATA_FILE}")
    print(f"  FastText: {FASTTEXT_MODEL_PATH}")
    print(f"  Device: {DEVICE}")
    print(f"  CNN Model: {CNN_MODEL}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
