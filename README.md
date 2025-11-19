# 🖼️ Image Captioning Indonesia - CNN-LSTM dengan FastText

**Pipeline Image Captioning Bahasa Indonesia** menggunakan arsitektur CNN-LSTM dengan pre-trained FastText embeddings.

## 📋 Deskripsi Pipeline

Pipeline ini mengimplementasikan 4 fase utama:

### 🚀 FASE 0: Persiapan (Pekerjaan Awal)
- ✅ Load dataset (gambar + captions)
- ✅ Build vocabulary (kamus kata) dari semua captions
- ✅ Load FastText pre-trained model Bahasa Indonesia
- ✅ Buat **Embedding Matrix** (jembatan vocab ↔️ FastText)
- ✅ Split data: Training (90%) & Validation (10%)

### 🧠 FASE 1: Training (Proses Belajar)
- ✅ **Encoder (CNN)**: VGG16/ResNet ekstrak fitur visual gambar
- ✅ **Decoder (LSTM)**: Generate caption word-by-word
- ✅ **Teacher Forcing**: Model dipaksa belajar dari kunci jawaban
- ✅ **Backpropagation**: Update bobot LSTM berdasarkan error

### 📝 FASE 2: Validasi (Proses Ujian)
- ✅ Generate caption untuk gambar validation (tanpa teacher forcing)
- ✅ Model menebak kata demi kata secara mandiri
- ✅ Proses berhenti saat model predict token `<end>`

### 📊 FASE 3: Evaluasi (Perhitungan Nilai)
- ✅ Hitung **BLEU-4 Score**: Bandingkan caption AI vs ground truth
- ✅ Track best model berdasarkan BLEU tertinggi

### 🏆 FASE 4: Selesai (Simpan Model)
- ✅ Simpan checkpoint setiap epoch
- ✅ Simpan **best model** (BLEU tertinggi)
- ✅ Early stopping jika BLEU tidak improve

---

## 🛠️ Setup & Installation

### 1. Clone/Download Project
```bash
cd "/home/manix/Documents/Semester 7/NLP/Kode"
```

### 2. Activate Virtual Environment
```bash
source NLP/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download FastText Model
Download model FastText Bahasa Indonesia:
- **URL**: https://fasttext.cc/docs/en/crawl-vectors.html
- **File**: `cc.id.300.bin` (6.8 GB)
- **Simpan di**: `fasttext/cc.id.300.bin`

```bash
mkdir -p fasttext
cd fasttext
# Download manual atau gunakan wget/curl
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.bin.gz
gunzip cc.id.300.bin.gz
cd ..
```

### 5. Download NLTK Data (untuk BLEU)
```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## 📂 Struktur Project

```
Kode/
├── config.py                 # Konfigurasi hyperparameter
├── dataload.py              # Script download dataset (Kaggle)
├── train.py                 # Main training pipeline
├── inference.py             # Generate caption untuk gambar baru
│
├── models/
│   ├── encoder.py           # CNN Encoder (VGG16/ResNet)
│   └── decoder.py           # LSTM Decoder dengan FastText
│
├── utils/
│   ├── vocabulary.py        # Vocabulary builder & embedding matrix
│   └── dataset.py           # PyTorch Dataset & DataLoader
│
├── Dataset/                 # Dataset (download otomatis)
│   └── datasets/joykaihatu/image-caption-indonesia/
│       └── versions/1/
│           ├── Images/      # Folder gambar
│           └── metadata.csv # Captions (3 per gambar)
│
├── fasttext/                # FastText model (download manual)
│   └── cc.id.300.bin
│
├── output/                  # Output training
│   ├── saved_models/        # Model checkpoints
│   ├── vocab/              # Vocabulary & embedding matrix
│   └── logs/               # Training logs
│
└── requirements.txt
```

---

## 🚀 Cara Penggunaan

### Step 1: Download Dataset
Dataset sudah otomatis ter-download ke folder `Dataset/` saat menjalankan `dataload.py`.

```bash
python dataload.py
```

**Output**: Dataset di `Dataset/datasets/joykaihatu/image-caption-indonesia/versions/1/`

---

### Step 2: Build Vocabulary & Embedding Matrix (FASE 0)

```bash
python -c "
from utils.dataset import load_metadata
from utils.vocabulary import build_vocabulary_and_embeddings
import config

# Load captions
_, _, all_captions = load_metadata()

# Build vocab & embedding
vocab, embeddings = build_vocabulary_and_embeddings(all_captions)

print(f'✅ Vocabulary: {len(vocab)} words')
print(f'✅ Embeddings: {embeddings.shape}')
"
```

**Output**:
- `output/vocab/vocab.pkl` (vocabulary object)
- `output/vocab/embedding_matrix.npy` (FastText embeddings)

---

### Step 3: Training (FASE 1-4)

```bash
python train.py
```

**Proses**:
1. Load vocab & embeddings
2. Prepare train/validation dataloaders
3. Build CNN encoder & LSTM decoder
4. Training loop:
   - **FASE 1**: Train dengan teacher forcing
   - **FASE 2**: Validate & generate captions
   - **FASE 3**: Hitung BLEU score
   - **FASE 4**: Simpan best model

**Output**:
- `output/saved_models/checkpoint_epoch_X.pth` (checkpoint setiap epoch)
- `output/saved_models/best_model.pth` (model terbaik)

**Resume Training** (jika terputus):
```bash
python train.py output/saved_models/checkpoint_epoch_10.pth
```

---

### Step 4: Inference (Generate Caption)

Generate caption untuk gambar baru:

```bash
python inference.py --image path/ke/gambar.jpg --show
```

**Contoh**:
```bash
python inference.py \
  --image Dataset/datasets/joykaihatu/image-caption-indonesia/versions/1/Images/sample.jpg \
  --checkpoint output/saved_models/best_model.pth \
  --show
```

**Output**:
```
🔮 IMAGE CAPTIONING - INFERENCE
===============================

📚 Loading vocabulary...
✅ Loaded: Vocab size = 5432, Embedding shape = (5432, 300)

📂 Loading model from output/saved_models/best_model.pth...
✅ Model loaded (BLEU: 0.4523)

🖼️  Processing image: sample.jpg

✨ Generated Caption:
   seorang pria sedang duduk di taman
```

---

## ⚙️ Konfigurasi (config.py)

Edit `config.py` untuk mengubah hyperparameter:

```python
# Model
CNN_MODEL = 'vgg16'          # Options: 'vgg16', 'resnet50', 'resnet101'
HIDDEN_SIZE = 512            # LSTM hidden units
EMBEDDING_DIM = 300          # FastText dimension

# Training
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 5                 # Early stopping patience

# Data
TRAIN_VAL_SPLIT = 0.9        # 90% train, 10% val
MAX_CAPTION_LENGTH = 50
MIN_WORD_FREQ = 2            # Kata minimal muncul 2x
```

---

## 📊 Monitoring Training

Saat training, Anda akan melihat output seperti:

```
Epoch [1][0/150]    Batch Time 2.345 (2.345)    Loss 4.5123 (4.5123)
Epoch [1][100/150]  Batch Time 0.523 (0.612)    Loss 3.8765 (4.1234)

✅ Training Loss: 4.0234

📝 Sample Generated Caption:
   Reference: pria sedang bermain bola di lapangan
   Generated: pria di lapangan

📊 Validation Results:
   Loss: 3.7654
   BLEU-4: 0.1234

💾 Checkpoint saved: output/saved_models/checkpoint_epoch_1.pth
🏆 New best model saved (BLEU: 0.1234)
```

---

## 🎯 Tips & Best Practices

### 1. **FastText Model**
- **Wajib**: Download `cc.id.300.bin` sebelum training
- **Ukuran**: ~6.8 GB (pre-trained 2M+ kata Indonesia)
- **Alternative**: Bisa gunakan model Word2Vec lebih kecil (dengan edit `vocabulary.py`)

### 2. **GPU vs CPU**
- **GPU**: Training ~10-20x lebih cepat (recommended)
- **CPU**: Tetap bisa, tapi lambat (~2-3 jam per epoch)
- Auto-detect di `config.py`: `DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`

### 3. **Fine-Tuning**
Jika hasil kurang bagus, coba:
```python
# Di config.py atau train.py
encoder.fine_tune(True)  # Fine-tune CNN layers
decoder.fine_tune_embeddings = True  # Fine-tune FastText embeddings
```

### 4. **Augmentasi Data**
Edit `utils/dataset.py` untuk tambah augmentasi:
```python
transforms.ColorJitter(brightness=0.2, contrast=0.2)
transforms.RandomRotation(10)
```

### 5. **Beam Search**
Untuk hasil lebih baik, implementasi beam search di `decoder.py` (sudah ada placeholder).

---

## 📈 Expected Results

Dengan konfigurasi default:

| Epoch | Train Loss | Val Loss | BLEU-4 |
|-------|-----------|----------|--------|
| 1     | 4.23      | 3.87     | 0.12   |
| 10    | 2.45      | 2.31     | 0.28   |
| 20    | 1.67      | 1.89     | 0.38   |
| 30    | 1.23      | 1.65     | 0.45   |
| 40    | 0.98      | 1.52     | 0.48   |

**Target BLEU**: 0.40-0.50 (good), 0.50+ (excellent)

---

## 🐛 Troubleshooting

### Error: "FastText model not found"
```bash
# Download FastText model manual:
cd fasttext
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.bin.gz
gunzip cc.id.300.bin.gz
```

### Error: "CUDA out of memory"
```python
# Kurangi batch size di config.py:
BATCH_SIZE = 16  # atau 8
```

### Error: "ModuleNotFoundError: No module named 'torch'"
```bash
# Install dependencies:
pip install -r requirements.txt
```

### Caption hasil jelek / random
- **Solusi**: Training belum cukup lama (perlu 20-30 epoch)
- **Cek**: BLEU score harus > 0.30 untuk hasil layak

---

## 📚 Referensi

- **Paper**: "Show and Tell: A Neural Image Caption Generator" (Vinyals et al., 2015)
- **FastText**: https://fasttext.cc/
- **Dataset**: Kaggle - Image Caption Indonesia
- **Framework**: PyTorch

---

## 📧 Support

Jika ada masalah:
1. Cek `config.py` apakah path sudah benar
2. Pastikan FastText model sudah ter-download
3. Cek GPU/CPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

---

**Selamat mencoba! 🚀**
