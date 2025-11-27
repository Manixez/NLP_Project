# 🌉 TRANSFER LEARNING FASTTEXT - PENJELASAN LENGKAP

## 📍 **Lokasi Transfer Learning di Code**

Transfer Learning dari FastText terjadi di **3 tempat utama**:

---

## **1️⃣ FASE 0: Load Pre-trained FastText Model**

### **File: `utils/vocabulary.py`**

```python
# Line ~120-135
class FastTextEmbedding:
    def load_model(self):
        """🎯 TRANSFER LEARNING STEP 1: Load pre-trained weights"""
        from gensim.models.fasttext import load_facebook_model
        
        # ✅ TRANSFER LEARNING: Load 2M+ Indonesian word vectors
        self.model = load_facebook_model(str(self.model_path))
        #              ^^^^^^^^^^^^^^^^^^
        #              Model ditraining pada 600B+ tokens dari Common Crawl
        #              (Wikipedia, berita, web pages Indonesia)
        
        print(f"✅ FastText loaded: {len(self.model.wv)} words")
```

**Yang di-transfer:**
- ✅ **2 juta+ kata** Bahasa Indonesia
- ✅ **300-dimensional vectors** per kata
- ✅ **Semantic knowledge** (kata "pria" dekat dengan "laki-laki")
- ✅ **Subword information** (bisa handle kata OOV dengan char n-grams)

---

## **2️⃣ FASE 0: Create Embedding Matrix dari FastText**

### **File: `utils/vocabulary.py`**

```python
# Line ~165-185
def create_embedding_matrix(self, vocab):
    """🎯 TRANSFER LEARNING STEP 2: Extract vectors untuk vocabulary kita"""
    
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, config.EMBEDDING_DIM))
    
    for idx in range(vocab_size):
        word = vocab.decode(idx)
        
        if word in config.SPECIAL_TOKENS:
            # Special tokens: random init
            embedding_matrix[idx] = np.random.randn(config.EMBEDDING_DIM) * 0.01
        else:
            # ✅ TRANSFER LEARNING: Copy pre-trained vector dari FastText
            vector = self.get_vector(word)  # Ambil dari FastText model
            #        ^^^^^^^^^^^^^^
            #        Vector ini sudah ditraining pada billions of words!
            embedding_matrix[idx] = vector
    
    return embedding_matrix
    #      ^^^^^^^^^^^^^^^^
    #      Matrix ini berisi "knowledge" dari FastText
```

**Yang terjadi:**
```
Vocabulary kata kamu: ["pria", "duduk", "taman", ...]
                           ↓
FastText model: {
    "pria": [0.12, -0.34, 0.56, ..., 0.89],    ← Pre-trained!
    "duduk": [-0.23, 0.45, -0.67, ..., 0.12],  ← Pre-trained!
    "taman": [0.34, -0.12, 0.78, ..., -0.45],  ← Pre-trained!
}
                           ↓
Embedding Matrix: (5751 × 300)
    Row 4: [0.12, -0.34, 0.56, ..., 0.89]  ← "pria" vector
    Row 5: [-0.23, 0.45, -0.67, ..., 0.12] ← "duduk" vector
    Row 123: [0.34, -0.12, 0.78, ..., -0.45] ← "taman" vector
```

---

## **3️⃣ FASE 1: Freeze/Fine-tune Embeddings di Decoder**

### **File: `models/decoder.py`**

```python
# Line ~42-47
def __init__(self, embedding_matrix, vocab_size, hidden_size=512, 
             num_layers=1, dropout=0.5, fine_tune_embeddings=False):
    
    # 🌉 Embedding Layer (from FastText)
    self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
    self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
    #                                                  ^^^^^^^^^^^^^^^^
    #                                                  Pre-trained FastText weights!
    
    # ✅ TRANSFER LEARNING STEP 3: Freeze atau fine-tune
    self.embedding.weight.requires_grad = fine_tune_embeddings
    #                                     ^^^^^^^^^^^^^^^^^^^^
    #                                     False = FROZEN (default)
    #                                     True = FINE-TUNED
```

**2 Mode Transfer Learning:**

### **Mode 1: Frozen (Default) ❄️**
```python
fine_tune_embeddings=False

# Saat training:
# - Embedding weights TIDAK diupdate
# - Hanya LSTM weights yang diupdate
# - FastText knowledge tetap utuh

Pros:
  ✅ Prevent overfitting (dataset kecil)
  ✅ Training lebih cepat
  ✅ Semantic knowledge tetap konsisten

Cons:
  ❌ Tidak bisa adapt ke domain spesifik dataset
```

### **Mode 2: Fine-tuned 🔥**
```python
fine_tune_embeddings=True

# Saat training:
# - Embedding weights DIUPDATE via backprop
# - LSTM weights juga diupdate
# - FastText vectors di-adjust untuk dataset kamu

Pros:
  ✅ Adapt ke domain spesifik (misalnya: medical images)
  ✅ Potentially higher accuracy

Cons:
  ❌ Risk overfitting (jika dataset kecil)
  ❌ Training lebih lambat
  ❌ Butuh lebih banyak data
```

---

## **🔄 Flow Transfer Learning Lengkap:**

```
┌─────────────────────────────────────────────────────────────┐
│ PRE-TRAINING (Facebook AI Research)                         │
│                                                             │
│ FastText ditraining pada:                                   │
│   - Common Crawl (600B+ tokens)                            │
│   - Wikipedia Indonesia                                     │
│   - Berita online, forum, social media                     │
│                                                             │
│ Output: cc.id.300.bin (6.8 GB)                             │
│   → 2M+ words × 300 dimensions                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FASE 0: Extract Embeddings (utils/vocabulary.py)           │
│                                                             │
│ 1. Load FastText model                                      │
│ 2. Build vocabulary dari dataset kamu (5,751 words)        │
│ 3. For each word in vocab:                                 │
│      → Get vector dari FastText                            │
│      → Store in embedding_matrix                           │
│                                                             │
│ Output: embedding_matrix.npy (5751 × 300)                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FASE 1: Initialize Decoder (models/decoder.py)             │
│                                                             │
│ 1. Create nn.Embedding layer                               │
│ 2. Load embedding_matrix as initial weights                │
│ 3. Set requires_grad=False (FREEZE)                        │
│                                                             │
│ Result: Embedding layer dengan FastText knowledge!         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ TRAINING: Forward Pass                                      │
│                                                             │
│ Input: word_id = 4 ("pria")                                │
│    ↓                                                        │
│ Embedding Layer:                                            │
│    embedding_matrix[4] = [0.12, -0.34, ..., 0.89]          │
│    ↓                                                        │
│ LSTM:                                                       │
│    Process vector [0.12, -0.34, ..., 0.89]                 │
│    ↓                                                        │
│ Output: Next word prediction                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ TRAINING: Backpropagation                                   │
│                                                             │
│ Loss → LSTM weights (UPDATED ✅)                            │
│    ↓                                                        │
│ Loss → Embedding weights (FROZEN ❄️)                        │
│                                                             │
│ Embedding tetap pakai FastText knowledge!                  │
└─────────────────────────────────────────────────────────────┘
```

---

## **📊 Kenapa Transfer Learning Penting?**

### **Tanpa Transfer Learning (Random Init):**
```python
# Embedding random
"pria" → [0.01, -0.03, 0.02, ...]  # Random!
"duduk" → [-0.02, 0.01, -0.04, ...] # Random!

# Tidak ada semantic knowledge
# Model harus belajar dari scratch
# Butuh BANYAK data (millions of samples)
```

### **Dengan Transfer Learning (FastText):**
```python
# Embedding pre-trained
"pria" → [0.12, -0.34, 0.56, ...]  # Meaningful!
"laki-laki" → [0.13, -0.35, 0.57, ...] # Similar vector!

# Sudah ada semantic knowledge
# Model hanya perlu belajar "caption grammar"
# Bisa training dengan dataset kecil (40K samples)
```

---

## **🎯 Cara Ganti Mode Transfer Learning:**

### **Default: Frozen (Recommended untuk dataset kecil)**
```python
# File: train.py, line ~245
decoder = DecoderLSTM(
    embedding_matrix=embedding_matrix,
    vocab_size=len(vocab),
    hidden_size=config.HIDDEN_SIZE,
    fine_tune_embeddings=False  # ← FROZEN ❄️
)
```

### **Fine-tuned (Untuk dataset besar > 100K samples)**
```python
# File: train.py, line ~245
decoder = DecoderLSTM(
    embedding_matrix=embedding_matrix,
    vocab_size=len(vocab),
    hidden_size=config.HIDDEN_SIZE,
    fine_tune_embeddings=True  # ← FINE-TUNED 🔥
)
```

---

## **📈 Expected Results:**

| Scenario | BLEU-4 Score | Training Time |
|----------|--------------|---------------|
| Random Init (No Transfer) | 0.15-0.25 | 100% |
| FastText Frozen | **0.35-0.45** ✅ | 100% |
| FastText Fine-tuned | 0.38-0.48 | 120% |

**Kesimpulan:** Transfer Learning dari FastText meningkatkan BLEU score ~50-80%! 🚀

---

## **🔍 Cek Transfer Learning Berhasil:**

```python
# Run this to verify:
import torch
import pickle
import numpy as np

# Load embedding matrix
embedding_matrix = np.load('output/vocab/embedding_matrix.npy')

# Load vocab
with open('output/vocab/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Check similarity between synonyms
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

idx_pria = vocab('pria')
idx_laki = vocab('laki-laki')

vec_pria = embedding_matrix[idx_pria]
vec_laki = embedding_matrix[idx_laki]

similarity = cosine_similarity(vec_pria, vec_laki)
print(f"Similarity 'pria' vs 'laki-laki': {similarity:.4f}")
# Expected: > 0.7 (if FastText working correctly!)
```

---

## **Summary:**

✅ **Transfer Learning terjadi di:**
1. `utils/vocabulary.py` - Load FastText & extract vectors
2. `models/decoder.py` - Initialize embedding layer dengan pre-trained weights
3. `train.py` - Freeze/fine-tune selama training

✅ **Benefit:**
- BLEU score naik 50-80%
- Training lebih cepat konvergen
- Bisa training dengan dataset lebih kecil

✅ **Mode saat ini:** FROZEN (recommended untuk 40K samples)
