"""
FASE 0: Vocabulary Builder & FastText Embedding Matrix
=======================================================
Membuat kamus kata dan matriks embedding dari FastText pre-trained
"""
import pickle
import numpy as np
from collections import Counter
from pathlib import Path
import gensim.downloader as api
from tqdm import tqdm

import config


class Vocabulary:
    """
    Kelas untuk membuat dan mengelola vocabulary (kamus kata)
    """
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # Tambahkan special tokens
        for idx, token in enumerate(config.SPECIAL_TOKENS):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
    
    def add_word(self, word):
        """Tambahkan kata ke vocabulary"""
        self.word_freq[word] += 1
    
    def add_caption(self, caption):
        """Tambahkan semua kata dari caption"""
        tokens = caption.lower().split()
        for token in tokens:
            self.add_word(token)
    
    def build_vocab(self, min_freq=1):
        """
        Finalisasi vocabulary dengan filter frekuensi minimum
        
        Args:
            min_freq: Kata harus muncul minimal N kali untuk masuk vocab
        """
        idx = len(config.SPECIAL_TOKENS)
        
        for word, freq in self.word_freq.items():
            if freq >= min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"✅ Vocabulary built: {len(self.word2idx)} words")
        print(f"   - Special tokens: {len(config.SPECIAL_TOKENS)}")
        print(f"   - Unique words (freq >= {min_freq}): {len(self.word2idx) - len(config.SPECIAL_TOKENS)}")
    
    def __len__(self):
        return len(self.word2idx)
    
    def __call__(self, word):
        """Konversi kata ke indeks (return UNK jika tidak ada)"""
        return self.word2idx.get(word, self.word2idx[config.UNK_TOKEN])
    
    def decode(self, idx):
        """Konversi indeks ke kata"""
        return self.idx2word.get(idx, config.UNK_TOKEN)
    
    def encode_caption(self, caption):
        """
        Konversi caption (string) ke list indeks
        
        Args:
            caption: "pria duduk di taman"
        
        Returns:
            [1, 45, 67, 89, 123, 2]  # <start> pria duduk di taman <end>
        """
        tokens = [config.START_TOKEN] + caption.lower().split() + [config.END_TOKEN]
        return [self(token) for token in tokens]
    
    def decode_caption(self, indices):
        """
        Konversi list indeks ke caption (string)
        
        Args:
            indices: [1, 45, 67, 89, 123, 2]
        
        Returns:
            "pria duduk di taman"
        """
        words = []
        for idx in indices:
            word = self.decode(idx)
            if word == config.END_TOKEN:
                break
            if word != config.START_TOKEN and word != config.PAD_TOKEN:
                words.append(word)
        return ' '.join(words)


class FastTextEmbedding:
    """
    Kelas untuk memuat FastText dan membuat embedding matrix
    """
    
    def __init__(self, model_path=None):
        """
        Args:
            model_path: Path ke file .bin FastText (opsional)
        """
        self.model = None
        self.model_path = model_path
    
    def load_model(self):
        """Muat FastText pre-trained model"""
        if self.model_path and Path(self.model_path).exists():
            print(f"📦 Loading FastText from {self.model_path}...")
            # Untuk file .bin lokal, gunakan gensim.models.fasttext
            from gensim.models.fasttext import load_facebook_model
            self.model = load_facebook_model(str(self.model_path))
            print(f"✅ FastText loaded: {len(self.model.wv)} words")
        else:
            print("📦 Downloading FastText Indonesian model (ini mungkin lama)...")
            # Alternative: gunakan model dari gensim downloader (lebih kecil)
            # Note: Untuk Indonesian yang optimal, download manual cc.id.300.bin
            print("⚠️  Model FastText tidak ditemukan!")
            print(f"   Download dari: https://fasttext.cc/docs/en/crawl-vectors.html")
            print(f"   Simpan di: {config.FASTTEXT_MODEL_PATH}")
            print(f"   File: cc.id.300.bin (6.8 GB)")
            raise FileNotFoundError(f"FastText model not found at {self.model_path}")
    
    def get_vector(self, word):
        """
        Ambil vektor untuk satu kata
        
        Returns:
            np.array dengan shape (300,) atau None jika tidak ada
        """
        if self.model is None:
            self.load_model()
        
        try:
            return self.model.wv[word]
        except KeyError:
            # Jika kata tidak ada, return vektor random kecil
            return np.random.randn(config.EMBEDDING_DIM) * 0.01
    
    def create_embedding_matrix(self, vocab):
        """
        🌉 JEMBATAN: Buat Embedding Matrix dari Vocabulary + FastText
        
        Args:
            vocab: Vocabulary object
        
        Returns:
            np.array dengan shape (vocab_size, embedding_dim)
        """
        if self.model is None:
            self.load_model()
        
        vocab_size = len(vocab)
        embedding_matrix = np.zeros((vocab_size, config.EMBEDDING_DIM))
        
        print(f"\n🌉 Creating Embedding Matrix ({vocab_size} x {config.EMBEDDING_DIM})...")
        
        found_count = 0
        for idx in tqdm(range(vocab_size), desc="Building matrix"):
            word = vocab.decode(idx)
            
            # Special tokens: random small vectors
            if word in config.SPECIAL_TOKENS:
                embedding_matrix[idx] = np.random.randn(config.EMBEDDING_DIM) * 0.01
            else:
                vector = self.get_vector(word)
                embedding_matrix[idx] = vector
                if word in self.model.wv:
                    found_count += 1
        
        coverage = (found_count / (vocab_size - len(config.SPECIAL_TOKENS))) * 100
        print(f"✅ Embedding Matrix created!")
        print(f"   - Vocab coverage: {coverage:.2f}% ({found_count}/{vocab_size - len(config.SPECIAL_TOKENS)} words)")
        
        return embedding_matrix


def build_vocabulary_and_embeddings(captions_list, save_dir=None):
    """
    FASE 0 - MAIN PIPELINE:
    1. Build vocabulary dari semua captions
    2. Load FastText
    3. Create embedding matrix
    
    Args:
        captions_list: List of all captions (strings)
        save_dir: Directory untuk save vocab & embedding
    
    Returns:
        vocab, embedding_matrix
    """
    save_dir = Path(save_dir) if save_dir else config.VOCAB_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("🚀 FASE 0: PERSIAPAN - Building Vocabulary & Embeddings")
    print("="*60)
    
    # Step 1: Build Vocabulary
    print("\n📚 Step 1: Building Vocabulary...")
    vocab = Vocabulary()
    
    for caption in tqdm(captions_list, desc="Processing captions"):
        vocab.add_caption(caption)
    
    vocab.build_vocab(min_freq=config.MIN_WORD_FREQ)
    
    # Save vocabulary
    vocab_path = save_dir / "vocab.pkl"
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"💾 Vocabulary saved to {vocab_path}")
    
    # Step 2 & 3: Load FastText & Create Embedding Matrix
    print("\n🌉 Step 2-3: Creating FastText Embedding Matrix...")
    fasttext = FastTextEmbedding(model_path=config.FASTTEXT_MODEL_PATH)
    embedding_matrix = fasttext.create_embedding_matrix(vocab)
    
    # Save embedding matrix
    embedding_path = save_dir / "embedding_matrix.npy"
    np.save(embedding_path, embedding_matrix)
    print(f"💾 Embedding matrix saved to {embedding_path}")
    
    print("\n✅ FASE 0 SELESAI!")
    return vocab, embedding_matrix


def load_vocabulary_and_embeddings(load_dir=None):
    """
    Load vocabulary dan embedding matrix yang sudah disimpan
    
    Returns:
        vocab, embedding_matrix
    """
    load_dir = Path(load_dir) if load_dir else config.VOCAB_DIR
    
    vocab_path = load_dir / "vocab.pkl"
    embedding_path = load_dir / "embedding_matrix.npy"
    
    print(f"📂 Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    print(f"📂 Loading embedding matrix from {embedding_path}...")
    embedding_matrix = np.load(embedding_path)
    
    print(f"✅ Loaded: Vocab size = {len(vocab)}, Embedding shape = {embedding_matrix.shape}")
    return vocab, embedding_matrix


if __name__ == "__main__":
    # Test dengan dummy data
    print("Testing Vocabulary Builder...")
    
    dummy_captions = [
        "pria duduk di taman",
        "kucing bermain dengan bola",
        "pria membaca buku di taman",
        "anak kecil bermain bola"
    ]
    
    vocab, embeddings = build_vocabulary_and_embeddings(
        dummy_captions,
        save_dir=config.VOCAB_DIR
    )
    
    # Test encoding/decoding
    test_caption = "pria duduk di taman"
    encoded = vocab.encode_caption(test_caption)
    decoded = vocab.decode_caption(encoded)
    
    print(f"\n🧪 Test Encoding/Decoding:")
    print(f"   Original: {test_caption}")
    print(f"   Encoded: {encoded}")
    print(f"   Decoded: {decoded}")
