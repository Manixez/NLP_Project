"""
Memory-Efficient Vocabulary Builder
====================================
Versi yang lebih hemat memory untuk sistem dengan RAM terbatas
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import load_metadata
from utils.vocabulary import Vocabulary
import config
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
import gc


def build_vocab_only():
    """Build vocabulary tanpa FastText dulu (hemat memory)"""
    print("\n" + "="*80)
    print("🔧 STEP 1/2: Building Vocabulary (Memory-Efficient Mode)")
    print("="*80)
    
    config.create_output_dirs()
    
    # Load captions
    print("\n📂 Loading captions...")
    _, _, all_captions = load_metadata()
    print(f"✅ Loaded {len(all_captions)} captions")
    
    # Build vocab
    print("\n📚 Building vocabulary...")
    vocab = Vocabulary()
    
    for caption in tqdm(all_captions, desc="Processing"):
        vocab.add_caption(caption)
    
    vocab.build_vocab(min_freq=config.MIN_WORD_FREQ)
    
    # Save vocabulary
    vocab_path = config.VOCAB_DIR / "vocab.pkl"
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    print(f"\n💾 Vocabulary saved to {vocab_path}")
    print(f"   Vocab size: {len(vocab)} words")
    
    # Clean up
    del all_captions
    gc.collect()
    
    return vocab


def build_embeddings_only():
    """Build embedding matrix dari vocab yang sudah ada (separate process)"""
    print("\n" + "="*80)
    print("🔧 STEP 2/2: Creating FastText Embedding Matrix")
    print("="*80)
    print("⚠️  WARNING: Proses ini akan pakai ~8-10 GB RAM!")
    print("💡 Tutup semua aplikasi lain dulu (browser, VSCode, dll)")
    print("\nPress ENTER to continue or Ctrl+C to cancel...")
    input()
    
    # Load vocab
    vocab_path = config.VOCAB_DIR / "vocab.pkl"
    print(f"\n📂 Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"✅ Loaded: {len(vocab)} words")
    
    # Load FastText (heavy!)
    print(f"\n📦 Loading FastText model (6.8 GB)...")
    print("   This may take 2-5 minutes...")
    
    from gensim.models.fasttext import load_facebook_model
    model = load_facebook_model(str(config.FASTTEXT_MODEL_PATH))
    print(f"✅ FastText loaded!")
    
    # Create embedding matrix
    print(f"\n🌉 Creating embedding matrix ({len(vocab)} x {config.EMBEDDING_DIM})...")
    embedding_matrix = np.zeros((len(vocab), config.EMBEDDING_DIM))
    
    found = 0
    for idx in tqdm(range(len(vocab)), desc="Building matrix"):
        word = vocab.decode(idx)
        
        if word in config.SPECIAL_TOKENS:
            embedding_matrix[idx] = np.random.randn(config.EMBEDDING_DIM) * 0.01
        else:
            try:
                embedding_matrix[idx] = model.wv[word]
                found += 1
            except KeyError:
                embedding_matrix[idx] = np.random.randn(config.EMBEDDING_DIM) * 0.01
    
    # Save
    embedding_path = config.VOCAB_DIR / "embedding_matrix.npy"
    np.save(embedding_path, embedding_matrix)
    
    coverage = (found / (len(vocab) - 4)) * 100
    print(f"\n💾 Embedding matrix saved to {embedding_path}")
    print(f"   Coverage: {coverage:.2f}% ({found}/{len(vocab)-4} words)")
    
    # Cleanup
    del model
    gc.collect()


def main():
    """Main pipeline dengan 2 step terpisah"""
    import sys
    
    if len(sys.argv) < 2:
        print("\n" + "="*80)
        print("MEMORY-EFFICIENT VOCABULARY BUILDER")
        print("="*80)
        print("\nUsage:")
        print("  Step 1 (light): python prepare_data_safe.py vocab")
        print("  Step 2 (heavy): python prepare_data_safe.py embeddings")
        print("  Both steps:     python prepare_data_safe.py all")
        print("\n💡 Run step 1 first, then close all apps before step 2")
        print("="*80)
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "vocab":
        vocab = build_vocab_only()
        print("\n✅ Step 1/2 SELESAI!")
        print("💡 Next: Tutup VSCode, lalu jalankan:")
        print("   python prepare_data_safe.py embeddings")
        
    elif mode == "embeddings":
        build_embeddings_only()
        print("\n✅ Step 2/2 SELESAI!")
        print("💡 Next: Jalankan training dengan: python train.py")
        
    elif mode == "all":
        print("⚠️  Mode 'all' akan pakai banyak RAM!")
        print("Press ENTER to continue or Ctrl+C to cancel...")
        input()
        vocab = build_vocab_only()
        print("\n" + "="*80)
        build_embeddings_only()
        print("\n✅ SEMUA SELESAI!")
    
    else:
        print(f"❌ Invalid mode: {mode}")
        print("Use: vocab, embeddings, or all")


if __name__ == "__main__":
    main()
