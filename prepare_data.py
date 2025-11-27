"""
Preprocessing Script: Build Vocabulary & Embedding Matrix
==========================================================
Jalankan script ini SEBELUM training untuk prepare vocabulary
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import load_metadata
from utils.vocabulary import build_vocabulary_and_embeddings
import config

def main():
    """Main preprocessing pipeline"""
    print("\n" + "="*80)
    print("🔧 PREPROCESSING: Building Vocabulary & Embeddings")
    print("="*80)
    
    # Create output directories
    config.create_output_dirs()
    
    # Load all captions from metadata
    print("\n📂 Loading captions from metadata...")
    _, _, all_captions = load_metadata()
    
    print(f"✅ Loaded {len(all_captions)} captions")
    print(f"\nSample captions:")
    for i, cap in enumerate(all_captions[:5]):
        print(f"  {i+1}. {cap}")
    
    # Build vocabulary and embedding matrix
    vocab, embeddings = build_vocabulary_and_embeddings(
        all_captions,
        save_dir=config.VOCAB_DIR
    )
    
    print(f"\n" + "="*80)
    print("✅ PREPROCESSING SELESAI!")
    print(f"   Vocabulary size: {len(vocab)}")
    print(f"   Embedding matrix: {embeddings.shape}")
    print(f"   Saved to: {config.VOCAB_DIR}")
    print("="*80)
    print("\n💡 Next step: Jalankan training dengan: python train.py")

if __name__ == "__main__":
    main()
