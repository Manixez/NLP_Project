"""
INFERENCE: Generate Caption untuk Gambar Baru
==============================================
Load trained model dan generate caption untuk gambar baru
"""
import torch
from PIL import Image
import argparse
from pathlib import Path

import config
from models.encoder import EncoderCNN
from models.decoder import DecoderLSTM
from utils.vocabulary import load_vocabulary_and_embeddings
from utils.dataset import get_transform


def load_model(checkpoint_path, vocab, embedding_matrix):
    """
    Load trained model dari checkpoint
    
    Args:
        checkpoint_path: Path ke file .pth
        vocab: Vocabulary object
        embedding_matrix: Embedding matrix
    
    Returns:
        encoder, decoder (dalam eval mode)
    """
    print(f"📂 Loading model from {checkpoint_path}...")
    
    # Build models
    encoder = EncoderCNN(
        encoded_size=config.ENCODED_IMAGE_SIZE,
        fine_tune=False
    ).to(config.DEVICE)
    
    decoder = DecoderLSTM(
        embedding_matrix=embedding_matrix,
        vocab_size=len(vocab),
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        fine_tune_embeddings=False
    ).to(config.DEVICE)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    encoder.eval()
    decoder.eval()
    
    print(f"✅ Model loaded (BLEU: {checkpoint.get('bleu', 'N/A')})")
    
    return encoder, decoder


def generate_caption(image_path, encoder, decoder, vocab, transform=None):
    """
    Generate caption untuk satu gambar
    
    Args:
        image_path: Path ke gambar
        encoder: EncoderCNN
        decoder: DecoderLSTM
        vocab: Vocabulary
        transform: Image transformation
    
    Returns:
        caption_text: String caption
    """
    if transform is None:
        transform = get_transform(train=False)
    
    # Load dan preprocess gambar
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)  # (1, 3, 224, 224)
    
    # Generate caption
    with torch.no_grad():
        # Encode image
        encoder_out = encoder(image_tensor)  # (1, 14, 14, 512)
        
        # Decode to caption
        sampled_ids = decoder.sample(
            encoder_out,
            start_token=vocab(config.START_TOKEN),
            end_token=vocab(config.END_TOKEN),
            max_length=config.MAX_CAPTION_LENGTH
        )
    
    # Convert indices to words
    sampled_ids = sampled_ids[0].cpu().numpy()
    caption_words = []
    
    for word_id in sampled_ids:
        word = vocab.decode(word_id)
        if word == config.END_TOKEN:
            break
        if word != config.START_TOKEN and word != config.PAD_TOKEN:
            caption_words.append(word)
    
    caption_text = ' '.join(caption_words)
    return caption_text


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Generate caption untuk gambar')
    parser.add_argument('--image', type=str, required=True,
                       help='Path ke gambar input')
    parser.add_argument('--checkpoint', type=str,
                       default=str(config.MODELS_DIR / 'best_model.pth'),
                       help='Path ke model checkpoint')
    parser.add_argument('--show', action='store_true',
                       help='Tampilkan gambar dengan caption')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🔮 IMAGE CAPTIONING - INFERENCE")
    print("="*60)
    
    # Load vocabulary & embeddings
    print("\n📚 Loading vocabulary...")
    vocab, embedding_matrix = load_vocabulary_and_embeddings()
    
    # Load model
    encoder, decoder = load_model(args.checkpoint, vocab, embedding_matrix)
    
    # Generate caption
    print(f"\n🖼️  Processing image: {args.image}")
    caption = generate_caption(args.image, encoder, decoder, vocab)
    
    print(f"\n✨ Generated Caption:")
    print(f"   {caption}")
    print()
    
    # Show image jika diminta
    if args.show:
        try:
            from matplotlib import pyplot as plt
            
            img = Image.open(args.image)
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.title(f'Caption: {caption}', fontsize=14, pad=20)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("⚠️  Matplotlib tidak tersedia. Install: pip install matplotlib")


if __name__ == "__main__":
    main()
