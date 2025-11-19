"""
FASE 1, 2, 3, 4: Training Pipeline
===================================
Main training loop dengan:
- Teacher Forcing (Fase 1)
- Validation & Generation (Fase 2)
- BLEU Evaluation (Fase 3)
- Model Saving (Fase 4)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import time
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from pathlib import Path

import config
from models.encoder import EncoderCNN
from models.decoder import DecoderLSTM
from utils.dataset import prepare_dataloaders
from utils.vocabulary import load_vocabulary_and_embeddings


class AverageMeter:
    """Utility untuk track average nilai (loss, BLEU, dll)"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(encoder, decoder, dataloader, criterion, optimizer, epoch, vocab):
    """
    🎓 FASE 1: Training satu epoch dengan Teacher Forcing
    
    Args:
        encoder: EncoderCNN model
        decoder: DecoderLSTM model
        dataloader: Training DataLoader
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer
        epoch: Current epoch number
        vocab: Vocabulary object
    
    Returns:
        avg_loss: Average loss untuk epoch ini
    """
    encoder.train()
    decoder.train()
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    
    start = time.time()
    
    for i, (images, captions, lengths) in enumerate(dataloader):
        # Move to device
        images = images.to(config.DEVICE)
        captions = captions.to(config.DEVICE)
        
        # Forward pass
        # 1. Encoder: Extract CNN features
        encoder_out = encoder(images)  # (batch, 14, 14, 512)
        
        # 2. Decoder: Predict captions dengan teacher forcing
        predictions = decoder(encoder_out, captions, lengths)  # (batch, seq_len-1, vocab_size)
        
        # Calculate loss
        # Target: captions[:, 1:] (skip <start>, include <end>)
        # Predictions: output dari LSTM untuk setiap timestep
        
        # Flatten untuk cross entropy
        # predictions: (batch * seq_len, vocab_size)
        # targets: (batch * seq_len)
        targets = captions[:, 1:]  # Skip <start> token
        
        # Pack untuk handle variable lengths
        # Kita hitung loss hanya untuk non-padding tokens
        targets_packed = nn.utils.rnn.pack_padded_sequence(
            targets, [l - 1 for l in lengths], batch_first=True, enforce_sorted=True
        ).data
        
        predictions_packed = nn.utils.rnn.pack_padded_sequence(
            predictions, [l - 1 for l in lengths], batch_first=True, enforce_sorted=True
        ).data
        
        loss = criterion(predictions_packed, targets_packed)
        
        # Backward & optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        clip_grad_norm_(decoder.parameters(), config.GRADIENT_CLIP)
        if encoder.fine_tune:
            clip_grad_norm_(encoder.parameters(), config.GRADIENT_CLIP)
        
        optimizer.step()
        
        # Track metrics
        losses.update(loss.item(), sum([l - 1 for l in lengths]))
        batch_time.update(time.time() - start)
        start = time.time()
        
        # Print status
        if i % config.PRINT_FREQ == 0:
            print(f'Epoch [{epoch}][{i}/{len(dataloader)}]\t'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')
    
    return losses.avg


def validate(encoder, decoder, dataloader, criterion, vocab):
    """
    📝 FASE 2 & 3: Validation dengan Caption Generation & BLEU Evaluation
    
    Args:
        encoder: EncoderCNN model
        decoder: DecoderLSTM model
        dataloader: Validation DataLoader
        criterion: Loss function
        vocab: Vocabulary object
    
    Returns:
        avg_loss: Average validation loss
        bleu4: BLEU-4 score
    """
    encoder.eval()
    decoder.eval()
    
    losses = AverageMeter()
    
    references = []  # List of reference captions (ground truth)
    hypotheses = []  # List of generated captions (predictions)
    
    with torch.no_grad():
        for images, captions, lengths in dataloader:
            images = images.to(config.DEVICE)
            captions = captions.to(config.DEVICE)
            
            # 1. Encoder
            encoder_out = encoder(images)
            
            # 2. Decoder - Teacher forcing untuk loss
            predictions = decoder(encoder_out, captions, lengths)
            
            targets = captions[:, 1:]
            targets_packed = nn.utils.rnn.pack_padded_sequence(
                targets, [l - 1 for l in lengths], batch_first=True, enforce_sorted=True
            ).data
            predictions_packed = nn.utils.rnn.pack_padded_sequence(
                predictions, [l - 1 for l in lengths], batch_first=True, enforce_sorted=True
            ).data
            
            loss = criterion(predictions_packed, targets_packed)
            losses.update(loss.item(), sum([l - 1 for l in lengths]))
            
            # 3. Generate captions (untuk BLEU)
            sampled_ids = decoder.sample(
                encoder_out,
                start_token=vocab(config.START_TOKEN),
                end_token=vocab(config.END_TOKEN),
                max_length=config.MAX_CAPTION_LENGTH
            )
            
            # Convert ke list of words
            sampled_ids = sampled_ids.cpu().numpy()
            
            for i in range(len(images)):
                # Generated caption
                sampled_caption = []
                for word_id in sampled_ids[i]:
                    word = vocab.decode(word_id)
                    if word == config.END_TOKEN:
                        break
                    if word != config.START_TOKEN and word != config.PAD_TOKEN:
                        sampled_caption.append(word)
                
                hypotheses.append(sampled_caption)
                
                # Reference caption (ground truth)
                # Note: idealnya kita punya multiple references per image
                # Untuk simplicity, kita pakai satu reference
                ref_caption = []
                for word_id in captions[i].cpu().numpy():
                    word = vocab.decode(word_id)
                    if word == config.END_TOKEN:
                        break
                    if word != config.START_TOKEN and word != config.PAD_TOKEN:
                        ref_caption.append(word)
                
                references.append([ref_caption])  # Wrap in list (bisa multiple refs)
    
    # 📊 FASE 3: Calculate BLEU-4
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    # Print sample
    print(f'\n📝 Sample Generated Caption:')
    print(f'   Reference: {" ".join(references[0][0])}')
    print(f'   Generated: {" ".join(hypotheses[0])}')
    
    return losses.avg, bleu4


def train(resume_from=None):
    """
    🏋️ MAIN TRAINING PIPELINE
    
    Args:
        resume_from: Path ke checkpoint untuk resume training
    """
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING PIPELINE")
    print("="*80)
    
    # Create output directories
    config.create_output_dirs()
    
    # Load vocabulary & embeddings
    print("\n📚 Loading Vocabulary & Embeddings...")
    vocab, embedding_matrix = load_vocabulary_and_embeddings()
    
    # Prepare dataloaders
    print("\n📦 Preparing DataLoaders...")
    train_loader, val_loader = prepare_dataloaders(vocab)
    
    # Build models
    print("\n🏗️  Building Models...")
    encoder = EncoderCNN(
        encoded_size=config.ENCODED_IMAGE_SIZE,
        fine_tune=False  # Freeze CNN initially
    ).to(config.DEVICE)
    
    decoder = DecoderLSTM(
        embedding_matrix=embedding_matrix,
        vocab_size=len(vocab),
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        fine_tune_embeddings=False  # Freeze embeddings
    ).to(config.DEVICE)
    
    print(f"   Encoder: {config.CNN_MODEL}")
    print(f"   Decoder: LSTM ({config.HIDDEN_SIZE} units)")
    print(f"   Vocabulary: {len(vocab)} words")
    print(f"   Device: {config.DEVICE}")
    
    # Loss & Optimizer
    # Ignore padding index dalam loss
    criterion = nn.CrossEntropyLoss(ignore_index=vocab(config.PAD_TOKEN))
    
    # Optimizer hanya untuk decoder (encoder frozen)
    optimizer = optim.Adam(
        decoder.parameters(),
        lr=config.LEARNING_RATE
    )
    
    # Resume dari checkpoint jika ada
    start_epoch = 0
    best_bleu = 0.0
    
    if resume_from and Path(resume_from).exists():
        print(f"\n♻️  Resuming from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=config.DEVICE)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_bleu = checkpoint.get('bleu', 0.0)
        print(f"   Resumed from epoch {start_epoch}, best BLEU: {best_bleu:.4f}")
    
    # Training loop
    print("\n" + "="*80)
    print("🎓 FASE 1-4: TRAINING LOOP")
    print("="*80)
    
    patience_counter = 0
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"{'='*80}")
        
        # 🎓 FASE 1: Training
        train_loss = train_one_epoch(
            encoder, decoder, train_loader,
            criterion, optimizer, epoch + 1, vocab
        )
        
        print(f'\n✅ Training Loss: {train_loss:.4f}')
        
        # 📝 FASE 2 & 3: Validation & BLEU
        val_loss, bleu4 = validate(encoder, decoder, val_loader, criterion, vocab)
        
        print(f'\n📊 Validation Results:')
        print(f'   Loss: {val_loss:.4f}')
        print(f'   BLEU-4: {bleu4:.4f}')
        
        # 🏆 FASE 4: Save best model
        is_best = bleu4 > best_bleu
        best_bleu = max(bleu4, best_bleu)
        
        if (epoch + 1) % config.SAVE_FREQ == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'bleu': bleu4,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            
            save_path = config.MODELS_DIR / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, save_path)
            print(f'\n💾 Checkpoint saved: {save_path}')
            
            if is_best:
                best_path = config.MODELS_DIR / 'best_model.pth'
                torch.save(checkpoint, best_path)
                print(f'🏆 New best model saved: {best_path} (BLEU: {bleu4:.4f})')
                patience_counter = 0
            else:
                patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f'\n⏹️  Early stopping triggered (patience: {config.PATIENCE})')
            print(f'   Best BLEU: {best_bleu:.4f}')
            break
    
    print("\n" + "="*80)
    print("🎉 TRAINING SELESAI!")
    print(f"   Best BLEU-4: {best_bleu:.4f}")
    print(f"   Best model: {config.MODELS_DIR / 'best_model.pth'}")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    resume_checkpoint = None
    if len(sys.argv) > 1:
        resume_checkpoint = sys.argv[1]
    
    train(resume_from=resume_checkpoint)
