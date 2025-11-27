"""
Model Parameter Analysis Script
================================
Script untuk menghitung jumlah parameter dan breakdown per layer
"""
import sys
sys.path.insert(0, '/home/manix/Documents/Semester 7/NLP/Kode')

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

import config
from models.encoder import EncoderCNN
from models.decoder import DecoderLSTM
from utils.vocabulary import load_vocabulary_and_embeddings


def count_parameters(model, model_name="Model"):
    """
    Hitung total parameters (trainable & non-trainable)
    
    Args:
        model: PyTorch model
        model_name: Nama model untuk display
    
    Returns:
        total_params, trainable_params
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*80}")
    print(f"📊 {model_name} - Parameter Summary")
    print(f"{'='*80}")
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    print(f"  Frozen parameters:     {total_params - trainable_params:,}")
    print(f"{'='*80}\n")
    
    return total_params, trainable_params


def analyze_layer_parameters(model, model_name="Model"):
    """
    Analisis detail parameter per layer
    
    Args:
        model: PyTorch model
        model_name: Nama model
    """
    print(f"\n{'='*80}")
    print(f"🔍 {model_name} - Detailed Layer Analysis")
    print(f"{'='*80}")
    print(f"{'Layer Name':<50} {'Shape':<25} {'Parameters':>12} {'Trainable':<10}")
    print(f"{'-'*80}")
    
    total = 0
    trainable_total = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        is_trainable = "✅ Yes" if param.requires_grad else "❄️  No"
        
        print(f"{name:<50} {str(tuple(param.shape)):<25} {num_params:>12,} {is_trainable:<10}")
        
        total += num_params
        if param.requires_grad:
            trainable_total += num_params
    
    print(f"{'-'*80}")
    print(f"{'TOTAL':<50} {'':<25} {total:>12,} {'':<10}")
    print(f"{'Trainable':<50} {'':<25} {trainable_total:>12,} {'':<10}")
    print(f"{'Frozen':<50} {'':<25} {total - trainable_total:>12,} {'':<10}")
    print(f"{'='*80}\n")


def analyze_lstm_breakdown(decoder):
    """
    Breakdown khusus untuk LSTM parameters
    
    Args:
        decoder: DecoderLSTM model
    """
    print(f"\n{'='*80}")
    print(f"🧠 LSTM Layer - Detailed Breakdown")
    print(f"{'='*80}")
    
    # LSTM parameters formula:
    # 4 gates (input, forget, cell, output)
    # Each gate: W_ih (input to hidden) + W_hh (hidden to hidden) + bias
    
    input_size = decoder.embedding_dim
    hidden_size = decoder.hidden_size
    num_layers = decoder.num_layers
    
    print(f"\nLSTM Configuration:")
    print(f"  Input size (embedding dim):  {input_size}")
    print(f"  Hidden size:                 {hidden_size}")
    print(f"  Number of layers:            {num_layers}")
    print(f"  Bidirectional:               No")
    
    print(f"\n{'Layer':<15} {'Component':<30} {'Shape':<25} {'Parameters':>15}")
    print(f"{'-'*90}")
    
    total_lstm = 0
    
    for layer_idx in range(num_layers):
        # Input size untuk layer pertama berbeda
        in_size = input_size if layer_idx == 0 else hidden_size
        
        # Weight input to hidden (4 gates)
        w_ih_shape = (4 * hidden_size, in_size)
        w_ih_params = 4 * hidden_size * in_size
        print(f"Layer {layer_idx+1:<8} {'weight_ih (W_ih)':<30} {str(w_ih_shape):<25} {w_ih_params:>15,}")
        total_lstm += w_ih_params
        
        # Weight hidden to hidden (4 gates)
        w_hh_shape = (4 * hidden_size, hidden_size)
        w_hh_params = 4 * hidden_size * hidden_size
        print(f"{'':15} {'weight_hh (W_hh)':<30} {str(w_hh_shape):<25} {w_hh_params:>15,}")
        total_lstm += w_hh_params
        
        # Bias input (4 gates)
        b_ih_shape = (4 * hidden_size,)
        b_ih_params = 4 * hidden_size
        print(f"{'':15} {'bias_ih (b_ih)':<30} {str(b_ih_shape):<25} {b_ih_params:>15,}")
        total_lstm += b_ih_params
        
        # Bias hidden (4 gates)
        b_hh_shape = (4 * hidden_size,)
        b_hh_params = 4 * hidden_size
        print(f"{'':15} {'bias_hh (b_hh)':<30} {str(b_hh_shape):<25} {b_hh_params:>15,}")
        total_lstm += b_hh_params
        
        print(f"{'-'*90}")
    
    print(f"{'TOTAL LSTM':<15} {'':<30} {'':<25} {total_lstm:>15,}")
    print(f"{'='*90}\n")
    
    # Explain LSTM gates
    print("📚 LSTM Gates Explanation:")
    print("  Each LSTM cell has 4 gates (i, f, c, o):")
    print("    • Input gate (i):   Controls new information")
    print("    • Forget gate (f):  Controls what to forget")
    print("    • Cell gate (c):    Candidate values")
    print("    • Output gate (o):  Controls output")
    print(f"\n  Formula per gate: y = W_ih @ x + b_ih + W_hh @ h + b_hh")
    print(f"  Total parameters per layer: 4 × (W_ih + W_hh + b_ih + b_hh)")
    print(f"{'='*80}\n")


def compare_configurations():
    """
    Compare different model configurations
    """
    print(f"\n{'='*80}")
    print(f"📊 Model Configuration Comparison")
    print(f"{'='*80}")
    
    configurations = [
        {"name": "Current (1 layer, 512 hidden)", "layers": 1, "hidden": 512},
        {"name": "2 layers, 512 hidden", "layers": 2, "hidden": 512},
        {"name": "1 layer, 768 hidden", "layers": 1, "hidden": 768},
        {"name": "2 layers, 256 hidden", "layers": 2, "hidden": 256},
    ]
    
    vocab_size = 5751  # Approximate
    embedding_dim = 300
    
    print(f"\n{'Configuration':<35} {'LSTM Params':>15} {'FC Params':>15} {'Total':>15}")
    print(f"{'-'*80}")
    
    for conf in configurations:
        num_layers = conf['layers']
        hidden_size = conf['hidden']
        
        # LSTM parameters
        lstm_params = 0
        for layer_idx in range(num_layers):
            in_size = embedding_dim if layer_idx == 0 else hidden_size
            # w_ih + w_hh + b_ih + b_hh (4 gates each)
            lstm_params += 4 * hidden_size * in_size  # W_ih
            lstm_params += 4 * hidden_size * hidden_size  # W_hh
            lstm_params += 4 * hidden_size  # b_ih
            lstm_params += 4 * hidden_size  # b_hh
        
        # FC parameters
        fc_params = hidden_size * vocab_size + vocab_size  # W + b
        
        total = lstm_params + fc_params
        
        marker = " ✅" if conf['layers'] == 1 and conf['hidden'] == 512 else ""
        print(f"{conf['name']:<35}{marker} {lstm_params:>15,} {fc_params:>15,} {total:>15,}")
    
    print(f"{'='*80}\n")


def memory_analysis(total_params):
    """
    Estimasi memory usage
    
    Args:
        total_params: Total number of parameters
    """
    print(f"\n{'='*80}")
    print(f"💾 Memory Usage Estimation")
    print(f"{'='*80}")
    
    # Float32 = 4 bytes per parameter
    # During training: need gradients, optimizer states (Adam: 2x params)
    
    param_memory_mb = (total_params * 4) / (1024 * 1024)
    gradient_memory_mb = param_memory_mb  # Same size for gradients
    optimizer_memory_mb = param_memory_mb * 2  # Adam: momentum + velocity
    activation_memory_mb = 500  # Approximate (depends on batch size)
    
    total_training_mb = param_memory_mb + gradient_memory_mb + optimizer_memory_mb + activation_memory_mb
    
    print(f"\n  Parameters:           {param_memory_mb:.2f} MB")
    print(f"  Gradients:            {gradient_memory_mb:.2f} MB")
    print(f"  Optimizer states:     {optimizer_memory_mb:.2f} MB")
    print(f"  Activations (est):    {activation_memory_mb:.2f} MB")
    print(f"  {'─'*50}")
    print(f"  Total (training):     {total_training_mb:.2f} MB (~{total_training_mb/1024:.2f} GB)")
    print(f"\n  Inference only:       {param_memory_mb:.2f} MB (~{param_memory_mb/1024:.2f} GB)")
    print(f"{'='*80}\n")


def main():
    """Main analysis pipeline"""
    print("\n" + "="*80)
    print("🔬 IMAGE CAPTIONING MODEL - PARAMETER ANALYSIS")
    print("="*80)
    
    # Load vocabulary untuk get embedding matrix
    print("\n📚 Loading vocabulary and embeddings...")
    vocab, embedding_matrix = load_vocabulary_and_embeddings()
    
    # Build models
    print("\n🏗️  Building models...")
    encoder = EncoderCNN(
        encoded_size=config.ENCODED_IMAGE_SIZE,
        fine_tune=False
    )
    
    decoder = DecoderLSTM(
        embedding_matrix=embedding_matrix,
        vocab_size=len(vocab),
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        fine_tune_embeddings=False
    )
    
    # Analyze Encoder
    encoder_total, encoder_trainable = count_parameters(encoder, "ENCODER (ResNet50)")
    analyze_layer_parameters(encoder, "ENCODER")
    
    # Analyze Decoder
    decoder_total, decoder_trainable = count_parameters(decoder, "DECODER (LSTM)")
    analyze_layer_parameters(decoder, "DECODER")
    
    # LSTM breakdown
    analyze_lstm_breakdown(decoder)
    
    # Total system
    total_params = encoder_total + decoder_total
    total_trainable = encoder_trainable + decoder_trainable
    
    print(f"\n{'='*80}")
    print(f"🎯 COMPLETE SYSTEM SUMMARY")
    print(f"{'='*80}")
    print(f"  Encoder parameters:        {encoder_total:>15,}")
    print(f"  Decoder parameters:        {decoder_total:>15,}")
    print(f"  {'─'*50}")
    print(f"  Total parameters:          {total_params:>15,}")
    print(f"  Trainable parameters:      {total_trainable:>15,}")
    print(f"  Frozen parameters:         {total_params - total_trainable:>15,}")
    print(f"\n  Trainable ratio:           {(total_trainable/total_params)*100:>14.2f}%")
    print(f"{'='*80}\n")
    
    # Configuration comparison
    compare_configurations()
    
    # Memory analysis
    memory_analysis(total_trainable)
    
    # Save report
    print("💾 Saving report to parameter_analysis.txt...")
    # You can redirect output: python analyze_parameters.py > parameter_analysis.txt
    
    print("✅ Analysis complete!\n")


if __name__ == "__main__":
    main()
