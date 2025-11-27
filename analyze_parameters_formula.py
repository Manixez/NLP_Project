"""
Model Parameter Calculator (Formula-based)
===========================================
Calculate parameters tanpa perlu load actual model
"""

def calculate_lstm_params(input_size, hidden_size, num_layers):
    """
    Calculate LSTM parameters
    
    Formula per layer:
    - W_ih: 4 × hidden × input (4 gates)
    - W_hh: 4 × hidden × hidden (4 gates)
    - b_ih: 4 × hidden
    - b_hh: 4 × hidden
    """
    total = 0
    
    print(f"\n{'='*90}")
    print(f"🧠 LSTM Parameter Calculation")
    print(f"{'='*90}")
    print(f"Configuration: {num_layers} layer(s), {hidden_size} hidden units")
    print(f"\n{'Layer':<10} {'Component':<25} {'Formula':<35} {'Parameters':>15}")
    print(f"{'-'*90}")
    
    for layer_idx in range(num_layers):
        in_size = input_size if layer_idx == 0 else hidden_size
        
        # W_ih (input to hidden, 4 gates)
        w_ih = 4 * hidden_size * in_size
        print(f"Layer {layer_idx+1:<5} {'W_ih (4 gates)':<25} {'4 × %d × %d' % (hidden_size, in_size):<35} {w_ih:>15,}")
        total += w_ih
        
        # W_hh (hidden to hidden, 4 gates)
        w_hh = 4 * hidden_size * hidden_size
        print(f"{'':10} {'W_hh (4 gates)':<25} {'4 × %d × %d' % (hidden_size, hidden_size):<35} {w_hh:>15,}")
        total += w_hh
        
        # b_ih (bias input, 4 gates)
        b_ih = 4 * hidden_size
        print(f"{'':10} {'b_ih (4 gates)':<25} {'4 × %d' % hidden_size:<35} {b_ih:>15,}")
        total += b_ih
        
        # b_hh (bias hidden, 4 gates)
        b_hh = 4 * hidden_size
        print(f"{'':10} {'b_hh (4 gates)':<25} {'4 × %d' % hidden_size:<35} {b_hh:>15,}")
        total += b_hh
        
        layer_total = w_ih + w_hh + b_ih + b_hh
        print(f"{'':10} {'Layer subtotal':<25} {'':<35} {layer_total:>15,}")
        print(f"{'-'*90}")
    
    print(f"{'TOTAL':<10} {'':<25} {'':<35} {total:>15,}")
    print(f"{'='*90}\n")
    
    return total


def calculate_embedding_params(vocab_size, embedding_dim):
    """Calculate embedding layer parameters"""
    params = vocab_size * embedding_dim
    print(f"\n📚 Embedding Layer:")
    print(f"  Formula: vocab_size × embedding_dim")
    print(f"  Calculation: {vocab_size:,} × {embedding_dim}")
    print(f"  Parameters: {params:,}")
    print(f"  Status: ❄️  FROZEN (not trained)\n")
    return params


def calculate_fc_params(input_size, output_size):
    """Calculate fully connected layer parameters"""
    weights = input_size * output_size
    bias = output_size
    total = weights + bias
    
    print(f"\n🔗 Fully Connected Layer:")
    print(f"  Weights: {input_size:,} × {output_size:,} = {weights:,}")
    print(f"  Bias: {output_size:,}")
    print(f"  Total: {total:,}\n")
    return total


def calculate_resnet50_params():
    """Approximate ResNet50 parameters"""
    # ResNet50 has ~25M parameters
    # But we freeze it, so not trainable
    total = 25_557_032  # Exact ResNet50 params
    print(f"\n🖼️  ResNet50 Encoder:")
    print(f"  Total parameters: {total:,}")
    print(f"  Status: ❄️  FROZEN (pretrained, not trained)")
    print(f"  Trainable: 0\n")
    return total, 0  # total, trainable


def analyze_decoder(vocab_size=5751, embedding_dim=300, hidden_size=512, num_layers=1):
    """Complete decoder analysis"""
    print(f"\n{'='*90}")
    print(f"📊 DECODER ANALYSIS")
    print(f"{'='*90}")
    
    # Embedding
    embedding_params = calculate_embedding_params(vocab_size, embedding_dim)
    
    # LSTM
    lstm_params = calculate_lstm_params(embedding_dim, hidden_size, num_layers)
    
    # FC Layer
    fc_params = calculate_fc_params(hidden_size, vocab_size)
    
    # Init layers (h0, c0)
    init_h_params = 2048 * hidden_size  # ResNet50: 2048
    init_c_params = 2048 * hidden_size
    init_total = init_h_params + init_c_params
    print(f"\n🎯 Initialization Layers:")
    print(f"  init_h: 2048 × {hidden_size} = {init_h_params:,}")
    print(f"  init_c: 2048 × {hidden_size} = {init_c_params:,}")
    print(f"  Total: {init_total:,}\n")
    
    # Decoder total
    decoder_trainable = lstm_params + fc_params + init_total
    decoder_total = embedding_params + decoder_trainable
    
    print(f"\n{'='*90}")
    print(f"DECODER SUMMARY:")
    print(f"{'-'*90}")
    print(f"  Embedding (frozen):        {embedding_params:>15,}")
    print(f"  LSTM:                      {lstm_params:>15,}")
    print(f"  Fully Connected:           {fc_params:>15,}")
    print(f"  Init Layers (h0, c0):      {init_total:>15,}")
    print(f"{'-'*90}")
    print(f"  Total parameters:          {decoder_total:>15,}")
    print(f"  Trainable parameters:      {decoder_trainable:>15,}")
    print(f"  Frozen parameters:         {embedding_params:>15,}")
    print(f"{'='*90}\n")
    
    return decoder_total, decoder_trainable


def compare_configurations():
    """Compare different model configurations"""
    print(f"\n{'='*90}")
    print(f"🔄 MODEL CONFIGURATION COMPARISON")
    print(f"{'='*90}\n")
    
    configs = [
        {"name": "Current (1 layer, 512 hidden)", "layers": 1, "hidden": 512},
        {"name": "2 layers, 512 hidden", "layers": 2, "hidden": 512},
        {"name": "1 layer, 768 hidden", "layers": 1, "hidden": 768},
        {"name": "2 layers, 256 hidden", "layers": 2, "hidden": 256},
        {"name": "1 layer, 1024 hidden", "layers": 1, "hidden": 1024},
    ]
    
    vocab_size = 5751
    embedding_dim = 300
    
    print(f"{'Configuration':<40} {'LSTM':>15} {'FC':>15} {'Init':>12} {'Total':>15}")
    print(f"{'-'*90}")
    
    for conf in configs:
        layers = conf['layers']
        hidden = conf['hidden']
        
        # LSTM
        lstm = 0
        for i in range(layers):
            in_size = embedding_dim if i == 0 else hidden
            lstm += 4 * hidden * (in_size + hidden + 2)
        
        # FC
        fc = hidden * vocab_size + vocab_size
        
        # Init
        init = 2048 * hidden * 2
        
        total = lstm + fc + init
        
        marker = " ✅" if layers == 1 and hidden == 512 else ""
        print(f"{conf['name']:<40}{marker} {lstm:>15,} {fc:>15,} {init:>12,} {total:>15,}")
    
    print(f"{'='*90}\n")


def memory_estimate(trainable_params):
    """Estimate memory usage"""
    print(f"\n{'='*90}")
    print(f"💾 MEMORY USAGE ESTIMATION")
    print(f"{'='*90}\n")
    
    # Float32 = 4 bytes
    param_mb = (trainable_params * 4) / (1024 * 1024)
    grad_mb = param_mb
    optimizer_mb = param_mb * 2  # Adam: momentum + variance
    activation_mb = 500  # Approximate
    
    total_mb = param_mb + grad_mb + optimizer_mb + activation_mb
    
    print(f"  Parameters:              {param_mb:>10.2f} MB")
    print(f"  Gradients:               {grad_mb:>10.2f} MB")
    print(f"  Optimizer states (Adam): {optimizer_mb:>10.2f} MB")
    print(f"  Activations (batch=32):  {activation_mb:>10.2f} MB")
    print(f"  {'-'*60}")
    print(f"  Total (training):        {total_mb:>10.2f} MB ({total_mb/1024:.2f} GB)")
    print(f"\n  Inference only:          {param_mb:>10.2f} MB ({param_mb/1024:.2f} GB)")
    print(f"{'='*90}\n")


def main():
    """Main analysis"""
    print("\n" + "="*90)
    print("🔬 IMAGE CAPTIONING MODEL - PARAMETER ANALYSIS (Formula-Based)")
    print("="*90)
    
    # Configuration
    vocab_size = 5751
    embedding_dim = 300
    hidden_size = 512
    num_layers = 1
    
    print(f"\n📋 Current Configuration:")
    print(f"  Vocabulary size:     {vocab_size:,}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  LSTM hidden size:    {hidden_size}")
    print(f"  LSTM layers:         {num_layers}")
    print(f"  CNN Encoder:         ResNet50 (frozen)")
    
    # Encoder
    encoder_total, encoder_trainable = calculate_resnet50_params()
    
    # Decoder
    decoder_total, decoder_trainable = analyze_decoder(
        vocab_size, embedding_dim, hidden_size, num_layers
    )
    
    # System total
    system_total = encoder_total + decoder_total
    system_trainable = encoder_trainable + decoder_trainable
    
    print(f"\n{'='*90}")
    print(f"🎯 COMPLETE SYSTEM SUMMARY")
    print(f"{'='*90}")
    print(f"  Encoder (ResNet50):        {encoder_total:>15,} ({encoder_trainable:,} trainable)")
    print(f"  Decoder (LSTM):            {decoder_total:>15,} ({decoder_trainable:,} trainable)")
    print(f"  {'-'*60}")
    print(f"  Total parameters:          {system_total:>15,}")
    print(f"  Trainable parameters:      {system_trainable:>15,}")
    print(f"  Frozen parameters:         {system_total - system_trainable:>15,}")
    print(f"\n  Trainable ratio:           {(system_trainable/system_total)*100:>14.2f}%")
    print(f"{'='*90}\n")
    
    # Comparison
    compare_configurations()
    
    # Memory
    memory_estimate(system_trainable)
    
    print("✅ Analysis complete!")
    print("\n💡 To see actual model parameters, run: python train.py (will show during model build)")
    print("💡 To save this report: python analyze_parameters_formula.py > parameter_report.txt\n")


if __name__ == "__main__":
    main()
