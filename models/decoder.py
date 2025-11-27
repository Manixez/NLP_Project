"""
FASE 1: LSTM Decoder dengan FastText Embeddings
================================================
Generate caption word-by-word menggunakan LSTM + Teacher Forcing
"""
import torch
import torch.nn as nn
import numpy as np

import config


class DecoderLSTM(nn.Module):
    """
    LSTM Decoder untuk generate caption
    
    Menggunakan:
    - Pre-trained FastText embeddings (frozen atau fine-tunable)
    - Teacher forcing saat training
    - Greedy/Beam search saat inference
    """
    
    def __init__(self, embedding_matrix, vocab_size, hidden_size=512, 
                 num_layers=1, dropout=0.5, fine_tune_embeddings=False):
        """
        Args:
            embedding_matrix: np.array (vocab_size, embedding_dim) dari FastText
            vocab_size: Ukuran vocabulary
            hidden_size: LSTM hidden units
            num_layers: Jumlah LSTM layers
            dropout: Dropout probability
            fine_tune_embeddings: True jika ingin fine-tune embeddings
        """
        super(DecoderLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_matrix.shape[1]
        
        # 🌉 Embedding Layer (from FastText)
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        
        # Freeze atau fine-tune embeddings
        self.embedding.weight.requires_grad = fine_tune_embeddings
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer (LSTM hidden -> vocabulary)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Layer untuk init LSTM hidden state dari CNN features
        # CNN output ResNet101: (batch, 14, 14, 2048) -> flatten -> init hidden
        # Kita akan terima global average pooled features: (batch, feature_dim)
        
        self.init_h = nn.Linear(2048, hidden_size)  # Init h0 (ResNet101: 2048)
        self.init_c = nn.Linear(2048, hidden_size)  # Init c0 (ResNet101: 2048)
    
    def init_hidden_state(self, encoder_out):
        """
        Inisialisasi LSTM hidden state dari CNN features
        
        Args:
            encoder_out: Tensor (batch_size, encoded_size, encoded_size, feature_dim)
                        ResNet50: (batch_size, 14, 14, 2048)
                        Contoh: (32, 14, 14, 512)
        
        Returns:
            h0: (num_layers, batch_size, hidden_size)
            c0: (num_layers, batch_size, hidden_size)
        """
        batch_size = encoder_out.size(0)
        
        # Global average pooling: (batch, 14, 14, 512) -> (batch, 512)
        mean_encoder_out = encoder_out.mean(dim=[1, 2])  # Average over spatial dims
        
        # Generate h0 dan c0
        h = self.init_h(mean_encoder_out)  # (batch, hidden_size)
        c = self.init_c(mean_encoder_out)  # (batch, hidden_size)
        
        # Expand untuk num_layers: (num_layers, batch, hidden_size)
        h = h.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = c.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        return h, c
    
    def forward(self, encoder_out, captions, lengths):
        """
        🎓 TRAINING MODE dengan Teacher Forcing
        
        Args:
            encoder_out: Tensor (batch_size, encoded_size, encoded_size, feature_dim)
            captions: Tensor (batch_size, max_caption_length) - ground truth caption indices
            lengths: List panjang caption untuk setiap sample
        
        Returns:
            predictions: Tensor (batch_size, max_caption_length, vocab_size)
        """
        batch_size = encoder_out.size(0)
        
        # Init hidden state dari CNN features
        h, c = self.init_hidden_state(encoder_out)
        
        # Embedding captions
        # Input: captions[:, :-1] karena kita predict word NEXT
        # Misal caption: [<start>, pria, duduk, <end>]
        # Input ke LSTM: [<start>, pria, duduk]
        # Target: [pria, duduk, <end>]
        embeddings = self.embedding(captions[:, :-1])  # (batch, seq_len-1, embed_dim)
        embeddings = self.dropout(embeddings)
        
        # Pack padded sequence (untuk efisiensi dengan variable-length sequences)
        # lengths - 1 karena kita buang token terakhir
        lengths_input = [l - 1 for l in lengths]
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths_input, batch_first=True, enforce_sorted=True
        )
        
        # LSTM forward
        hiddens, _ = self.lstm(packed, (h, c))
        
        # Unpack
        hiddens, _ = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)
        
        # Predictions
        predictions = self.fc(hiddens)  # (batch, seq_len-1, vocab_size)
        
        return predictions
    
    def sample(self, encoder_out, start_token, end_token, max_length=50):
        """
        🔮 INFERENCE MODE: Generate caption (greedy decoding)
        
        Args:
            encoder_out: Tensor (batch_size, encoded_size, encoded_size, feature_dim)
            start_token: Index token <start>
            end_token: Index token <end>
            max_length: Panjang maksimal caption
        
        Returns:
            sampled_ids: List of word indices untuk setiap sample di batch
        """
        batch_size = encoder_out.size(0)
        
        # Init hidden state
        h, c = self.init_hidden_state(encoder_out)
        
        # Start dengan <start> token
        inputs = torch.full((batch_size, 1), start_token, dtype=torch.long).to(encoder_out.device)
        
        sampled_ids = []
        
        for _ in range(max_length):
            # Embed input
            embeddings = self.embedding(inputs)  # (batch, 1, embed_dim)
            
            # LSTM step
            hiddens, (h, c) = self.lstm(embeddings, (h, c))  # (batch, 1, hidden_size)
            
            # Predict next word
            outputs = self.fc(hiddens.squeeze(1))  # (batch, vocab_size)
            
            # Greedy: ambil word dengan prob tertinggi
            predicted = outputs.argmax(dim=1)  # (batch,)
            
            sampled_ids.append(predicted)
            
            # Next input adalah predicted word
            inputs = predicted.unsqueeze(1)  # (batch, 1)
            
            # Stop jika semua batch sudah predict <end>
            if (predicted == end_token).all():
                break
        
        # Stack: (max_length, batch) -> (batch, max_length)
        sampled_ids = torch.stack(sampled_ids, 1)  # (batch, length)
        
        return sampled_ids
    
    def beam_search(self, encoder_out, start_token, end_token, beam_size=3, max_length=50):
        """
        🔍 INFERENCE MODE: Generate caption dengan Beam Search
        
        Args:
            encoder_out: Tensor (1, encoded_size, encoded_size, feature_dim) - single image
            start_token: Index token <start>
            end_token: Index token <end>
            beam_size: Beam width
            max_length: Panjang maksimal caption
        
        Returns:
            best_sequence: List of word indices (best caption)
        """
        # Beam search lebih kompleks, untuk simplicity kita implementasi greedy dulu
        # Anda bisa upgrade ke beam search nanti
        return self.sample(encoder_out, start_token, end_token, max_length)


if __name__ == "__main__":
    print("Testing LSTM Decoder...")
    
    # Dummy embedding matrix
    vocab_size = 1000
    embedding_dim = 300
    dummy_embeddings = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    
    # Create decoder
    decoder = DecoderLSTM(
        embedding_matrix=dummy_embeddings,
        vocab_size=vocab_size,
        hidden_size=512,
        num_layers=1,
        dropout=0.5,
        fine_tune_embeddings=False
    )
    
    # Dummy encoder output (VGG16)
    encoder_out = torch.randn(4, 14, 14, 512)  # batch=4
    
    # Dummy captions (ground truth)
    captions = torch.randint(0, vocab_size, (4, 20))  # batch=4, max_len=20
    lengths = [15, 18, 12, 20]  # Variable lengths
    
    # Test training mode
    decoder.train()
    predictions = decoder(encoder_out, captions, lengths)
    
    print(f"\n✅ Decoder Training Test:")
    print(f"   Encoder out shape: {encoder_out.shape}")
    print(f"   Captions shape: {captions.shape}")
    print(f"   Predictions shape: {predictions.shape}")
    
    # Test inference mode
    decoder.eval()
    with torch.no_grad():
        sampled = decoder.sample(encoder_out[:1], start_token=1, end_token=2, max_length=20)
    
    print(f"\n✅ Decoder Inference Test:")
    print(f"   Sampled caption shape: {sampled.shape}")
    print(f"   Sampled IDs: {sampled[0].tolist()[:10]}...")
    
    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Embedding frozen: {not decoder.embedding.weight.requires_grad}")
