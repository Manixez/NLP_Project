#!/bin/bash
# ==============================================================================
# 🚀 QUICK REFERENCE - Image Captioning Commands
# ==============================================================================

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════════╗
║                    IMAGE CAPTIONING - QUICK COMMANDS                     ║
╚══════════════════════════════════════════════════════════════════════════╝

📦 SETUP (Run Once)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Activate virtual environment:
   $ source NLP/bin/activate

2. Install dependencies:
   $ pip install -r requirements.txt

3. Download NLTK data:
   $ python -c "import nltk; nltk.download('punkt')"

4. Download FastText model (6.8 GB):
   $ mkdir -p fasttext && cd fasttext
   $ wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.bin.gz
   $ gunzip cc.id.300.bin.gz
   $ cd ..

5. OR run automated setup:
   $ ./setup.sh


📚 PREPROCESSING (FASE 0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Build vocabulary & embedding matrix (run ONCE before training):
$ python prepare_data.py

Output:
  ✓ output/vocab/vocab.pkl
  ✓ output/vocab/embedding_matrix.npy


🏋️ TRAINING (FASE 1-4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Start fresh training:
$ python train.py

Resume from checkpoint:
$ python train.py output/saved_models/checkpoint_epoch_10.pth

Monitor training:
  - Check terminal output (loss, BLEU score)
  - Checkpoints saved to: output/saved_models/
  - Best model: output/saved_models/best_model.pth


🔮 INFERENCE (Generate Captions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate caption for one image:
$ python inference.py --image path/to/image.jpg

With visualization:
$ python inference.py --image path/to/image.jpg --show

Use specific checkpoint:
$ python inference.py \
    --image path/to/image.jpg \
    --checkpoint output/saved_models/checkpoint_epoch_20.pth


🧪 TESTING & DEBUGGING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test encoder:
$ python models/encoder.py

Test decoder:
$ python models/decoder.py

Test vocabulary:
$ python utils/vocabulary.py

Test dataset loader:
$ python utils/dataset.py

View pipeline diagram:
$ python PIPELINE_DIAGRAM.py

Check GPU availability:
$ python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"


⚙️  CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Edit config.py to change:
  - CNN model: VGG16 / ResNet50 / ResNet101
  - Batch size, learning rate, epochs
  - LSTM hidden size, dropout
  - Data split, caption length

Common adjustments:
  CNN_MODEL = 'resnet50'      # Change encoder
  BATCH_SIZE = 16             # Reduce if GPU memory error
  NUM_EPOCHS = 30             # More epochs for better results
  LEARNING_RATE = 5e-5        # Lower LR for fine-tuning


📊 MONITORING & EVALUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

During training, monitor:
  - Training loss (should decrease)
  - Validation loss (should decrease)
  - BLEU-4 score (should increase)
  - Sample generated captions

Target BLEU scores:
  BLEU < 0.20  → Poor (needs more training)
  BLEU 0.20-0.35 → Fair (improving)
  BLEU 0.35-0.45 → Good (usable)
  BLEU > 0.45    → Excellent


🔧 COMMON ISSUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Issue: "FastText model not found"
Fix: Download cc.id.300.bin to fasttext/ folder

Issue: "CUDA out of memory"
Fix: Reduce BATCH_SIZE in config.py (try 16 or 8)

Issue: "Vocabulary not found"
Fix: Run prepare_data.py first

Issue: Poor caption quality
Fix: Train longer (30-50 epochs), check BLEU > 0.30

Issue: Training too slow
Fix: Use GPU, reduce batch size, or use smaller CNN (VGG16)


📁 PROJECT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Kode/
├── config.py              # Hyperparameters ⚙️
├── prepare_data.py        # Build vocab (FASE 0) 📚
├── train.py              # Training (FASE 1-4) 🏋️
├── inference.py          # Generate captions 🔮
├── models/
│   ├── encoder.py        # CNN (VGG/ResNet)
│   └── decoder.py        # LSTM
├── utils/
│   ├── vocabulary.py     # Vocab + embeddings
│   └── dataset.py        # DataLoader
├── Dataset/              # Dataset folder 📂
├── fasttext/             # FastText model 🌉
└── output/               # Results 💾
    ├── saved_models/     # Checkpoints
    ├── vocab/           # Vocab files
    └── logs/            # Logs


📚 DOCUMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Full guide: README.md
Pipeline diagram: PIPELINE_DIAGRAM.py
Config reference: config.py
Quick start: ./setup.sh


════════════════════════════════════════════════════════════════════════════

✨ TYPICAL WORKFLOW:

1. ./setup.sh                      # One-time setup
2. python prepare_data.py          # Build vocab (once)
3. python train.py                 # Train model (long)
4. python inference.py --image ... # Test on images

════════════════════════════════════════════════════════════════════════════

EOF
