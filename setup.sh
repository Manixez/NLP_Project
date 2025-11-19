#!/bin/bash
# Quick Start Script - Image Captioning Pipeline
# ===============================================

echo "🚀 Image Captioning Pipeline - Quick Start"
echo "=========================================="
echo ""

# 1. Activate virtual environment
echo "1️⃣  Activating virtual environment..."
source NLP/bin/activate

# 2. Install dependencies
echo ""
echo "2️⃣  Installing dependencies..."
pip install -q -r requirements.txt

# 3. Download NLTK data
echo ""
echo "3️⃣  Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True)"

# 4. Create output directories
echo ""
echo "4️⃣  Creating output directories..."
python -c "import config; config.create_output_dirs()"

# 5. Check FastText model
echo ""
echo "5️⃣  Checking FastText model..."
if [ -f "fasttext/cc.id.300.bin" ]; then
    echo "   ✅ FastText model found!"
else
    echo "   ⚠️  FastText model NOT found!"
    echo "   📥 Download from: https://fasttext.cc/docs/en/crawl-vectors.html"
    echo "   💾 Save to: fasttext/cc.id.300.bin"
    echo ""
    echo "   Quick download:"
    echo "   mkdir -p fasttext"
    echo "   cd fasttext"
    echo "   wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.bin.gz"
    echo "   gunzip cc.id.300.bin.gz"
    echo "   cd .."
fi

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Download FastText model (jika belum)"
echo "   2. Run: python prepare_data.py   (build vocabulary)"
echo "   3. Run: python train.py          (start training)"
echo "   4. Run: python inference.py --image <path> (generate caption)"
echo ""
