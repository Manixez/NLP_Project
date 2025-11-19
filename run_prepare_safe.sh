#!/bin/bash
# Safe script untuk build vocabulary tanpa crash VSCode

echo "=============================================="
echo "🔧 Building Vocabulary (Safe Mode)"
echo "=============================================="

# Set memory limits untuk prevent crash
export PYTHONHASHSEED=0

# Close unnecessary processes
echo "💡 TIP: Tutup VSCode dan aplikasi lain dulu untuk free up RAM"
echo ""
echo "Press ENTER to continue or Ctrl+C to cancel..."
read

# Activate venv
source NLP/bin/activate

# Run dengan monitoring
echo "🚀 Starting vocabulary build..."
echo "⏱️  Estimated time: 2-5 minutes"
echo ""

# Run dengan nice priority (lower CPU usage, prevent system freeze)
nice -n 10 python prepare_data.py

echo ""
echo "=============================================="
echo "✅ Done! Check output/vocab/ for results"
echo "=============================================="
