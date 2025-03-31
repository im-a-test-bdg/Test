#!/bin/bash

# Hardcoded Kaggle credentials
KAGGLE_USERNAME="josephcristini"
KAGGLE_KEY="ae9de1e12100d4a02f851f52e071f6ea"

# Set variables
DOWNLOAD_DIR=~/Downloads
MODEL_ZIP="$DOWNLOAD_DIR/mobilebert.zip"  # Kaggle CLI downloads as .zip
EXTRACT_DIR="$DOWNLOAD_DIR/mobilebert_extracted"
OUTPUT_MODEL="$DOWNLOAD_DIR/MobileBERT.mlmodel"

# Create directories
mkdir -p "$EXTRACT_DIR"

# Configure Kaggle CLI with hardcoded credentials
echo "Configuring Kaggle CLI..."
mkdir -p ~/.kaggle
echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Install Kaggle CLI if not present
if ! command -v kaggle &> /dev/null; then
    echo "Installing Kaggle CLI..."
    pip install kaggle
fi

# Download MobileBERT using the specified Kaggle URL
echo "Downloading MobileBERT from Kaggle..."
kaggle models instances versions download google/mobilebert/tensorFlow1/uncased-l-24-h-128-b-512-a-4-f-4-opt/1 -p "$DOWNLOAD_DIR"

# Check download
if [ $? -ne 0 ] || [ ! -f "$MODEL_ZIP" ]; then
  echo "Error: Failed to download model. Check credentials or network."
  exit 1
fi

# Extract the zip (Kaggle CLI downloads as .zip, not .tar.gz)
echo "Extracting model..."
unzip -o "$MODEL_ZIP" -d "$EXTRACT_DIR"

if [ $? -ne 0 ]; then
  echo "Error: Failed to extract model."
  exit 1
fi

# Python script for conversion
cat << EOF > convert_to_coreml.py
import coremltools as ct
import tensorflow as tf

# Load TensorFlow model
model_path = "$EXTRACT_DIR/saved_model"
model = tf.keras.models.load_model(model_path)

# Convert to CoreML
mlmodel = ct.convert(
    model,
    source="tensorflow",
    inputs=[ct.TensorType(name="input_ids", shape=(1, 128), dtype=tf.int32)],
    respect_trainable=True
)

# Optimize (8-bit quantization)
from coremltools.models import optimize
optimized_model = optimize.quantize_weights(mlmodel, quantization_mode="linear", nbits=8)

# Save
optimized_model.save("$OUTPUT_MODEL")
print("Model saved to $OUTPUT_MODEL")
EOF

# Install dependencies
echo "Installing dependencies..."
pip install coremltools tensorflow

# Convert
echo "Converting to CoreML..."
python3 convert_to_coreml.py

# Verify
if [ -f "$OUTPUT_MODEL" ]; then
  echo "Success! MobileBERT.mlmodel is at $OUTPUT_MODEL"
else
  echo "Error: Conversion failed."
  exit 1
fi