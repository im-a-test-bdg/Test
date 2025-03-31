#!/bin/bash

# Hardcoded Kaggle credentials
KAGGLE_USERNAME="josephcristini"
KAGGLE_KEY="ae9de1e12100d4a02f851f52e071f6ea"

# Set variables
DOWNLOAD_DIR=~/Downloads
MODEL_TAR="$DOWNLOAD_DIR/model.tar.gz"
EXTRACT_DIR="$DOWNLOAD_DIR/mobilebert_extracted"
OUTPUT_MODEL="$DOWNLOAD_DIR/MobileBERT.mlmodel"

# Create directories
mkdir -p "$EXTRACT_DIR"

# Download MobileBERT from Kaggle with hardcoded credentials
echo "Downloading MobileBERT from Kaggle..."
curl -L -o "$MODEL_TAR" \
  https://www.kaggle.com/api/v1/models/google/mobilebert/tensorFlow1/uncased-l-24-h-128-b-512-a-4-f-4-opt/1/download \
  -H "Authorization: Basic $(echo -n "$KAGGLE_USERNAME:$KAGGLE_KEY" | base64)"

# Check download
if [ $? -ne 0 ]; then
  echo "Error: Failed to download model. Check credentials or network."
  exit 1
fi

# Extract the tar.gz
echo "Extracting model..."
tar -xzf "$MODEL_TAR" -C "$EXTRACT_DIR"

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
