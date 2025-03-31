#!/bin/bash

# convert_model.sh
# This script automates the conversion of a TensorFlow SavedModel to a Core ML model in a GitHub Codespace.

# Exit on any error
set -e

# Define variables
MODEL_TAR="model.tar.gz"
EXTRACT_DIR="extracted_model"
OUTPUT_MODEL="converted_model.mlmodel"
PYTHON_VERSION="3.10"
TENSORFLOW_VERSION="2.14.0"
COREMLTOOLS_VERSION="7.2"

# Step 1: Verify the presence of model.tar.gz
echo "Checking for $MODEL_TAR..."
if [ ! -f "$MODEL_TAR" ]; then
    echo "Error: $MODEL_TAR not found in the current directory."
    echo "Please ensure $MODEL_TAR is present in the repository root."
    exit 1
fi

# Step 2: Extract the model
echo "Extracting $MODEL_TAR..."
mkdir -p "$EXTRACT_DIR"
tar -xzf "$MODEL_TAR" -C "$EXTRACT_DIR"

# Verify extraction
echo "Listing contents of $EXTRACT_DIR:"
ls -la "$EXTRACT_DIR"
MODEL_SUBDIR=$(find "$EXTRACT_DIR" -maxdepth 1 -type d | grep -v "^$EXTRACT_DIR$" | head -n 1)
if [ -z "$MODEL_SUBDIR" ]; then
    echo "Error: No model directory found in $EXTRACT_DIR after extraction."
    exit 1
fi
echo "Found model directory: $MODEL_SUBDIR"

# Verify the presence of saved_model.pb
if [ ! -f "$MODEL_SUBDIR/saved_model.pb" ]; then
    echo "Error: saved_model.pb not found in $MODEL_SUBDIR."
    exit 1
fi

# Step 3: Set up the Python environment
echo "Setting up Python environment..."

# Check if Python 3.10 is available, install if necessary
if ! command -v "python$PYTHON_VERSION" &> /dev/null; then
    echo "Python $PYTHON_VERSION not found. Attempting to install..."
    sudo apt update
    sudo apt install -y "python$PYTHON_VERSION" "python$PYTHON_VERSION-venv" "python$PYTHON_VERSION-dev"
fi

# Create and activate a virtual environment
echo "Creating virtual environment..."
rm -rf venv
python$PYTHON_VERSION -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install tensorflow==$TENSORFLOW_VERSION coremltools==$COREMLTOOLS_VERSION numpy

# Verify installation
echo "Verifying TensorFlow installation..."
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
echo "Verifying coremltools installation..."
python -c "import coremltools as ct; print('coremltools version:', ct.__version__)"

# Step 4: Convert the model
echo "Converting the model to Core ML..."

# Create a temporary Python script for conversion
cat << 'EOF' > convert_model.py
import tensorflow as tf
import coremltools as ct
import os
import sys
import numpy as np

# Find the model directory in extracted files
model_dir = './extracted_model/'
model_path = None
for item in os.listdir(model_dir):
    full_path = os.path.join(model_dir, item)
    if os.path.isdir(full_path):
        model_path = full_path
        break
if not model_path:
    print("Error: No model directory found in extracted files")
    sys.exit(1)

# The model directory should be 'backdoor-ai-tensorflow1-bdg-v1', but it's a MobileBERT model
print(f"Found model directory: {model_path}")

# Verify the presence of saved_model.pb
saved_model_pb = os.path.join(model_path, "saved_model.pb")
if not os.path.exists(saved_model_pb):
    print(f"Error: saved_model.pb not found in {model_path}")
    sys.exit(1)

# Load TensorFlow SavedModel
try:
    tf_model = tf.saved_model.load(model_path)
except Exception as e:
    print(f"Error loading TensorFlow SavedModel: {e}")
    sys.exit(1)

# Define a concrete function for conversion (Core ML requires a concrete function)
# MobileBERT expects input_ids, attention_mask, and token_type_ids
try:
    concrete_func = tf_model.signatures["serving_default"]
    if not concrete_func:
        print("Error: No serving_default signature found in SavedModel")
        sys.exit(1)
    print("Model signatures:", tf_model.signatures.keys())
    print("Input signature for serving_default:", concrete_func.structured_input_signature)
except Exception as e:
    print(f"Error retrieving concrete function: {e}")
    sys.exit(1)

# Define input shapes for MobileBERT (batch size 1, sequence length 128)
input_shape = (1, 128)  # Batch size 1, sequence length 128
input_ids = tf.TensorSpec(input_shape, tf.int32, name="input_ids")
attention_mask = tf.TensorSpec(input_shape, tf.int32, name="attention_mask")
token_type_ids = tf.TensorSpec(input_shape, tf.int32, name="token_type_ids")

# Convert to Core ML
try:
    mlmodel = ct.convert(
        concrete_func,
        inputs=[
            ct.TensorType(name="input_ids", shape=input_shape, dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=input_shape, dtype=np.int32),
            ct.TensorType(name="token_type_ids", shape=input_shape, dtype=np.int32),
        ],
        source='tensorflow',
        convert_to='mlmodel'
    )
except Exception as e:
    print(f"Error converting to Core ML: {e}")
    sys.exit(1)

# Save the converted model
output_path = './converted_model.mlmodel'
mlmodel.save(output_path)
print(f"Model successfully converted and saved to {output_path}")
EOF

# Run the conversion script
python convert_model.py

# Step 5: Verify the output
if [ -f "$OUTPUT_MODEL" ]; then
    echo "Conversion successful! $OUTPUT_MODEL has been created."
    ls -la "$OUTPUT_MODEL"
else
    echo "Error: Conversion failed. $OUTPUT_MODEL was not created."
    exit 1
fi

# Step 6: Clean up
echo "Cleaning up..."
rm convert_model.py

echo "Done! You can now download $OUTPUT_MODEL from the Codespace file explorer."