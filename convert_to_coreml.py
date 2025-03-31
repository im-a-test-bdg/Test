import sys
import coremltools as ct
import tensorflow as tf

def convert_to_mlmodel(model_path):
    try:
        # Load the TensorFlow SavedModel
        print(f"Loading SavedModel from: {model_path}")
        # Use tf.saved_model.load to load the entire SavedModel directory
        loaded_model = tf.saved_model.load(model_path)

        # Get the default serving signature
        print("Available signatures:", list(loaded_model.signatures.keys()))
        signature = loaded_model.signatures["serving_default"]
        print("Signature inputs:", signature.inputs)
        print("Signature outputs:", signature.outputs)

        # Define input shape (MobileBERT typically expects [batch, seq_length])
        input_shape = [1, 384]  # Batch size 1, sequence length 384 (adjust as needed)

        # MobileBERT typically has inputs like "input_ids", "attention_mask", "token_type_ids"
        # Outputs are often "logits" or "pooled_output" for classification tasks
        # Adjust these based on the output of saved_model_cli or the signature above
        inputs = [
            ct.TensorType(name="input_ids", shape=input_shape, dtype=int),
            ct.TensorType(name="attention_mask", shape=input_shape, dtype=int),
            ct.TensorType(name="token_type_ids", shape=input_shape, dtype=int),
        ]
        outputs = ["pooled_output"]  # Adjust based on the signature outputs

        # Convert to Core ML
        print("Converting to Core ML...")
        mlmodel = ct.convert(
            loaded_model,
            inputs=inputs,
            outputs=outputs,
        )

        # Save the model
        mlmodel.save("mobilebert.mlmodel")
        print("Model converted and saved as mobilebert.mlmodel")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_to_coreml.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    convert_to_mlmodel(model_path)