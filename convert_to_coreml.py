import sys
import coremltools as ct
import tensorflow as tf

def convert_to_mlmodel(model_path):
    try:
        # Load the TensorFlow SavedModel with the appropriate tags
        print(f"Loading SavedModel from: {model_path}")
        loaded_model = tf.saved_model.load(model_path, tags=[])

        # Get the available signatures
        print("Available signatures:", list(loaded_model.signatures.keys()))

        # Inspect each signature
        for signature_name in loaded_model.signatures:
            print(f"\nInspecting signature: {signature_name}")
            signature = loaded_model.signatures[signature_name]
            print("Inputs:", signature.inputs)
            print("Outputs:", signature.outputs)

        # If there's no serving_default signature, we can't proceed with conversion
        if "serving_default" not in loaded_model.signatures:
            raise ValueError("This SavedModel does not contain a 'serving_default' signature for inference. It might be a tokenizer or preprocessing model.")

        # If we had a serving_default signature, we would proceed with conversion
        signature = loaded_model.signatures["serving_default"]
        print("Signature inputs:", signature.inputs)
        print("Signature outputs:", signature.outputs)

        # Define input shape (MobileBERT typically expects [batch, seq_length])
        input_shape = [1, 384]  # Batch size 1, sequence length 384 (adjust as needed)

        # MobileBERT typically has inputs like "input_ids", "attention_mask", "token_type_ids"
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