import sys
import coremltools as ct
import tensorflow as tf

def convert_to_mlmodel(model_path):
    try:
        # Load the TensorFlow 1 SavedModel
        print(f"Loading SavedModel from: {model_path}")
        with tf.io.gfile.GFile(f"{model_path}/saved_model.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        # Import the graph definition into a new TensorFlow graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        # Inspect the graph to find input and output tensors
        print("Graph operations:")
        for op in graph.get_operations():
            print(op.name)

        # Define input shape (MobileBERT typically expects [batch, seq_length])
        # MobileBERT often expects multiple inputs: input_ids, attention_mask, token_type_ids
        input_shape = [1, 384]  # Batch size 1, sequence length 384 (adjust as needed)

        # MobileBERT typically has inputs like "input_ids", "attention_mask", "token_type_ids"
        # Outputs are often "logits" or "pooled_output" for classification tasks
        # Adjust these based on the graph inspection above
        inputs = [
            ct.TensorType(name="input_ids", shape=input_shape, dtype=int),
            ct.TensorType(name="attention_mask", shape=input_shape, dtype=int),
            ct.TensorType(name="token_type_ids", shape=input_shape, dtype=int),
        ]
        outputs = ["pooled_output"]  # Adjust based on your task (e.g., "logits" for classification)

        # Convert to Core ML
        print("Converting to Core ML...")
        mlmodel = ct.convert(
            graph,
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