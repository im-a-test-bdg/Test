import sys
import coremltools as ct
import tensorflow as tf

def convert_to_mlmodel(model_path):
    # Load the TensorFlow 1 SavedModel
    try:
        # Assuming the path points to a directory with saved_model.pb
        with tf.io.gfile.GFile(f"{model_path}/saved_model.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        # Import the graph definition into a new TensorFlow graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        # Define input shape (MobileBERT typically expects [batch, seq_length, hidden_size])
        input_shape = [1, 384, 128]  # Adjust based on your model's requirements

        # Convert to Core ML
        mlmodel = ct.convert(
            graph,
            inputs=[ct.TensorType(name="input_ids", shape=input_shape)],
            outputs=["output"],  # Replace with actual output tensor name if known
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