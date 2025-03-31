import sys
import coremltools as ct
import tensorflow as tf

def convert_to_mlmodel(model_path):
    # Load the TensorFlow 1 model (assuming it's a frozen .pb file)
    with tf.io.gfile.GFile(model_path + "/saved_model.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Import the graph definition into a new TensorFlow graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    # Define input shape (MobileBERT typically expects [batch, seq_length, hidden_size])
    # Adjust this based on your needs or model documentation
    input_shape = [1, 384, 128]  # Example: batch=1, seq_length=384, hidden_size=128

    # Convert to Core ML
    mlmodel = ct.convert(
        graph,
        inputs=[ct.TensorType(name="input_ids", shape=input_shape)],
        outputs=["output"],  # Adjust output name based on model graph inspection
    )

    # Save the model
    mlmodel.save("mobilebert.mlmodel")
    print("Model converted and saved as mobilebert.mlmodel")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_to_coreml.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    convert_to_mlmodel(model_path)
