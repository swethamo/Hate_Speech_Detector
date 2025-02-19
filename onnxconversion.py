import tensorflow as tf
import tf2onnx

max_len = 100
# Load the Keras model
model = tf.keras.models.load_model("hate_speech_model.keras")
model.output_names = ["output"]

spec = (
    tf.TensorSpec(shape=[None, max_len], dtype=tf.float32),
)  

# Convert the model to ONNX format
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

# Save the ONNX model
with open("hate_speech_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model successfully converted to ONNX format!")
