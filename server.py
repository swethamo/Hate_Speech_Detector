from typing import TypedDict
from pathlib import Path
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    DirectoryInput,
    FileResponse,
    InputSchema,
    InputType,
    ResponseBody,
    TaskSchema,
)
import onnxruntime as ort
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK data (if not already downloaded)
nltk.download("stopwords")
nltk.download("omw-1.4")
nltk.download("wordnet")


# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    token = pickle.load(f)

# Load the ONNX model
session = ort.InferenceSession("hate_speech_model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

max_len = 100  # Ensure this matches your model's input length


class HateSpeechInputs(TypedDict):
    input_dir: DirectoryInput
    output_dir: DirectoryInput


class HateSpeechParameters(TypedDict):
    pass  # No parameters needed for this task


def create_hate_speech_task_schema() -> TaskSchema:
    input_schema = InputSchema(
        key="input_dir",
        label="Path to the directory containing text files",
        input_type=InputType.DIRECTORY,
    )
    output_schema = InputSchema(
        key="output_dir",
        label="Path to the output directory",
        input_type=InputType.DIRECTORY,
    )
    return TaskSchema(inputs=[input_schema, output_schema], parameters=[])


# Create a server instance
server = MLServer(__name__)

server.add_app_metadata(
    name="Hate Speech Detector",
    author="Your Name",
    version="1.0",
    info=load_file_as_string("app-info.md"),
)

# Preprocessing functions
punctuations_list = string.punctuation
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def remove_punctuations(text):
    temp = str.maketrans("", "", punctuations_list)
    return text.translate(temp)


def remove_stopwords(text):
    imp_words = []
    for word in str(text).split():
        if word not in stop_words:
            imp_words.append(lemmatizer.lemmatize(word))
    return " ".join(imp_words)


def preprocess_sentence(sentence):
    # Preprocess the sentence
    sentence = sentence.lower()  # Convert to lowercase
    sentence = remove_punctuations(sentence)  # Remove punctuation
    sentence = remove_stopwords(sentence)  # Remove stopwords
    return sentence


def predict(sentence):
    # Tokenize the sentence
    test_seq = token.texts_to_sequences([sentence])

    # Pad the sequence
    test_padded = pad_sequences(
        test_seq, maxlen=max_len, padding="post", truncating="post"
    )
    test_padded = test_padded.astype(np.float32)

    # Run inference
    outputs = session.run([output_name], {input_name: test_padded})

    # Get the predicted class
    predicted_class = np.argmax(outputs[0], axis=1)
    return predicted_class


@server.route("/detect_hate_speech", task_schema_func=create_hate_speech_task_schema)
def detect_hate_speech(
    inputs: HateSpeechInputs, parameters: HateSpeechParameters
) -> ResponseBody:
    input_path = Path(inputs["input_dir"].path)
    output_path = Path(inputs["output_dir"].path)
    output_path.mkdir(
        parents=True, exist_ok=True
    )  # Create output directory if it doesn't exist

    results = {}
    class_labels = [
        "Hate Speech",
        "Offensive",
        "Neither",
    ]  # Ensure this matches run_cli.py

    if input_path.is_file():
        # Process a single file
        with open(input_path, "r") as f:
            sentence = f.read().strip()
        preprocessed_sentence = preprocess_sentence(sentence)
        predicted_class = predict(preprocessed_sentence)
        results[input_path.name] = {
            "sentence": sentence,
            "prediction": class_labels[
                predicted_class[0]
            ],  # Use the correct class label
        }
    else:
        # Process all files in the input directory
        for input_file in input_path.glob("*"):
            if input_file.is_file():
                with open(input_file, "r") as f:
                    sentence = f.read().strip()
                preprocessed_sentence = preprocess_sentence(sentence)
                predicted_class = predict(preprocessed_sentence)
                results[input_file.name] = {
                    "sentence": sentence,
                    "prediction": class_labels[
                        predicted_class[0]
                    ],  # Use the correct class label
                }

    # Save results to a JSON file in the specified output directory
    output_file = output_path / "output.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    return ResponseBody(FileResponse(path=str(output_file), file_type="json"))


if __name__ == "__main__":
    server.run()
