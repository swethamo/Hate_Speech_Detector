# This python file is used to run the ONNX model on a Command Line Interface
import onnxruntime as ort
import numpy as np
import argparse
from pathlib import Path
import json
from pprint import pprint
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

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
    max_len = 100  # Ensure this matches the model's input length
    test_padded = pad_sequences(
        test_seq, maxlen=max_len, padding="post", truncating="post"
    )
    test_padded = test_padded.astype(np.float32)

    # Run inference
    outputs = session.run([output_name], {input_name: test_padded})

    # Get the predicted class
    predicted_class = np.argmax(outputs[0], axis=1)
    return predicted_class


# Arguments
parser = argparse.ArgumentParser(
    description="Run Hate Speech Detection on input directory or file."
)
parser.add_argument(
    "--input_dir",
    type=str,
    required=True,
    help="Directory or file containing input text.",
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Directory to save output results."
)
args = parser.parse_args()

# Convert input paths to Path objects
input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(
    parents=True, exist_ok=True
)  # Create output directory if it doesn't exist

# Class labels
class_labels = ["Hate Speech", "Offensive", "Neither"]

# Process input directory or file
results = {}
if input_dir.is_file():
    # Process a single file
    with open(input_dir, "r") as f:
        sentence = f.read().strip()
    preprocessed_sentence = preprocess_sentence(sentence)
    predicted_class = predict(preprocessed_sentence)
    results[input_dir.name] = {
        "sentence": sentence,
        "prediction": class_labels[predicted_class[0]],
    }
else:
    # Process all files in the input directory
    for input_file in input_dir.glob("*"):
        if input_file.is_file():
            with open(input_file, "r") as f:
                sentence = f.read().strip()
            preprocessed_sentence = preprocess_sentence(sentence)
            predicted_class = predict(preprocessed_sentence)
            results[input_file.name] = {
                "sentence": sentence,
                "prediction": class_labels[predicted_class[0]],
            }

# Print results to console
pprint(results)

# Save results to output directory
with open(output_dir / "output.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_dir / 'output.json'}")
