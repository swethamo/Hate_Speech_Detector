# Hate Speech Detection Model with ONNX and Flask-ML

This repository demonstrates how to create, convert, and deploy a Hate Speech Detection model using ONNX and Flask-ML. This model is designed to detect and classify text into categories ( Hate Speech, Offensive Language, Neither ). 

- Input: Raw text (e.g., sentences, paragraphs).
- Output: A prediction indicating whether the text is Hate Speech, Offensive, or Neither.
- Use Case: This model can be integrated into platforms to automatically flag or filter harmful content, ensuring safer online environments.

**How It Works**
- Text Preprocessing: The input text is cleaned by removing punctuation, stopwords, and lemmatizing words.
- Tokenization: The text is converted into sequences of integers using a pre-trained tokenizer.
- Prediction: The preprocessed text is fed into the ONNX model, which outputs the predicted class.

This model is lightweight, efficient, and ready for deployment in production environments using frameworks like Flask or ONNX RuntimeThe model is designed to be compatible with the RescueBox Desktop application.

## Overview

1. **Steps to run the project**
   1. Installations
   2. Running the project on RescueBox
   3. Running the project through the CLI
      
2. **ONNX Conversion**
   1. How to convert a project to ONNX
   2. Steps to convert this model to ONNX
      
## Getting Started

### Prerequisites

- Python 3.12.4
- Pip (Python package manager)

## Installation

1. **Clone the Repository**:

   ```bash
   git clone [https://github.com/your-username/hate-speech-detection.git](https://github.com/swethamo/Hate_Speech_Detector.git)
   cd Hate_Speech_Detector
   ```

2. **Install Dependencies:**
   Install the required Python packages using the following command:

   ```bash
   pip install -r requirements.txt
   ```

3. **Running the Flask-ML Server**
   Start the Flask-ML server to work with RescueBox for predictions:

   ```bash
   python server.py
   ```

   The server will start running on 127.0.0.1 5000

   **Download and run RescueBox Desktop from the following link: [Rescue Box Desktop](https://github.com/UMass-Rescue/RescueBox-Desktop/releases)**

   Open the RescueBox Desktop application and register the model
   
   ![RescueBox Desktop](images/Register_model.png)

   Run the model

   ![RescueBox Desktop](images/Run_the_model.png)

   Set the Input and Output directory.

   Input directory should have a txt(s) with a sentence for prediction and an output directory where the json file with the predictions will be outputted.
   
   ![RescueBox Desktop](images/Input_output_dir.png)

   View the results

   ![RescueBox Desktop](images/output.png)

5. **Using the Command Line Interface (CLI)**
   The CLI allows you to test the model on text files and save the predictions in JSON format.

   Prepare Input Files:
   Place the txt files containing sentences for prediction in an inputs folder

   Run the CLI:
   Use the following command to generate predictions:

   ```bash
   python run_cli.py --input_dir inputs --output_dir outputs
   ```

   --input_dir: Directory containing the input text files.
   --output_dir: Directory where the JSON file with predictions will be saved.

## ONNX Conversion
   
   Dataset used:   [Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)

   Run the `HateSpeech.ipynb` Jupyter notebook to train, save, convert the model to onnx format and make predictions through it. [Reference](https://www.geeksforgeeks.org/hate-speech-detection-using-deep-learning/)

**ONNX Conversion and Prediction Process**
1. ONNX Conversion
ONNX (Open Neural Network Exchange) is a format for representing machine learning models. The conversion process involves transforming a model from one framework (e.g., TensorFlow/Keras) into the ONNX format. Here's how it works in your code:

- Load the Keras Model: The TensorFlow/Keras model (hate_speech_model.keras) is loaded using tf.keras.models.load_model.
- Define Input Signature: The input signature is explicitly defined using tf.TensorSpec to specify the expected input shape and data type.
- Convert to ONNX: The tf2onnx.convert.from_keras function converts the Keras model to ONNX format. The input_signature ensures the ONNX model knows the expected input format.
- Save the ONNX Model: The converted ONNX model is serialized and saved to a file (hate_speech_model.onnx).

2. Loading and Predicting with ONNX Runtime

Once the model is in ONNX format, you can use the ONNX Runtime to make predictions.
   
- Preprocess Input: The input sentence is preprocessed (lowercased, punctuation removed, stopwords removed, tokenized, and padded) to match the input format expected by the model.
- Load ONNX Model: The ONNX model is loaded using onnxruntime.InferenceSession.
- Get Input/Output Names: The input and output names of the model are retrieved using session.get_inputs() and session.get_outputs().
- Run Inference: The preprocessed input is passed to the ONNX model using session.run. The model returns the output (e.g., logits or probabilities).
- Postprocess Output: The output is postprocessed (e.g., using np.argmax) to get the predicted class, which is then mapped to a human-readable label.

  Run the onnxconversion.py script to convert the trained model to ONNX format or the last 2 codeblocks of the `HateSpeech.ipynb`

   ```bash
   python onnxconversion.py
   ```
   This will generate the `hate_speech_model.onnx` file.


