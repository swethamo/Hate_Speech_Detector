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
   cd Hate_Speech_detector
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

   The server will start running.

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

   Run the `HateSpeech.ipynb` Jupyter notebook to train, save and convert it to onnx format. [Reference](https://www.geeksforgeeks.org/hate-speech-detection-using-deep-learning/)

   **Steps used to convert to ONNX:**   
   - Install the required dependencies - tensorflow, tf2onnx, and onnxruntime using pip.
   - Load the trained Keras model (hate_speech_model.keras) using TensorFlow.
   - Specify the input shape and data type for the ONNX model.
   - Use **tf2onnx** to convert the Keras model to ONNX format.
   - Save the converted model as hate_speech_model.onnx.
   
   Export the ONNX Model (Optional)
   If you need to debug or verify the ONNX model:
   - Set a breakpoint in the prediction function to inspect the model's input and output.
   - Use the onnxruntime API to test the model with sample inputs and verify the outputs.
   
   Run the onnxconversion.py script to convert the trained model to ONNX format or the last 2 codeblocks of the `HateSpeech.ipynb`

   ```bash
   python onnxconversion.py
   ```
   This will generate the `hate_speech_model.onnx` file.


