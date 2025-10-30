import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify
from pathlib import Path
import os
import io

# Import our custom library files
from chessboard_snipper import process_image
from flip_board_to_black_pov import assemble_fen_from_predictions, black_perspective_fen, FEN_MAP

# --- App Initialization ---
app = Flask(__name__)

# --- Global Variables (Load Model Once) ---
MODEL = None
CLASS_NAMES = None
MODEL_PATH = Path("model/chess_piece_model.keras")
LABELS_PATH = Path("labels/class_names.txt")

def load_model():
    """Load the Keras model and class names into memory."""
    global MODEL, CLASS_NAMES
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        CLASS_NAMES = LABELS_PATH.read_text().splitlines()
        print("--- Model and class names loaded successfully. Ready to serve. ---")
    else:
        print("--- WARNING: Model or labels file not found. ---")
        print(f"Looked for: {MODEL_PATH} and {LABELS_PATH}")
        print("Please run 'python train_model.py' first.")

@app.route('/')
def index():
    """Render the main HTML page."""
    # Renders the 'index.html' file from the 'templates' folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the image upload and return FEN."""
    if MODEL is None or CLASS_NAMES is None:
        return jsonify({'error': 'Model not loaded on server.'}), 500

    # 1. Get data from the form
    if 'file' not in request.files:
        return jsonify({'error': 'No file part.'}), 400
    
    file = request.files['file']
    pov = request.form.get('pov', 'w') # Default to white 'w'
    
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    try:
        # 2. Read image bytes and process
        image_bytes = file.read()
        processed_data = process_image(image_bytes) # Our library function handles bytes
        
        if processed_data is None:
            return jsonify({'error': 'Could not process image. Board not found.'}), 400
            
        model_inputs, _, _ = processed_data

        # 3. Run Prediction
        batch_to_predict = np.array(model_inputs)
        raw_predictions = MODEL.predict(batch_to_predict)
        predicted_indices = np.argmax(raw_predictions, axis=1)
        predicted_class_names = [CLASS_NAMES[i] for i in predicted_indices]

        # 4. Assemble FEN
        position_string = assemble_fen_from_predictions(predicted_class_names)
        
        if pov == 'b':
            print("Flipping FEN for Black's perspective...")
            position_string = black_perspective_fen(position_string)
            
        final_fen = position_string + " w KQkq - 0 1"
        
        # 5. Return success
        return jsonify({'fen': final_fen})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create required folders if they don't exist
    Path("templates").mkdir(exist_ok=True)
    
    # Load the model *before* starting the server
    load_model()
    
    # Run the Flask app
    # To access: http://127.0.0.1:5000
    print("Starting Flask server... Access at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
