# This script replaces 'test_model.py'
# It runs the full pipeline locally in your terminal.

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import os

# Import our custom library files
from chessboard_snipper import process_image
from flip_board_to_black_pov import assemble_fen_from_predictions, black_perspective_fen, FEN_MAP

# --- Configuration ---
MODEL_PATH = Path("model/chess_piece_model.keras")
LABELS_PATH = Path("labels/class_names.txt")

def load_model_and_classes_local():
    """Loads model and class names from local paths."""
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run 'python train_model.py' first.")
        return None, None
    if not LABELS_PATH.exists():
        print(f"Error: Labels file not found at {LABELS_PATH}")
        print("Please run 'python train_model.py' first.")
        return None, None
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = LABELS_PATH.read_text().splitlines()
        print("Model and class names loaded successfully.")
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def main():
    """
    Main function to run the full pipeline from the terminal.
    """
    # 1. Load Model
    model, class_names = load_model_and_classes_local()
    if model is None:
        return

    # 2. Get Image Path
    image_path = input("Enter the path to your chessboard image: ").strip('\'"')
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Processing image: {image_path}...")
    
    # 3. Process Image (using our library function)
    # process_image takes a path string directly
    processed_data = process_image(image_path)
    
    if processed_data is None:
        print("Image processing failed. Exiting.")
        return
        
    model_inputs, board_image_viz, bbox = processed_data
    print("Image processed successfully.")

    # 4. Run Prediction
    print("\nRunning model predictions...")
    batch_to_predict = np.array(model_inputs)
    raw_predictions = model.predict(batch_to_predict)
    predicted_indices = np.argmax(raw_predictions, axis=1)
    predicted_confidences = np.max(raw_predictions, axis=1)
    predicted_class_names = [class_names[i] for i in predicted_indices]

    # --- Debug Prints ---
    print("\n--- Detailed Prediction Output (Square Index: Class Name (Confidence %)) ---")
    square_names = [f"{chr(ord('a')+c)}{8-r}" for r in range(8) for c in range(8)]
    for i in range(64):
        print(f"{square_names[i]}: {predicted_class_names[i]} ({predicted_confidences[i]*100:.1f}%)", end="  ")
        if (i + 1) % 8 == 0: print() # Newline
    
    # 5. Get POV
    pov = input("\nIs the image from Black's perspective (Rank 8 at bottom)? [y/n]: ").lower().strip()
    
    # 6. Assemble FEN
    # We always assemble assuming White's POV first
    position_string = assemble_fen_from_predictions(predicted_class_names)
    
    if pov == 'y':
        print("Flipping FEN for Black's perspective...")
        position_string = black_perspective_fen(position_string)
        
    final_fen = position_string + " w KQkq - 0 1"

    print(f"\n--- FINAL GENERATED FEN ---")
    print(final_fen)

    # --- Visualization ---
    print("\nDisplaying visualizations... Press any key to close.")
    
    # Show original image with bounding box
    raw_image = cv2.imread(image_path)
    x, y, w, h = bbox
    cv2.rectangle(raw_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("1. Original Image with Detected Board", raw_image)

    # Show the cropped/resized board
    cv2.imshow("2. Sliced Board (for processing)", board_image_viz)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # To run:
    # 1. Activate venv: .\venv\Scripts\activate
    # 2. Run: python run_local.py
    main()
