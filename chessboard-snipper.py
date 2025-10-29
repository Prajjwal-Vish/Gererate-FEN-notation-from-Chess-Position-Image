import cv2
import numpy as np
import os
from typing import Optional, Tuple, List
from google.colab import files
from google.colab.patches import cv2_imshow

# --- Configuration ---
MODEL_SQUARE_SIZE = 64  # The input size your model was trained on (e.g., 64x64)
BOARD_SIZE_PX = MODEL_SQUARE_SIZE * 8  # 512 (e.g., 512x512)

def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    # Return 4 points in consistent order (tl, tr, br, bl)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _square_from_bbox(x:int, y:int, w:int, h:int, img_w:int, img_h:int, pad:int=0) -> Tuple[int,int,int,int]:

    side = max(w, h) + 2*pad
    cx = x + w // 2
    cy = y + h // 2
    half = side // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img_w, cx + half)
    y2 = min(img_h, cy + half)
    # recompute side after clipping so returned box is square (may be smaller near edges)
    side_w = x2 - x1
    side_h = y2 - y1
    side = min(side_w, side_h)
    # adjust if we clipped
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    return int(x1), int(y1), int(side), int(side)

def find_chessboard_contour(image: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    print("Finding chessboard contour (improved)...")
    h_img, w_img = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use bilateral filter to smooth while preserving edges (good for screenshots)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Try adaptive contrast to help faint edges
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Canny edges
    edges = cv2.Canny(gray, 40, 140)
    # Morphology to close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Warning: no contours found -> will fallback to center crop.")
        # final fallback will be returned below
    else:
        img_area = float(h_img * w_img)
        # Check candidate quads first
        quad_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.005 * img_area:  # ignore tiny contours
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                # compute bounding rect for aspect ratio check
                x,y,w,h = cv2.boundingRect(approx)
                aspect = w / float(h) if h != 0 else 0
                # Favor near-square and reasonably large squares
                score = (min(aspect, 1.0/aspect)) * (area / img_area)  # higher => better
                quad_candidates.append((score, area, approx.reshape(4,2).astype(np.float32), (x,y,w,h)))

        # Sort by score desc (higher score = more square-like and large)
        if quad_candidates:
            quad_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            best = quad_candidates[0]
            approx_pts = best[2]
            x,y,w,h = best[3]
            print(f"Selected quad contour bbox: x={x}, y={y}, w={w}, h={h}, area_ratio={best[1]/img_area:.3f}")
            # convert bbox into a square bbox centered on this quad
            sx, sy, sw, sh = _square_from_bbox(x,y,w,h, w_img, h_img, pad=4)
            return (sx, sy, sw, sh)

        # If no quad candidates found, try to find large near-square bounding boxes of other contours
        rect_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.01 * img_area:
                continue
            x,y,w,h = cv2.boundingRect(cnt)
            aspect = w / float(h) if h != 0 else 0
            if 0.6 <= aspect <= 1.6:
                rect_candidates.append((area, (x,y,w,h)))
        if rect_candidates:
            rect_candidates.sort(key=lambda x: x[0], reverse=True)
            x,y,w,h = rect_candidates[0][1]
            print(f"Selected rect candidate bbox: x={x}, y={y}, w={w}, h={h}, area_ratio={rect_candidates[0][0]/img_area:.3f}")
            sx, sy, sw, sh = _square_from_bbox(x,y,w,h, w_img, h_img, pad=6)
            return (sx, sy, sw, sh)

        # Otherwise try minAreaRect on large contours to capture rotated boards
        minarea_best = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.01 * img_area:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx,cy),(rw,rh), angle = rect
            if rw == 0 or rh == 0:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            # prefer more square-like minAreaRect
            if aspect < 2.5:
                # derive bounding square from this rotated rect's bounding box
                box = cv2.boxPoints(rect).astype(np.int32)
                x_min = int(box[:,0].min())
                y_min = int(box[:,1].min())
                x_max = int(box[:,0].max())
                y_max = int(box[:,1].max())
                w = x_max - x_min
                h = y_max - y_min
                if w > 10 and h > 10:
                    side = max(w, h)
                    sx, sy, sw, sh = _square_from_bbox(x_min, y_min, w, h, w_img, h_img, pad=6)
                    minarea_best = (sx, sy, sw, sh)
                    break
        if minarea_best is not None:
            print(f"Selected minAreaRect fallback bbox: {minarea_best}")
            return minarea_best

    # Final fallback: centered square (use the smaller image dimension minus margin)
    margin = int(min(h_img, w_img) * 0.04)  # leave small margin from edges
    side = min(h_img, w_img) - 2 * margin
    cx, cy = w_img // 2, h_img // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    print("No suitable contour found; using centered fallback square.")
    return (int(x1), int(y1), int(side), int(side))

def slice_board(image: np.ndarray, bounding_box: Tuple[int,int,int,int]) -> np.ndarray:
    print("Slicing board from image...")
    x, y, w, h = bounding_box
    h_img, w_img = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    board_image = image[y1:y2, x1:x2].copy()
    if board_image.size == 0:
        raise ValueError("Empty crop produced in slice_board(). Check bounding box.")
    return board_image

def slice_and_preprocess_board(board_image: np.ndarray) -> List[np.ndarray]:
    print("Slicing and preprocessing 64 squares...")
    # 1. Resize the board to a standard square size (BOARD_SIZE_PX x BOARD_SIZE_PX)
    # If board_image isn't square, resize to BOARD_SIZE_PX x BOARD_SIZE_PX directly (stretches)
    # Usually our find_chessboard_contour already returned a square crop.
    resized_board = cv2.resize(board_image, (BOARD_SIZE_PX, BOARD_SIZE_PX), interpolation=cv2.INTER_AREA)

    model_inputs = []
    # 2. Loop rows (rank 8 -> rank1 is fine; we keep row 0..7)
    for row in range(8):
        for col in range(8):
            y1 = row * MODEL_SQUARE_SIZE
            y2 = (row + 1) * MODEL_SQUARE_SIZE
            x1 = col * MODEL_SQUARE_SIZE
            x2 = (col + 1) * MODEL_SQUARE_SIZE
            square = resized_board[y1:y2, x1:x2]

            # Convert BGR -> RGB
            square_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)

            # Normalize to 0..1 float32
            square_normalized = square_rgb.astype(np.float32) / 255.0

            model_inputs.append(square_normalized)

    print(f"Generated {len(model_inputs)} preprocessed squares.")
    return model_inputs

def create_debug_montage(model_inputs: List[np.ndarray], grid_size: int = 8) -> np.ndarray:
    rows_list = []
    for i in range(grid_size):
        row_images = []
        for j in range(grid_size):
            img = model_inputs[i * grid_size + j]
            img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            row_images.append(img_bgr)
        rows_list.append(np.hstack(row_images))
    montage = np.vstack(rows_list)
    return montage

# --- Main execution (Colab) ---
def main():
    print("Please upload your chessboard image (phone screenshot or photo):")
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded. Exiting.")
        return

    filename = list(uploaded.keys())[0]
    file_bytes = uploaded[filename]
    nparr = np.frombuffer(file_bytes, np.uint8)
    raw_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if raw_image is None:
        print(f"Error: Could not decode image '{filename}'")
        return

    # Run improved detection
    bounding_box = find_chessboard_contour(raw_image)
    if bounding_box is None:
        print("Could not find chessboard. Exiting.")
        return

    # Slice and preprocess
    board_image = slice_board(raw_image, bounding_box)
    model_inputs = slice_and_preprocess_board(board_image)

    # Visualizations (Colab)
    x, y, w, h = bounding_box
    viz = raw_image.copy()
    cv2.rectangle(viz, (x, y), (x + w, y + h), (0, 255, 0), 3)
    print("--- 1. Original Image with Detected Board (green box) ---")
    cv2_imshow(viz)

    print("\n--- 2. Sliced Board (cropped) ---")
    cv2_imshow(board_image)

    print(f"\n--- 3. Processed 8x8 Grid ({BOARD_SIZE_PX}x{BOARD_SIZE_PX}) ---")
    montage = create_debug_montage(model_inputs)
    cv2_imshow(montage)

    print("\nProcessing complete.")
    print("The variable 'model_inputs' is a list of 64 numpy arrays (64x64x3 float32), normalized 0..1.")
    # Optional: save minimal debug outputs
    os.makedirs("debug_outputs", exist_ok=True)
    cv2.imwrite(os.path.join("debug_outputs", "detected_box.png"), viz)
    cv2.imwrite(os.path.join("debug_outputs", "sliced_board.png"), board_image)
    cv2.imwrite(os.path.join("debug_outputs", "montage.png"), montage)

if __name__ == "__main__":
    main()
