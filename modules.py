import cv2
import numpy as np
from skimage.feature import hog

IMAGE_SIZE = 64

def preprocess_image(image_bgr_or_gray):
    """Convert a photo containing ONE dark handwritten character on light paper
    into a centered 64x64 binary image. Returns None for blank/invalid images.
    """
    if image_bgr_or_gray is None:
        return None

    if len(image_bgr_or_gray.shape) == 3:
        gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr_or_gray.copy()

    # Reduce camera noise while preserving character edges.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Assume dark ink on light paper.
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove tiny isolated noise components.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.zeros_like(binary)
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 3:
            cleaned[labels == i] = 255

    coords = cv2.findNonZero(cleaned)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    crop = cleaned[y:y+h, x:x+w]

    # Preserve aspect ratio and place character on a square canvas.
    side = max(w, h) + 16
    canvas = np.zeros((side, side), dtype=np.uint8)
    x_offset = (side - w) // 2
    y_offset = (side - h) // 2
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = crop

    resized = cv2.resize(canvas, (IMAGE_SIZE, IMAGE_SIZE),
                         interpolation=cv2.INTER_AREA)
    return resized

def extract_features(binary_image):
    """Extract Histogram of Oriented Gradients (HOG) features."""
    if binary_image is None:
        raise ValueError("Could not find a character in the uploaded image.")

    return hog(
        binary_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    ).astype(np.float32)
