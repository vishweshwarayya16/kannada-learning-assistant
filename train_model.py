from pathlib import Path
import json
import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from modules import preprocess_image, extract_features

ROOT = Path(__file__).parent
IMAGE_ROOT = ROOT / "dataset" / "images"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

def main():
    X, y = [], []
    class_counts = {}

    if not IMAGE_ROOT.exists():
        raise SystemExit(f"Dataset folder not found: {IMAGE_ROOT}")

    for class_dir in sorted(IMAGE_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name
        count = 0

        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            processed = preprocess_image(image)
            if processed is None:
                print(f"Skipping blank image: {image_path}")
                continue

            X.append(extract_features(processed))
            y.append(label)
            count += 1

        if count:
            class_counts[label] = count

    if len(class_counts) < 2:
        raise SystemExit("Add images for at least TWO character classes.")

    too_small = {k: v for k, v in class_counts.items() if v < 5}
    if too_small:
        details = ", ".join(f"{k}: {v}" for k, v in too_small.items())
        raise SystemExit(
            "Each class needs at least 5 usable images for this starter "
            f"train/test split. Current counts: {details}"
        )

    X = np.asarray(X)
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=5, gamma="scale",
            probability=True, class_weight="balanced",
            random_state=42)
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nDataset counts:")
    for label, count in sorted(class_counts.items()):
        print(f"  {label}: {count}")
    print(f"\nTest accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("\nConfusion matrix (class order shown below):")
    print(model.classes_)
    print(confusion_matrix(y_test, predictions, labels=model.classes_))

    joblib.dump(model, MODEL_DIR / "kannada_hog_svm.joblib")
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "classes": model.classes_.tolist(),
            "test_accuracy": float(accuracy),
            "class_counts": class_counts,
            "feature": "HOG",
            "classifier": "SVM"
        }, f, ensure_ascii=False, indent=2)

    print("\nSaved model to model/kannada_hog_svm.joblib")
    print("IMPORTANT: A random split can overestimate performance if the same")
    print("writer appears in both training and test sets. For a stronger")
    print("evaluation, keep writers separate in the final experiment.")

if __name__ == "__main__":
    main()
