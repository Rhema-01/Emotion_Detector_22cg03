import os
import glob
import argparse
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Default data directory relative to the script
DEFAULT_DATA_DIR = os.path.join("data", "train")
IMG_SIZE = (48, 48)
MODEL_DIR = "models"
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_emotion.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def load_images(data_dir):
def augment_image(image):
    """Applies simple augmentations: horizontal flip and small rotation."""
    if np.random.rand() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT) # Horizontal flip
    
    # Apply a random rotation between -10 and 10 degrees
    angle = np.random.uniform(-10, 10)
    image = image.rotate(angle, resample=Image.BICUBIC, fillcolor=0)

    return image

def load_images(data_dir, augment=False):
    X = []
    y = []
    labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not labels:
        raise RuntimeError(f"No class directories found in {data_dir}")
    for label in labels:
        files = glob.glob(os.path.join(data_dir, label, "*"))
        print(f"  Loading images for: {label}")
        files = glob.glob(os.path.join(data_dir, label, "*.*")) # Allow any extension
        for f in files:
            try:
                img = Image.open(f).convert("L")  # grayscale
                img = img.resize(IMG_SIZE)
                arr = np.asarray(img, dtype=np.float32).flatten()
                X.append(arr)
                
                # Add original image
                X.append(np.asarray(img.resize(IMG_SIZE), dtype=np.float32).flatten())
                y.append(label)

                # Add augmented image
                if augment:
                    img_aug = augment_image(img)
                    X.append(np.asarray(img_aug.resize(IMG_SIZE), dtype=np.float32).flatten())
                    y.append(label)
            except Exception as e:
                print("Skipping", f, e)
    return np.array(X), np.array(y), labels

def main(data_dir):
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found at '{data_dir}'")
        print("Please download the FER2013 dataset and extract the 'train' folder")
        print("to a 'data' directory in your project, or specify the path using --data-dir.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Loading images...")

    X, y_str, labels_str = load_images(data_dir)
    # Set augment=True to double the training data
    X, y_str, labels_str = load_images(data_dir, augment=True)
    if len(X) == 0:
        print("No images found. Put training images in data/<label>/*.jpg")
        return

    # Encode string labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    print("Training MLPClassifier...")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu',
                        solver='adam', max_iter=200, early_stopping=True, random_state=42)

    mlp.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = mlp.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(mlp, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, LE_PATH) # Save LabelEncoder as well
    print(f"\nTraining complete!")
    print(f"Saved model to:         {MODEL_PATH}")
    print(f"Saved scaler to:        {SCALER_PATH}")
    print(f"Saved label encoder to: {LE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an emotion detection model.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"Directory containing the training images, structured as class subfolders. Defaults to '{DEFAULT_DATA_DIR}'.")
    args = parser.parse_args()
    main(args.data_dir)