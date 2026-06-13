# AI-INTERNSHIP---FRUIT-IMAGE-CLASSIFIER-
"""
Image Classifier for Fruits using CNN
Install dependencies:
    pip install tensorflow numpy matplotlib pillow scikit-learn

This script:
  - Builds a CNN model
  - Trains on synthetic fruit data (or your own folder)
  - Evaluates and predicts fruit type from an image
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
IMG_SIZE    = 64          # resize all images to 64x64
NUM_CLASSES = 3           # apple, banana, orange
EPOCHS      = 10
BATCH_SIZE  = 16
CLASS_NAMES = ['apple', 'banana', 'orange']


# ─────────────────────────────────────────
# OPTION A – SYNTHETIC DATA (runs without any dataset)
# ─────────────────────────────────────────

def generate_synthetic_data(samples_per_class=100):
    """Generate simple colored blobs to mimic fruits."""
    X, y = [], []

    np.random.seed(42)
    # Apple  → mostly red
    # Banana → mostly yellow
    # Orange → mostly orange

    color_map = [
        [200, 30, 30],   # apple  – red
        [230, 210, 30],  # banana – yellow
        [230, 120, 30],  # orange – orange
    ]

    for label, base_color in enumerate(color_map):
        for _ in range(samples_per_class):
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            # Random noise background
            img[:] = np.random.randint(200, 256, (IMG_SIZE, IMG_SIZE, 3))
            # Draw a rough circular blob in the fruit color
            cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
            radius = np.random.randint(20, 28)
            for i in range(IMG_SIZE):
                for j in range(IMG_SIZE):
                    if (i - cx)**2 + (j - cy)**2 < radius**2:
                        noise = np.random.randint(-20, 20, 3)
                        pixel = np.clip(np.array(base_color) + noise, 0, 255)
                        img[i, j] = pixel
            X.append(img)
            y.append(label)

    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y)
    return X, y


# ─────────────────────────────────────────
# OPTION B – LOAD FROM FOLDER (use your own images)
# ─────────────────────────────────────────

def load_from_folder(data_dir):
    """
    Expects folder structure:
        data_dir/
            apple/   *.jpg
            banana/  *.jpg
            orange/  *.jpg
    """
    X, y = [], []
    for label, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            continue
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, fname)
                try:
                    img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                    X.append(np.array(img, dtype='float32') / 255.0)
                    y.append(label)
                except Exception as e:
                    print(f"Skipping {fname}: {e}")
    return np.array(X), np.array(y)


# ─────────────────────────────────────────
# BUILD CNN MODEL
# ─────────────────────────────────────────

def build_model():
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Classifier head
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────

def train(X, y):
    y_cat = to_categorical(y, NUM_CLASSES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    model = build_model()
    model.summary()

    print("\n🚀 Training started...\n")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

    # Plot training curves
    plot_history(history)

    return model, X_test, y_test


# ─────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'],    label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0].set_title('Accuracy'); axes[0].legend()

    axes[1].plot(history.history['loss'],    label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss'); axes[1].legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    print("📊 Training curves saved as training_curves.png")


# ─────────────────────────────────────────
# PREDICT A SINGLE IMAGE
# ─────────────────────────────────────────

def predict_image(model, image_path):
    """Predict the fruit class for a given image file."""
    img = Image.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype='float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)          # shape: (1, 64, 64, 3)

    predictions = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    print(f"\n🍎 Predicted Fruit : {predicted_class}")
    print(f"   Confidence      : {confidence:.2f}%")
    for cls, prob in zip(CLASS_NAMES, predictions):
        print(f"   {cls:10s}: {prob*100:.1f}%")

    return predicted_class, confidence


# ─────────────────────────────────────────
# SAVE / LOAD MODEL
# ─────────────────────────────────────────

def save_model(model, path='fruit_classifier.h5'):
    model.save(path)
    print(f"💾 Model saved to {path}")

def load_model(path='fruit_classifier.h5'):
    return tf.keras.models.load_model(path)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("🍓 Fruit Image Classifier using CNN")
    print("="*45)

    # ── Choose data source ─────────────────
    USE_REAL_DATA = False          # ← set True and provide path if you have images
    DATA_DIR      = 'fruits_data'  # ← your folder with apple/banana/orange sub-folders

    if USE_REAL_DATA and os.path.exists(DATA_DIR):
        print(f"📂 Loading images from '{DATA_DIR}' ...")
        X, y = load_from_folder(DATA_DIR)
        print(f"   Loaded {len(X)} images.")
    else:
        print("🔧 Generating synthetic fruit data (no real images needed) ...")
        X, y = generate_synthetic_data(samples_per_class=150)
        print(f"   Generated {len(X)} synthetic images.")

    # ── Train ──────────────────────────────
    model, X_test, y_test = train(X, y)

    # ── Save model ────────────────────────
    save_model(model)

    # ── Predict on a real image ───────────
    # Uncomment below and provide a real image path to test:
    # predict_image(model, 'my_apple.jpg')

    print("\n✅ Done! Model trained and saved as fruit_classifier.h5")
    print("   To predict a new image, call: predict_image(model, 'your_image.jpg')")
