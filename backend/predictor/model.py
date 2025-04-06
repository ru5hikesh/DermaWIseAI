import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ================================
# Paths & Config
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "images")
CSV_FILE = os.path.join(BASE_DIR, "dataset", "HAM10000_metadata.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "skin_disease_model.h5")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50

# ================================
# Load & Prepare Data
# ================================
data = pd.read_csv(CSV_FILE)
data['image_path'] = data['image_id'].apply(lambda x: os.path.join(DATASET_DIR, f"{x}.jpg"))
data = data[data['image_path'].apply(os.path.exists)]  # Only valid paths

labels = data['dx'].values
image_paths = data['image_path'].values

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(np.unique(labels_encoded))

def preprocess_images(image_paths, labels, img_size=(128, 128)):
    images, valid_labels = [], []
    for path, label in zip(image_paths, labels):
        try:
            img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_array)
            valid_labels.append(label)
        except Exception as e:
            print(f"Skipping {path}: {e}")
    return np.array(images), np.array(valid_labels)

X, y = preprocess_images(image_paths, labels_encoded, IMG_SIZE)
X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

y_train = tf.keras.utils.to_categorical(y_train_raw, num_classes)
y_test = tf.keras.utils.to_categorical(y_test_raw, num_classes)

# ================================
# Handle Imbalanced Classes
# ================================
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_raw), y=y_train_raw)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# ================================
# CNN Model
# ================================
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
model = create_cnn_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ================================
# Data Augmentation
# ================================
datagen = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# ================================
# Callbacks
# ================================
checkpoint_cb = ModelCheckpoint(MODEL_PATH, save_best_only=True)
early_stop_cb = EarlyStopping(patience=5, restore_best_weights=True)

# ================================
# Training
# ================================
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict,
    callbacks=[checkpoint_cb, early_stop_cb]
)

# ================================
# Evaluation
# ================================
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

# ================================
# Confusion Matrix
# ================================
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

print("Classification Report:\n", classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))
