import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd

image_size = (128, 128)
batch_size = 32
epochs = 50
base_directory = r'C:\Users\yashm\Downloads\archive\sign_data\train'
model_path = r"D:\signature_verification\signature_verification_model.keras"

def load_images_from_multiple_folders(base_directory):
    images, labels = [], []
    for signature_folder in os.listdir(base_directory):
        signature_path = os.path.join(base_directory, signature_folder)
        if os.path.isdir(signature_path):
            original_folder = os.path.join(signature_path, 'original')
            if os.path.exists(original_folder):
                original_images, original_labels = load_images_and_labels(original_folder, 0)
                images.extend(original_images)
                labels.extend(original_labels)
            forged_folder = os.path.join(signature_path, 'forged')
            if os.path.exists(forged_folder):
                forged_images, forged_labels = load_images_and_labels(forged_folder, 1)
                images.extend(forged_images)
                labels.extend(forged_labels)
    return np.array(images), np.array(labels)

def load_images_and_labels(directory, label):
    images, labels = [], []
    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)
            img = img / 255.0
            images.append(img)
            labels.append(label)
    return images, labels

images, labels = load_images_from_multiple_folders(base_directory)
images = images.reshape(images.shape[0], image_size[0], image_size[1], 1)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
y_train, y_val = to_categorical(y_train, 2), to_categorical(y_val, 2)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
model.save(model_path)

train_loss, train_acc = model.evaluate(X_train, y_train)
print(f'Training accuracy: {train_acc * 100:.2f}%')

avg_train_loss = np.mean(history.history['loss'])
avg_val_loss = np.mean(history.history['val_loss'])
avg_train_acc = np.mean(history.history['accuracy'])
avg_val_acc = np.mean(history.history['val_accuracy'])

print(f"Average Training Loss: {avg_train_loss:.4f}")
print(f"Average Validation Loss: {avg_val_loss:.4f}")
print(f"Average Training Accuracy: {avg_train_acc * 100:.2f}%")
print(f"Average Validation Accuracy: {avg_val_acc * 100:.2f}%")

avg_results = pd.DataFrame({
    "Average Training Loss": [avg_train_loss],
    "Average Validation Loss": [avg_val_loss],
    "Average Training Accuracy": [avg_train_acc * 100],
    "Average Validation Accuracy": [avg_val_acc * 100]
})

print("\nAverage Training and Validation Results:")
print(avg_results)

def predict_signature(model_path, image_path):
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to read image file at {image_path}")
        return
    
    img = cv2.resize(img, image_size) / 255.0
    img = img.reshape(1, image_size[0], image_size[1], 1)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    result = "Genuine" if predicted_class == 0 else "Forged"

    print(f'Prediction: {result}, Confidence: {confidence:.2f}%')

