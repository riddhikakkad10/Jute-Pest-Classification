import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Define the paths
dataset_path = '/Users/riddhikakkad/Desktop/Jute_Pest_Dataset'
augmented_path = '/Users/riddhikakkad/Desktop/Jute_Pest_Dataset_Augmented'
os.makedirs(augmented_path, exist_ok=True)

# Function to preprocess and resize the image
def preprocess_and_resize_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Resize the image to 224x224
    img_resized = cv2.resize(img, (224, 224))
    
    return img_resized

# Function to load and preprocess images from a directory
def load_and_preprocess_images(directory):
    images = []
    labels = []
    class_names = os.listdir(directory)
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            
            # Preprocess and resize the image
            img_resized = preprocess_and_resize_image(img_path)
            images.append(img_resized)
            labels.append(class_name)
    
    return np.array(images), np.array(labels), class_names

# Load and preprocess images
train_images, train_labels, class_names = load_and_preprocess_images(os.path.join(dataset_path, 'train'))

# ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to save augmented images
def save_augmented_images(images, labels, class_names):
    for i, (img, label) in enumerate(zip(images, labels)):
        # Create directory for each class if it doesn't exist
        class_dir = os.path.join(augmented_path, class_names[label])
        os.makedirs(class_dir, exist_ok=True)
        
        # Reshape image for ImageDataGenerator
        img = img.reshape((1,) + img.shape)
        
        # Generate 20 augmented images
        j = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir=class_dir, save_prefix='aug', save_format='jpg'):
            j += 1
            if j >= 20:
                break

# Convert string labels to integer indices
label_to_index = {name: idx for idx, name in enumerate(class_names)}
train_labels_idx = np.array([label_to_index[label] for label in train_labels])

# Save augmented images
save_augmented_images(train_images, train_labels_idx, class_names)

# One-hot encode the labels
train_labels_cat = to_categorical(train_labels_idx, num_classes=len(class_names))

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels_cat, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_images, test_labels, _ = load_and_preprocess_images(os.path.join(dataset_path, 'test'))
test_labels_idx = np.array([label_to_index[label] for label in test_labels])
test_labels_cat = to_categorical(test_labels_idx, num_classes=len(class_names))

loss, accuracy = model.evaluate(test_images, test_labels_cat)
print(f'Test accuracy: {accuracy:.2f}')

print("Data augmentation and CNN training completed.")
