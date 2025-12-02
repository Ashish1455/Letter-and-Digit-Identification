import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

class DataPreprocessor:
    """Handles data loading, preprocessing, and augmentation"""

    def __init__(self, data_path="data/raw/augmented_images", target_size=(128, 128)):
        self.data_path = data_path
        self.target_size = target_size
        self.label_encoder = LabelEncoder()
        self.class_names = []

    def load_dataset(self):
        """Load dataset from directory structure"""
        images = []
        labels = []

        print("Loading dataset...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")

        # Expected structure: data/class_name/image.jpg
        class_folders = sorted([f for f in os.listdir(self.data_path) 
                               if os.path.isdir(os.path.join(self.data_path, f))])

        if not class_folders:
            raise ValueError(f"No class folders found in {self.data_path}")

        self.class_names = class_folders
        print(f"Found {len(self.class_names)} classes: {self.class_names[:10]}...")

        total_images = 0
        for class_name in class_folders:
            class_path = os.path.join(self.data_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            print(f"Loading class '{class_name}': {len(image_files)} images")

            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)

                # Load and preprocess image
                image = self.load_and_preprocess_image(image_path)
                if image is not None:
                    images.append(image)
                    labels.append(class_name)
                    total_images += 1

        print(f"Successfully loaded {total_images} images from {len(self.class_names)} classes")
        return np.array(images), np.array(labels)

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize image
            image = cv2.resize(image, self.target_size)

            # Convert to grayscale for character recognition
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Apply adaptive thresholding for better contrast
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )

            # Convert back to 3 channels for compatibility with pretrained models
            image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

            # Normalize pixel values to [0, 1]
            image = image.astype('float32') / 255.0

            return image

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def augment_data(self, X_train, y_train, augment_factor=2):
        """Apply data augmentation techniques"""

        print(f"Applying data augmentation (factor: {augment_factor})...")

        # Configure augmentation parameters optimized for character recognition
        datagen = ImageDataGenerator(
            rotation_range=15,              # Random rotation up to 15 degrees
            width_shift_range=0.1,          # Horizontal shift
            height_shift_range=0.1,         # Vertical shift
            shear_range=0.1,               # Shear transformation
            zoom_range=0.1,                # Random zoom
            horizontal_flip=False,          # No horizontal flip for characters
            brightness_range=[0.8, 1.2],   # Brightness variation
            fill_mode='nearest'
        )

        augmented_images = []
        augmented_labels = []

        # Fit generator on training data
        datagen.fit(X_train)

        # Generate augmented samples
        for i in range(len(X_train)):
            # Original image
            augmented_images.append(X_train[i])
            augmented_labels.append(y_train[i])

            # Generate additional augmented versions
            img = X_train[i].reshape(1, *X_train[i].shape)
            label = y_train[i]

            aug_iter = datagen.flow(img, batch_size=1)
            for _ in range(augment_factor - 1):
                aug_image = next(aug_iter)[0]
                augmented_images.append(aug_image)
                augmented_labels.append(label)

        print(f"Augmentation complete: {len(X_train)} -> {len(augmented_images)} samples")
        return np.array(augmented_images), np.array(augmented_labels)

    def prepare_data(self, test_size=0.2, validation_size=0.1, augment=True):
        """Complete data preparation pipeline"""
        print("DATA PREPARATION PIPELINE")

        # Load raw dataset
        X, y = self.load_dataset()
        print(f"Raw dataset: {len(X)} images, {len(set(y))} classes")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        num_classes = len(self.class_names)

        print(f"Label encoding: {num_classes} classes")
        print(f"Classes: {self.class_names}")

        # Split into train, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_categorical, 
            test_size=test_size, 
            random_state=42, 
            stratify=y_encoded
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=validation_size/(1-test_size), 
            random_state=42, 
            stratify=np.argmax(y_temp, axis=1)
        )

        print(f"Data splits:")
        print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"  Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

        # Apply data augmentation to training set only
        if augment:
            print("\nApplying data augmentation to training set...")
            X_train, y_train = self.augment_data(X_train, y_train, augment_factor=2)
            print(f"Final training set: {X_train.shape[0]} samples")

        print("\nData preparation complete!")
        print(f"Input shape: {X_train.shape[1:]}")
        print(f"Number of classes: {num_classes}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def visualize_samples(self, X, y, num_samples=20):
        """Visualize sample images from each class"""

        print(f"\nGenerating sample visualization...")

        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.ravel()

        # Get unique classes
        unique_classes = np.unique(np.argmax(y, axis=1))

        for i in range(min(num_samples, len(unique_classes), 20)):
            class_idx = unique_classes[i]
            class_samples = X[np.argmax(y, axis=1) == class_idx]

            if len(class_samples) > 0:
                sample_image = class_samples[0]

                # Display image
                axes[i].imshow(sample_image, cmap='gray')
                axes[i].set_title(f'Class: {self.class_names[class_idx]}', fontsize=10)
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(min(num_samples, 20), 20):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Sample visualization saved as 'sample_images.png'")

    def get_class_distribution(self, y):
        """Get class distribution statistics"""
        class_counts = {}
        y_labels = np.argmax(y, axis=1)

        for i, class_name in enumerate(self.class_names):
            count = np.sum(y_labels == i)
            class_counts[class_name] = count

        return class_counts

    def preprocess_single_image(self, image_path):
        """Preprocess a single image for prediction"""
        image = self.load_and_preprocess_image(image_path)
        if image is not None:
            return np.expand_dims(image, axis=0)  # Add batch dimension
        return None
