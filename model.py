import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization,
    Input, Conv2D, MaxPooling2D, Activation
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import os

class ResNet50CharacterModel:
    """ResNet50 pretrained backbone with custom classifier head"""

    def __init__(self, input_shape=(128, 128, 3), num_classes=62):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def create_model(self, freeze_backbone=True):
        """Create ResNet50 model with custom classifier"""

        print("Creating ResNet50 model...")
        print(f"Input shape: {self.input_shape}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Freeze backbone: {freeze_backbone}")

        # Input layer
        inputs = Input(shape=self.input_shape, name='input')

        # ResNet50 backbone (without top layers)
        backbone = ResNet50(
            input_tensor=inputs,
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze backbone weights if specified
        if freeze_backbone:
            backbone.trainable = False
            print("✓ ResNet50 backbone frozen")
        else:
            # Fine-tune only the top layers
            for layer in backbone.layers[:-20]:
                layer.trainable = False
            print("✓ ResNet50 backbone fine-tuning enabled (top 20 layers)")

        # Extract features from backbone
        x = backbone.output

        # Custom classifier head optimized for character recognition
        x = GlobalAveragePooling2D(name='global_avg_pooling')(x)
        x = BatchNormalization(name='bn1')(x)

        # First dense layer
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='dropout1')(x)
        x = BatchNormalization(name='bn2')(x)

        # Second dense layer
        x = Dense(512, activation='relu', name='fc2')(x)
        x = Dropout(0.3, name='dropout2')(x)
        x = BatchNormalization(name='bn3')(x)

        # Third dense layer
        x = Dense(256, activation='relu', name='fc3')(x)
        x = Dropout(0.2, name='dropout3')(x)

        # Output layer for 62 classes (0-9, A-Z, a-z)
        predictions = Dense(
            self.num_classes, 
            activation='softmax', 
            name='predictions'
        )(x)

        # Create final model
        self.model = Model(inputs=inputs, outputs=predictions, name='ResNet50_Character_Recognition')

        print("✓ ResNet50 model created successfully")
        return self.model

    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""

        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        # Create optimizer
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )

        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✓ Model compiled with Adam optimizer (lr={learning_rate})")
        return self.model

    def get_callbacks(self, model_name="best_resnet50_model.h5"):
        """Create training callbacks"""

        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                filepath=f"models/{model_name}",
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]

        return callbacks

    def get_model_summary(self):
        """Display model architecture summary"""
        if self.model:
            self.model.summary()


class DeepCNNModel:
    """Deep CNN custom architecture for character recognition"""

    def __init__(self, input_shape=(128, 128, 3), num_classes=62):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def create_model(self):
        """Create deep CNN architecture optimized for character recognition"""

        print("Creating Deep CNN model...")
        print(f"Input shape: {self.input_shape}")
        print(f"Number of classes: {self.num_classes}")

        # Input layer
        inputs = Input(shape=self.input_shape, name='input')

        # Initial convolution block
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='initial_conv')(inputs)
        x = BatchNormalization(name='initial_bn')(x)
        x = Activation('relu', name='initial_relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='initial_pool')(x)

        # Block 1: Feature extraction
        x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(x)
        x = BatchNormalization(name='block1_bn1')(x)
        x = Activation('relu', name='block1_relu1')(x)

        x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
        x = BatchNormalization(name='block1_bn2')(x)
        x = Activation('relu', name='block1_relu2')(x)

        x = MaxPooling2D((2, 2), name='block1_pool')(x)
        x = Dropout(0.25, name='block1_dropout')(x)

        # Block 2: Deeper feature extraction
        x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
        x = BatchNormalization(name='block2_bn1')(x)
        x = Activation('relu', name='block2_relu1')(x)

        x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
        x = BatchNormalization(name='block2_bn2')(x)
        x = Activation('relu', name='block2_relu2')(x)

        x = Conv2D(128, (3, 3), padding='same', name='block2_conv3')(x)
        x = BatchNormalization(name='block2_bn3')(x)
        x = Activation('relu', name='block2_relu3')(x)

        x = MaxPooling2D((2, 2), name='block2_pool')(x)
        x = Dropout(0.25, name='block2_dropout')(x)

        # Block 3: High-level features
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
        x = BatchNormalization(name='block3_bn1')(x)
        x = Activation('relu', name='block3_relu1')(x)

        x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
        x = BatchNormalization(name='block3_bn2')(x)
        x = Activation('relu', name='block3_relu2')(x)

        x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
        x = BatchNormalization(name='block3_bn3')(x)
        x = Activation('relu', name='block3_relu3')(x)

        x = MaxPooling2D((2, 2), name='block3_pool')(x)
        x = Dropout(0.3, name='block3_dropout')(x)

        # Block 4: Abstract features
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
        x = BatchNormalization(name='block4_bn1')(x)
        x = Activation('relu', name='block4_relu1')(x)

        x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
        x = BatchNormalization(name='block4_bn2')(x)
        x = Activation('relu', name='block4_relu2')(x)

        x = MaxPooling2D((2, 2), name='block4_pool')(x)
        x = Dropout(0.4, name='block4_dropout')(x)

        # Global pooling and classifier
        x = GlobalAveragePooling2D(name='global_avg_pooling')(x)

        # Dense layers with regularization
        x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),name='fc1')(x)
        x = Dropout(0.5, name='fc1_dropout')(x)
        x = BatchNormalization(name='fc1_bn')(x)

        x = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), name='fc2')(x)
        x = Dropout(0.4, name='fc2_dropout')(x)
        x = BatchNormalization(name='fc2_bn')(x)

        x = Dense(256, activation='relu', name='fc3')(x)
        x = Dropout(0.3, name='fc3_dropout')(x)

        # Output layer for 62 classes
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        # Create final model
        self.model = Model(inputs=inputs, outputs=predictions, name='DeepCNN_Character_Recognition')

        print("✓ Deep CNN model created successfully")
        return self.model

    def compile_model(self, learning_rate=0.001):
        """Compile the model"""

        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        # Create optimizer
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )

        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✓ Model compiled with Adam optimizer (lr={learning_rate})")
        return self.model

    def get_callbacks(self, model_name="best_deep_cnn_model.h5"):
        """Create training callbacks"""

        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                filepath=f"models/{model_name}",
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]

        return callbacks

    def get_model_summary(self):
        """Display model architecture summary"""
        if self.model:
            self.model.summary()