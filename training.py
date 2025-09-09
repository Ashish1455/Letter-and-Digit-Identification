import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_recall_fscore_support, top_k_accuracy_score
)
import time
from datetime import datetime

from data_preprocessing import DataPreprocessor
from model import ResNet50CharacterModel, DeepCNNModel

class ModelTrainer:
    """Handles training and evaluation of ResNet50 and Deep CNN models"""

    def __init__(self, data_path="data/augmented_images1", target_size=(128, 128)):
        self.data_path = data_path
        self.target_size = target_size
        self.preprocessor = DataPreprocessor(data_path, target_size)
        self.trained_models = {}
        self.evaluation_results = {}

        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)

    def prepare_data(self, test_size=0.2, validation_size=0.1, augment=True):
        """Prepare training, validation, and test datasets"""
        print("DATA PREPARATION")

        # Prepare data splits
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.preprocessor.prepare_data(
            test_size=test_size,
            validation_size=validation_size,
            augment=augment
        )

        # Visualize sample images
        print("\nVisualizing sample images...")
        self.preprocessor.visualize_samples(X_train, y_train)

        # Save class names for later use
        with open('models/class_names.txt', 'w') as f:
            for class_name in self.preprocessor.class_names:
                f.write(f"{class_name}\n")

        print(f"✓ Class names saved to models/class_names.txt")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train_resnet50_model(self, train_data, val_data, epochs=50, batch_size=32):
        """Train ResNet50 pretrained model"""
        print("TRAINING RESNET50 MODEL")

        # Create model
        model = ResNet50CharacterModel(
            input_shape=(*self.target_size, 3),
            num_classes=len(self.preprocessor.class_names)
        )

        # Create and compile model
        model.create_model(freeze_backbone=True)
        model.compile_model(learning_rate=0.001)

        # Print model summary
        print(f"\nResNet50 Model Architecture:")
        model.get_model_summary()

        # Get callbacks
        callbacks = model.get_callbacks("best_resnet50_model.h5")

        # Prepare training data
        X_train, y_train = train_data
        X_val, y_val = val_data

        print(f"\nStarting training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")

        # Train model
        history = model.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        # Store training history
        model.history = history

        # Plot and save training history
        self.plot_training_history(history, "ResNet50_training_history")

        # Store trained model
        self.trained_models["ResNet50"] = model

        print(f"\n✅ ResNet50 model training completed!")
        print(f"✓ Best model saved to: models/best_resnet50_model.h5")

        return model

    def train_deep_cnn_model(self, train_data, val_data, epochs=50, batch_size=32):
        """Train Deep CNN custom model"""
        print("TRAINING DEEP CNN MODEL")

        # Create model
        model = DeepCNNModel(
            input_shape=(*self.target_size, 3),
            num_classes=len(self.preprocessor.class_names)
        )

        # Create and compile model
        model.create_model()
        model.compile_model(learning_rate=0.001)

        # Print model summary
        print(f"\nDeep CNN Model Architecture:")
        model.get_model_summary()

        # Get callbacks
        callbacks = model.get_callbacks("best_deep_cnn_model.h5")

        # Prepare training data
        X_train, y_train = train_data
        X_val, y_val = val_data

        print(f"\nStarting training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")

        # Train model
        history = model.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        # Store training history
        model.history = history

        # Plot and save training history
        self.plot_training_history(history, "DeepCNN_training_history")

        # Store trained model
        self.trained_models["DeepCNN"] = model

        print(f"\n✅ Deep CNN model training completed!")
        print(f"✓ Best model saved to: models/best_deep_cnn_model.h5")

        return model

    def plot_training_history(self, history, title="training_history"):
        """Plot and save training history"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy plot
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title(f'{title} - Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss plot
        axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title(f'{title} - Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Top-3 Accuracy plot
        if 'top_3_accuracy' in history.history:
            axes[1, 0].plot(history.history['top_3_accuracy'], label='Training Top-3 Acc', linewidth=2)
            axes[1, 0].plot(history.history['val_top_3_accuracy'], label='Validation Top-3 Acc', linewidth=2)
            axes[1, 0].set_title(f'{title} - Top-3 Accuracy', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-3 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Top-3 Accuracy\nNot Available', 
                          ha='center', va='center', fontsize=14)
            axes[1, 0].set_title('Top-3 Accuracy')

        # Learning rate plot (if available)
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], label='Learning Rate', linewidth=2, color='red')
            axes[1, 1].set_title(f'{title} - Learning Rate', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Show training progress instead
            epochs = len(history.history['accuracy'])
            axes[1, 1].plot(range(1, epochs + 1), history.history['accuracy'], 
                          label='Training Progress', linewidth=2, color='green')
            axes[1, 1].set_title('Training Progress', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'results/{title}.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✓ Training history saved to: results/{title}.png")

    def evaluate_model(self, model, test_data, model_name):
        """Evaluate a single model comprehensively"""

        X_test, y_test = test_data

        print(f"\nEvaluating {model_name}...")

        # Time the inference
        start_time = time.time()
        y_pred_proba = model.model.predict(X_test, verbose=0)
        inference_time = time.time() - start_time

        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Calculate various metrics
        accuracy = accuracy_score(y_true, y_pred)
        top3_accuracy = top_k_accuracy_score(y_test, y_pred_proba, k=3)
        top5_accuracy = top_k_accuracy_score(y_test, y_pred_proba, k=5)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Model size calculation
        model_size_mb = 0
        try:
            temp_path = f"temp_{model_name.lower()}.h5"
            model.model.save(temp_path)
            model_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
            os.remove(temp_path)
        except Exception as e:
            print(f"Could not calculate model size: {e}")

        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'top3_accuracy': top3_accuracy,
            'top5_accuracy': top5_accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'per_class_support': support,
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'inference_time': inference_time,
            'avg_inference_per_sample': inference_time / len(X_test),
            'model_size_mb': model_size_mb,
            'num_parameters': model.model.count_params()
        }

        self.evaluation_results[model_name] = results

        # Print detailed results
        print(f"📊 EVALUATION RESULTS FOR {model_name}")
        print(f"{'─' * 50}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        print(f"Weighted F1-Score: {weighted_f1:.4f}")
        print(f"Inference Time: {inference_time:.3f}s ({len(X_test)} samples)")
        print(f"Avg Time per Sample: {inference_time/len(X_test)*1000:.2f}ms")
        print(f"Model Size: {model_size_mb:.2f}MB")
        print(f"Parameters: {model.model.count_params():,}")

        return results

    def evaluate_all_models(self, test_data):
        """Evaluate all trained models"""

        print("MODEL EVALUATION")

        for model_name, model in self.trained_models.items():
            self.evaluate_model(model, test_data, model_name)

        # Generate comparison
        if len(self.evaluation_results) >= 2:
            self.compare_models()

        # Plot confusion matrices
        self.plot_confusion_matrices()

    def compare_models(self):
        """Create comprehensive model comparison"""

        print("MODEL COMPARISON")

        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Top-3 Accuracy': results['top3_accuracy'],
                'Top-5 Accuracy': results['top5_accuracy'],
                'Macro F1': results['macro_f1'],
                'Weighted F1': results['weighted_f1'],
                'Inference Time (s)': results['inference_time'],
                'Avg Time/Sample (ms)': results['avg_inference_per_sample'] * 1000,
                'Model Size (MB)': results['model_size_mb'],
                'Parameters': results['num_parameters']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

        print("\n📋 PERFORMANCE COMPARISON")
        print("─" * 100)
        print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

        # Save comparison to CSV
        comparison_df.to_csv('results/model_comparison.csv', index=False)
        print(f"\n✓ Comparison saved to: results/model_comparison.csv")

        return comparison_df

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all evaluated models"""

        n_models = len(self.evaluation_results)
        if n_models == 0:
            print("No models evaluated yet")
            return

        # Create subplot layout
        if n_models == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
            if n_models == 1:
                axes = [axes]

        for idx, (model_name, results) in enumerate(self.evaluation_results.items()):
            cm = results['confusion_matrix']

            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Plot confusion matrix (simplified for readability with many classes)
            im = axes[idx].imshow(cm_normalized, interpolation='nearest', cmap='Blues')

            axes[idx].set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.3f}', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted Class')
            axes[idx].set_ylabel('True Class')

            # Add colorbar
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✓ Confusion matrices saved to: results/confusion_matrices.png")


def run_training_pipeline():
    """Complete training pipeline for ResNet50 and Deep CNN models"""

    print("🔤 HANDWRITTEN CHARACTER RECOGNITION - TRAINING PIPELINE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize trainer
    trainer = ModelTrainer()

    try:
        # Prepare data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.prepare_data(augment=True)

        # Train ResNet50 model
        print("🧠 TRAINING RESNET50 PRETRAINED MODEL")

        resnet_model = trainer.train_resnet50_model(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            epochs=50,
            batch_size=32
        )

        # Train Deep CNN model
        print("🔧 TRAINING DEEP CNN CUSTOM MODEL")

        deep_cnn_model = trainer.train_deep_cnn_model(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            epochs=50,
            batch_size=32
        )

        # Evaluate all models
        trainer.evaluate_all_models((X_test, y_test))

        # Final summary
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")

        return trainer

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the complete training pipeline
    trainer = run_training_pipeline()
