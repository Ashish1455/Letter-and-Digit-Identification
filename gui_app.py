import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tensorflow.keras.models import load_model


class CharacterRecognitionApp:
    """Tkinter GUI matching Gradio interface style for handwritten character recognition"""

    def __init__(self, root):
        self.root = root
        self.root.title("🔤 Letter and Digit Recognition")
        self.root.geometry("900x700")
        self.root.configure(bg="#f8f9fa")

        # Model configuration (matching your Gradio code)
        self.LOCAL_MODEL = True
        self.MODEL_LOCAL_PATH = "models/best_deep_cnn_model.h5"
        self.CLASS_NAMES_PATH = "models/class_names.txt"

        # Initialize variables
        self.current_image = None
        self.loaded_model = None
        self.class_names = []
        self.photo_reference = None

        # Model dimensions
        self.H = 28
        self.W = 28
        self.CHANNELS = 1
        self.CHANNELS_FIRST = False

        # Load model and class names
        self.load_class_names()
        self.load_model()
        self.setup_gui()

    def load_class_names(self):
        """Load class names for prediction labels (matching Gradio version)"""
        try:
            if os.path.exists(self.CLASS_NAMES_PATH):
                with open(self.CLASS_NAMES_PATH, "r") as f:
                    self.class_names = [line.strip() for line in f if line.strip()]
                print(f"✓ Loaded {len(self.class_names)} class names from file")
            else:
                # Default character classes (0-9, A-Z, a-z) = 62 classes
                digits = [str(i) for i in range(10)]
                uppercase = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
                lowercase = [chr(i) for i in range(ord('a'), ord('z') + 1)]
                self.class_names = digits + uppercase + lowercase
                print(f"✓ Using default {len(self.class_names)} class names")
        except Exception as e:
            print(f"Error loading class names: {e}")

    def load_model(self):
        """Load the machine learning model (matching Gradio version)"""
        try:
            if os.path.exists(self.MODEL_LOCAL_PATH):
                self.loaded_model = load_model(self.MODEL_LOCAL_PATH)

                # Get model input shape (matching Gradio logic)
                input_shape = self.loaded_model.input_shape
                if len(input_shape) == 4:
                    if input_shape[1] in [1, 3]:
                        self.CHANNELS = input_shape[1]
                        self.H, self.W = input_shape[2], input_shape[3]
                        self.CHANNELS_FIRST = True
                    else:
                        self.H, self.W, self.CHANNELS = input_shape[1], input_shape[2], input_shape[3]
                        self.CHANNELS_FIRST = False

                print(f"✓ Model loaded: {input_shape}")
            else:
                print(f"Model file not found: {self.MODEL_LOCAL_PATH}")
                self.loaded_model = None
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.loaded_model = None

    def setup_gui(self):
        """Setup GUI to match Gradio interface style"""
        # Title (matching Gradio)
        title_frame = tk.Frame(self.root, bg="#f8f9fa", height=80)
        title_frame.pack(fill="x", pady=(0, 20))
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="🔤 Letter and Digit Recognition",
            font=('Segoe UI', 20, 'bold'),
            bg="#f8f9fa",
            fg="#2c3e50"
        )
        title_label.pack(expand=True)

        # Description (matching Gradio)
        desc_label = tk.Label(
            self.root,
            text="Upload an image of a single handwritten/drawn character (0-9, A-Z, a-z). The model will predict the character with confidence scores.",
            font=('Segoe UI', 11),
            bg="#f8f9fa",
            fg="#666666",
            wraplength=800,
            justify="center"
        )
        desc_label.pack(pady=(0, 30))

        # Main content frame
        content_frame = tk.Frame(self.root, bg="#f8f9fa")
        content_frame.pack(fill="both", expand=True, padx=40)

        # Left side - Image input (matching Gradio layout)
        left_frame = tk.Frame(content_frame, bg="#f8f9fa")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 20))

        # Image upload area (matching Gradio image component)
        image_container = tk.Frame(left_frame, bg="#ffffff", relief="solid", bd=1)
        image_container.pack(fill="both", expand=True)

        self.image_label = tk.Label(
            image_container,
            text="📁 Click to upload an image\n\nSupported: PNG, JPEG, GIF, BMP\nImage will be displayed here",
            bg="#ffffff",
            fg="#999999",
            font=('Segoe UI', 12),
            justify="center",
            cursor="hand2"
        )
        self.image_label.pack(expand=True, padx=20, pady=20)
        self.image_label.bind("<Button-1>", lambda e: self.upload_image())

        # Button frame for upload and predict buttons
        button_frame = tk.Frame(left_frame, bg="#f8f9fa")
        button_frame.pack(pady=10)

        # Upload button (matching Gradio style)
        upload_btn = tk.Button(
            button_frame,
            text="📁 Upload Image",
            command=self.upload_image,
            font=('Segoe UI', 11),
            bg="#ff6b35",
            fg="white",
            relief="flat",
            padx=20,
            pady=8,
            cursor="hand2"
        )
        upload_btn.pack(side="left", padx=(0, 10))

        # Predict button (new addition)
        self.predict_btn = tk.Button(
            button_frame,
            text="🔍 Predict Character",
            command=self.predict_image,
            font=('Segoe UI', 11, 'bold'),
            bg="#27ae60",
            fg="white",
            relief="flat",
            padx=20,
            pady=8,
            cursor="hand2",
            state="disabled"  # Disabled until image is loaded
        )
        self.predict_btn.pack(side="left")

        # Right side - Outputs (matching Gradio outputs)
        right_frame = tk.Frame(content_frame, bg="#f8f9fa")
        right_frame.pack(side="right", fill="both", padx=(20, 0))

        # Top 3 Predictions with Confidence Scores (matching Gradio Label)
        pred_frame = tk.LabelFrame(
            right_frame,
            text="Top 3 Predictions with Confidence Scores",
            font=('Segoe UI', 12, 'bold'),
            bg="#f8f9fa",
            fg="#2c3e50",
            padx=15,
            pady=10
        )
        pred_frame.pack(fill="x", pady=(0, 15))

        # Predictions display (matching Gradio Label output)
        self.predictions_text = tk.Text(
            pred_frame,
            height=4,
            width=35,
            font=('Segoe UI', 10),
            bg="#ffffff",
            fg="#2c3e50",
            relief="solid",
            bd=1,
            state="disabled",
            wrap=tk.WORD
        )
        self.predictions_text.pack(fill="x", padx=5, pady=5)

        # Predicted Character (matching Gradio Textbox)
        char_frame = tk.LabelFrame(
            right_frame,
            text="Predicted Character",
            font=('Segoe UI', 12, 'bold'),
            bg="#f8f9fa",
            fg="#2c3e50",
            padx=15,
            pady=10
        )
        char_frame.pack(fill="x", pady=(0, 15))

        self.predicted_char_var = tk.StringVar(value="")
        char_entry = tk.Entry(
            char_frame,
            textvariable=self.predicted_char_var,
            font=('Segoe UI', 14),
            bg="#ffffff",
            fg="#2c3e50",
            relief="solid",
            bd=1,
            justify="center",
            state="readonly"
        )
        char_entry.pack(fill="x", padx=5, pady=5)

        # Confidence Score (matching Gradio Number)
        conf_frame = tk.LabelFrame(
            right_frame,
            text="Confidence Score",
            font=('Segoe UI', 12, 'bold'),
            bg="#f8f9fa",
            fg="#2c3e50",
            padx=15,
            pady=10
        )
        conf_frame.pack(fill="x", pady=(0, 15))

        self.confidence_var = tk.StringVar(value="0.0000")
        conf_entry = tk.Entry(
            conf_frame,
            textvariable=self.confidence_var,
            font=('Segoe UI', 14),
            bg="#ffffff",
            fg="#2c3e50",
            relief="solid",
            bd=1,
            justify="center",
            state="readonly"
        )
        conf_entry.pack(fill="x", padx=5, pady=5)

    def upload_image(self):
        """Handle image upload (now without automatic prediction)"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=filetypes
        )

        if filename:
            try:
                # Load image
                image = Image.open(filename)

                # Convert to RGB if needed (matching Gradio image_mode)
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                self.current_image = image.copy()
                self.display_image(image)

                # Enable predict button when image is loaded
                self.predict_btn.config(state="normal")

                # Clear previous results
                self.clear_results()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def clear_results(self):
        """Clear prediction results"""
        self.predictions_text.config(state="normal")
        self.predictions_text.delete(1.0, tk.END)
        self.predictions_text.config(state="disabled")

        self.predicted_char_var.set("")
        self.confidence_var.set("0.0000")

    def display_image(self, image):
        """Display image in GUI (matching Gradio image display)"""
        try:
            # Resize to fit display area (matching Gradio width=500, height=400)
            display_size = (400, 320)
            img_width, img_height = image.size

            # Calculate scaling to fit display area
            scale_w = display_size[0] / img_width
            scale_h = display_size[1] / img_height
            scale = min(scale_w, scale_h)

            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            display_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            self.photo_reference = ImageTk.PhotoImage(display_image)

            # Update label
            self.image_label.config(image=self.photo_reference, text="")
            self.image_label.image = self.photo_reference

        except Exception as e:
            messagebox.showerror("Display Error", f"Failed to display image: {str(e)}")

    def preprocess_image(self, img):
        """Preprocess image for model prediction (matching Gradio preprocess_image)"""
        try:
            # Convert image mode (matching Gradio logic)
            img = img.convert("RGB") if self.CHANNELS == 3 else img.convert("L")

            # Resize to model input size
            img = img.resize((self.W, self.H))

            # Convert to array and normalize
            arr = np.array(img).astype("float32") / 255.0

            # Handle channels
            if self.CHANNELS == 1 and arr.ndim == 2:
                arr = arr[..., np.newaxis]

            # Handle channel order
            if self.CHANNELS_FIRST:
                arr = np.transpose(arr, (2, 0, 1)) if self.CHANNELS > 1 else arr[np.newaxis, ...]
                arr = np.expand_dims(arr, axis=0)
            else:
                arr = np.expand_dims(arr, axis=0)

            return arr

        except Exception as e:
            raise Exception(f"Preprocessing failed: {str(e)}")

    def predict_image(self):
        """Make prediction on image (triggered by button click)"""
        if not self.loaded_model:
            messagebox.showwarning("No Model", "Model not loaded properly.")
            return

        if not self.current_image:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        try:
            # Show processing message
            self.predict_btn.config(text="🔄 Processing...", state="disabled")
            self.root.update_idletasks()

            # Preprocess image (matching Gradio)
            arr = self.preprocess_image(self.current_image)

            # Make prediction (matching Gradio)
            preds = self.loaded_model.predict(arr, verbose=0)[0]

            # Get top predictions (matching Gradio logic)
            top_k = 3
            top_idx = preds.argsort()[-top_k:][::-1]
            top = [(self.class_names[i], float(preds[i])) for i in top_idx]

            # Create label_dict for top 3 (matching Gradio Label output)
            label_dict = {self.class_names[i]: float(preds[i]) for i in top_idx}

            # Get predicted character and confidence (matching Gradio outputs)
            predicted_label = top[0][0]
            predicted_confidence = top[0][1]

            # Update GUI outputs (matching Gradio interface)
            self.update_predictions_display(label_dict)
            self.predicted_char_var.set(predicted_label)
            self.confidence_var.set(f"{predicted_confidence:.4f}")


        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to make prediction: {str(e)}")

        finally:
            # Reset button
            self.predict_btn.config(text="🔍 Predict Character", state="normal")

    def update_predictions_display(self, label_dict):
        """Update predictions display (matching Gradio Label component)"""
        self.predictions_text.config(state="normal")
        self.predictions_text.delete(1.0, tk.END)

        # Sort by confidence and display (matching Gradio num_top_classes=3)
        sorted_predictions = sorted(label_dict.items(), key=lambda x: x[1], reverse=True)

        for i, (char, conf) in enumerate(sorted_predictions, 1):
            confidence_percent = conf * 100
            self.predictions_text.insert(tk.END, f"{i}. {char}: {confidence_percent:.2f}%\n")

        self.predictions_text.config(state="disabled")


def main():
    """Main function matching Gradio demo.launch()"""
    root = tk.Tk()
    app = CharacterRecognitionApp(root)

    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    print("🚀 Character Recognition App launched!")
    root.mainloop()


if __name__ == "__main__":
    main()
