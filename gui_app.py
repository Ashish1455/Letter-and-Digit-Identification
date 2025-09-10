import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageOps
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import json
from datetime import datetime

class CharacterRecognitionGUI:
    """GUI application for testing trained character recognition models"""

    def __init__(self, root):
        self.root = root
        self.root.title("🔤 Handwritten Character Recognition - Model Testing")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f8f9fa")

        # Initialize variables
        self.current_image = None
        self.processed_image = None
        self.loaded_model = None
        self.class_names = self.load_class_names()
        self.available_models = self.scan_available_models()

        # Setup GUI components
        self.setup_styles()
        self.create_widgets()
        self.create_menu()

        # Load default model if available
        self.load_default_model()

    def setup_styles(self):
        """Configure modern ttk styles"""

        style = ttk.Style()
        style.theme_use('clam')

        # Configure custom styles with modern colors
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 18, 'bold'), 
                       background='#f8f9fa', 
                       foreground='#2c3e50')

        style.configure('Header.TLabel', 
                       font=('Segoe UI', 12, 'bold'), 
                       background='#f8f9fa', 
                       foreground='#34495e')

        style.configure('Result.TLabel', 
                       font=('Segoe UI', 16, 'bold'), 
                       background='#f8f9fa', 
                       foreground='#27ae60')

        style.configure('Custom.TButton', 
                       font=('Segoe UI', 10, 'bold'),
                       padding=(10, 5))

        style.configure('Model.TCombobox',
                       font=('Segoe UI', 9))

    def load_class_names(self):
        """Load class names for prediction labels"""

        # Try to load from saved file first
        if os.path.exists('models/class_names.txt'):
            try:
                with open('models/class_names.txt', 'r') as f:
                    class_names = [line.strip() for line in f.readlines()]
                    print(f"✓ Loaded {len(class_names)} class names from file")
                    return class_names
            except Exception as e:
                print(f"Error loading class names: {e}")

        # Default character classes (0-9, A-Z, a-z) = 62 classes
        digits = [str(i) for i in range(10)]
        uppercase = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        lowercase = [chr(i) for i in range(ord('a'), ord('z') + 1)]

        class_names = digits + uppercase + lowercase
        print(f"✓ Using default {len(class_names)} class names")
        return class_names

    def scan_available_models(self):
        """Scan for available model files"""
        models = {}

        if os.path.exists('models'):
            for file in os.listdir('models'):
                if file.endswith('.h5') or file.endswith('.keras'):
                    model_path = os.path.join('models', file)
                    # Clean up model name for display
                    display_name = file.replace('best_', '').replace('_model', '').replace('.h5', '').replace('_', ' ').title()
                    models[display_name] = model_path

        return models

    def create_menu(self):
        """Create professional application menu"""

        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image", command=self.load_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Load Custom Model", command=self.load_custom_model)
        file_menu.add_separator()
        file_menu.add_command(label="Save Results", command=self.save_results, accelerator="Ctrl+S")
        file_menu.add_command(label="Save Predictions", command=self.save_prediction_history)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Clear Image", command=self.clear_canvas)
        tools_menu.add_command(label="Reset Application", command=self.reset_all)
        tools_menu.add_command(label="Model Information", command=self.show_model_info)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Refresh Models", command=self.refresh_models)
        view_menu.add_command(label="Show Class Names", command=self.show_class_names)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_command(label="About", command=self.show_about)

        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.load_image())
        self.root.bind('<Control-s>', lambda e: self.save_results())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
        self.root.bind('<F5>', lambda e: self.refresh_models())

    def create_widgets(self):
        """Create main GUI widgets with modern layout"""

        # Main title bar
        title_frame = tk.Frame(self.root, bg="#3498db", height=70)
        title_frame.pack(fill="x", padx=0, pady=0)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame, 
            text="🔤 Handwritten Character Recognition System", 
            font=('Segoe UI', 20, 'bold'),
            bg="#3498db",
            fg="white"
        )
        title_label.pack(expand=True)

        # Main container
        main_frame = tk.Frame(self.root, bg="#f8f9fa")
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        # Left panel - Image display and controls
        left_panel = tk.LabelFrame(
            main_frame, 
            text=" 📷 Image Input & Processing ",
            font=('Segoe UI', 12, 'bold'),
            bg="#f8f9fa", 
            fg="#2c3e50", 
            padx=15, 
            pady=15
        )
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Image display area with border
        image_container = tk.Frame(left_panel, bg="#ffffff", relief="solid", bd=2)
        image_container.pack(fill="both", expand=True, pady=(0, 15))

        self.image_frame = tk.Frame(image_container, bg="#ffffff")
        self.image_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.image_label = tk.Label(
            self.image_frame, 
            text="📁 No image loaded\n\nClick 'Load Image' or 'Draw Character'\nto get started", 
            bg="#ffffff", 
            fg="#7f8c8d", 
            font=('Segoe UI', 14),
            justify="center"
        )
        self.image_label.pack(expand=True)

        # Image control buttons
        control_frame = tk.Frame(left_panel, bg="#f8f9fa")
        control_frame.pack(fill="x", pady=(0, 10))

        # First row of buttons
        button_row1 = tk.Frame(control_frame, bg="#f8f9fa")
        button_row1.pack(fill="x", pady=(0, 5))

        load_btn = ttk.Button(
            button_row1, 
            text="📁 Load Image", 
            command=self.load_image, 
            style='Custom.TButton'
        )
        load_btn.pack(side="left", padx=(0, 10))

        draw_btn = ttk.Button(
            button_row1, 
            text="✏️ Draw Character", 
            command=self.open_drawing_window, 
            style='Custom.TButton'
        )
        draw_btn.pack(side="left", padx=(0, 10))

        clear_btn = ttk.Button(
            button_row1, 
            text="🗑️ Clear", 
            command=self.clear_canvas, 
            style='Custom.TButton'
        )
        clear_btn.pack(side="left")

        # Image info display
        info_frame = tk.LabelFrame(
            left_panel, 
            text=" ℹ️ Image Information ", 
            font=('Segoe UI', 10, 'bold'),
            bg="#f8f9fa", 
            fg="#2c3e50"
        )
        info_frame.pack(fill="x", pady=(10, 0))

        self.image_info_var = tk.StringVar(value="No image loaded")
        info_label = tk.Label(
            info_frame,
            textvariable=self.image_info_var,
            font=('Segoe UI', 9),
            bg="#f8f9fa",
            fg="#34495e",
            justify="left"
        )
        info_label.pack(anchor="w", padx=10, pady=5)

        # Right panel - Model and results
        right_panel = tk.LabelFrame(
            main_frame, 
            text=" 🧠 Model & Prediction Results ",
            font=('Segoe UI', 12, 'bold'),
            bg="#f8f9fa", 
            fg="#2c3e50", 
            padx=15, 
            pady=15
        )
        right_panel.pack(side="right", fill="both", expand=False, padx=(10, 0), ipadx=80)

        # Model selection section
        model_frame = tk.LabelFrame(
            right_panel, 
            text=" 🔧 Model Selection ",
            font=('Segoe UI', 11, 'bold'), 
            bg="#f8f9fa", 
            fg="#2c3e50"
        )
        model_frame.pack(fill="x", pady=(0, 15))

        # Model dropdown
        tk.Label(model_frame, text="Available Models:", font=('Segoe UI', 9), 
                bg="#f8f9fa", fg="#2c3e50").pack(anchor="w", padx=5, pady=(5, 2))

        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=list(self.available_models.keys()),
            state="readonly",
            style='Model.TCombobox',
            width=25
        )
        self.model_combobox.pack(fill="x", padx=5, pady=(0, 10))
        self.model_combobox.bind('<<ComboboxSelected>>', self.on_model_selected)

        # Load custom model button
        custom_model_btn = ttk.Button(
            model_frame, 
            text="📦 Load Custom Model", 
            command=self.load_custom_model, 
            style='Custom.TButton'
        )
        custom_model_btn.pack(fill="x", padx=5, pady=(0, 5))

        # Model status
        self.model_status_var = tk.StringVar(value="No model loaded")
        status_label = tk.Label(
            model_frame,
            textvariable=self.model_status_var,
            font=('Segoe UI', 9, 'italic'),
            bg="#f8f9fa",
            fg="#e74c3c"
        )
        status_label.pack(anchor="w", padx=5, pady=(5, 5))

        # Prediction section
        prediction_frame = tk.Frame(right_panel, bg="#f8f9fa")
        prediction_frame.pack(fill="x", pady=(0, 15))

        self.predict_btn = ttk.Button(
            prediction_frame, 
            text="🔍 Predict Character", 
            command=self.predict_character, 
            style='Custom.TButton',
            state="disabled"
        )
        self.predict_btn.pack(fill="x")

        # Results display
        results_frame = tk.LabelFrame(
            right_panel, 
            text=" 📊 Prediction Results ",
            font=('Segoe UI', 11, 'bold'), 
            bg="#f8f9fa", 
            fg="#2c3e50"
        )
        results_frame.pack(fill="both", expand=True)

        # Top prediction display
        top_pred_frame = tk.Frame(results_frame, bg="#f8f9fa")
        top_pred_frame.pack(fill="x", padx=5, pady=5)

        tk.Label(top_pred_frame, text="Top Prediction:", 
                font=('Segoe UI', 10, 'bold'), 
                bg="#f8f9fa", fg="#2c3e50").pack(anchor="w")

        # Large prediction display
        prediction_display = tk.Frame(results_frame, bg="#ffffff", relief="solid", bd=2)
        prediction_display.pack(fill="x", padx=5, pady=5)

        self.top_prediction_var = tk.StringVar(value="None")
        self.prediction_label = tk.Label(
            prediction_display,
            textvariable=self.top_prediction_var,
            font=('Segoe UI', 36, 'bold'),
            bg="#ffffff",
            fg="#27ae60",
            width=3,
            height=2
        )
        self.prediction_label.pack(side="left", padx=15, pady=15)

        # Confidence display
        confidence_frame = tk.Frame(prediction_display, bg="#ffffff")
        confidence_frame.pack(side="right", fill="y", padx=15, pady=15)

        tk.Label(confidence_frame, text="Confidence:", 
                font=('Segoe UI', 9, 'bold'), 
                bg="#ffffff", fg="#2c3e50").pack(anchor="w")

        self.confidence_var = tk.StringVar(value="0.00%")
        confidence_label = tk.Label(
            confidence_frame,
            textvariable=self.confidence_var,
            font=('Segoe UI', 18, 'bold'),
            bg="#ffffff",
            fg="#27ae60"
        )
        confidence_label.pack()

        # Top 5 predictions table
        tk.Label(results_frame, text="Top 5 Predictions:", 
                font=('Segoe UI', 10, 'bold'), 
                bg="#f8f9fa", fg="#2c3e50").pack(anchor="w", padx=5, pady=(15, 5))

        # Create treeview with custom styling
        tree_frame = tk.Frame(results_frame, bg="#f8f9fa")
        tree_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        self.predictions_tree = ttk.Treeview(
            tree_frame, 
            columns=('Rank', 'Character', 'Confidence'), 
            show='headings', 
            height=6
        )

        # Configure columns
        self.predictions_tree.heading('Rank', text='Rank')
        self.predictions_tree.heading('Character', text='Character')
        self.predictions_tree.heading('Confidence', text='Confidence')

        self.predictions_tree.column('Rank', width=50, anchor='center')
        self.predictions_tree.column('Character', width=80, anchor='center')
        self.predictions_tree.column('Confidence', width=100, anchor='center')

        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.predictions_tree.yview)
        self.predictions_tree.configure(yscrollcommand=scrollbar.set)

        self.predictions_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load a model and image to begin")
        status_bar = tk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief="sunken", 
            anchor="w", 
            font=('Segoe UI', 9),
            bg="#ecf0f1", 
            fg="#2c3e50"
        )
        status_bar.pack(side="bottom", fill="x")

        # Initialize prediction history
        self.prediction_history = []

    def load_default_model(self):
        """Try to load the first available model"""

        if self.available_models:
            # Try to load ResNet50 first, then any other model
            preferred_models = ['Resnet50', 'ResNet50', 'Deep Cnn', 'DeepCNN']

            selected_model = None
            for pref in preferred_models:
                if pref in self.available_models:
                    selected_model = pref
                    break

            if not selected_model:
                selected_model = list(self.available_models.keys())[0]

            self.model_var.set(selected_model)
            self.on_model_selected(None)

    def refresh_models(self):
        """Refresh the list of available models"""
        self.available_models = self.scan_available_models()
        self.model_combobox['values'] = list(self.available_models.keys())
        self.status_var.set(f"Refreshed - Found {len(self.available_models)} models")

    def on_model_selected(self, event):
        """Handle model selection from dropdown"""
        selected = self.model_var.get()
        if selected and selected in self.available_models:
            model_path = self.available_models[selected]
            self.load_model_from_path(model_path, selected)

    def load_model_from_path(self, model_path, model_name):
        """Load model from specified path"""
        try:
            self.loaded_model = load_model(model_path)
            self.model_status_var.set(f"✅ {model_name} loaded")

            # Enable prediction if image is loaded
            if self.current_image:
                self.predict_btn.config(state="normal")

            self.status_var.set(f"Model loaded: {model_name}")

            # Update model info
            params = self.loaded_model.count_params()
            self.status_var.set(f"✅ {model_name} loaded ({params:,} parameters)")

        except Exception as e:
            error_msg = f"❌ Failed to load {model_name}: {str(e)}"
            self.model_status_var.set("❌ Model load failed")
            messagebox.showerror("Model Load Error", error_msg)
            self.status_var.set("Model loading failed")

    def load_image(self):
        """Load an image file with enhanced file type support"""

        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff *.webp"),
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
                # Load and display image
                image = Image.open(filename)

                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                self.current_image = image.copy()

                # Display image
                self.display_image(image)

                # Update image info
                width, height = image.size
                file_size = os.path.getsize(filename) / 1024  # KB
                filename_short = os.path.basename(filename)

                self.image_info_var.set(
                    f"File: {filename_short}\n"
                    f"Size: {width}x{height} pixels\n"
                    f"File size: {file_size:.1f} KB"
                )

                # Enable prediction if model is loaded
                if self.loaded_model:
                    self.predict_btn.config(state="normal")

                self.status_var.set(f"Image loaded: {filename_short}")

            except Exception as e:
                messagebox.showerror("Image Load Error", f"Failed to load image:\n{str(e)}")
                self.status_var.set("Image loading failed")

    def display_image(self, image, target_size=(450, 450)):
        """Display image in the GUI with improved scaling"""

        # Calculate display size maintaining aspect ratio
        img_width, img_height = image.size
        display_width, display_height = target_size

        # Calculate scaling factor
        scale_w = display_width / img_width
        scale_h = display_height / img_height
        scale = min(scale_w, scale_h)

        # Resize image
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        # Use high-quality resampling
        display_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Add subtle border
        display_image = ImageOps.expand(display_image, border=2, fill='#bdc3c7')

        photo = ImageTk.PhotoImage(display_image)

        # Update label
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keep a reference

    def load_custom_model(self):
        """Load a custom model file"""

        filetypes = [
            ("Keras models", "*.h5 *.keras"),
            ("H5 files", "*.h5"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Select a trained model file",
            filetypes=filetypes
        )

        if filename:
            model_name = os.path.basename(filename).replace('.h5', '').replace('.keras', '')
            self.load_model_from_path(filename, f"Custom: {model_name}")

    def preprocess_image(self, image, target_size=(128, 128)):
        """Preprocess image for model prediction with enhanced processing"""

        try:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Resize image maintaining aspect ratio
            img_resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LANCZOS4)

            # Convert to grayscale
            if len(img_resized.shape) == 3:
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_resized

            # Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # Apply adaptive thresholding for better contrast
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )

            # Apply morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Convert back to 3 channels for model compatibility
            processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

            # Normalize pixel values to [0, 1]
            processed = processed.astype('float32') / 255.0

            # Add batch dimension
            processed = np.expand_dims(processed, axis=0)

            return processed

        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")

    def predict_character(self):
        """Predict character from current image with enhanced error handling"""

        if not self.loaded_model:
            messagebox.showwarning("No Model", "Please load a model first.")
            return

        if not self.current_image:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        try:
            self.status_var.set("🔄 Processing image...")
            self.root.update()

            # Preprocess image
            processed_image = self.preprocess_image(self.current_image)

            # Make prediction
            self.status_var.set("🧠 Making prediction...")
            self.root.update()

            predictions = self.loaded_model.predict(processed_image, verbose=0)

            # Get top 5 predictions
            top_indices = np.argsort(predictions[0])[::-1][:5]
            top_confidences = predictions[0][top_indices]

            # Update display
            self.update_prediction_display(top_indices, top_confidences)

            # Store in history
            prediction_entry = {
                'timestamp': datetime.now().isoformat(),
                'top_prediction': self.class_names[top_indices[0]],
                'confidence': float(top_confidences[0]),
                'top_5': [(self.class_names[idx], float(conf)) for idx, conf in zip(top_indices, top_confidences)]
            }
            self.prediction_history.append(prediction_entry)

            self.status_var.set(f"✅ Prediction complete - {self.class_names[top_indices[0]]} ({top_confidences[0]:.1%})")

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            messagebox.showerror("Prediction Error", error_msg)
            self.status_var.set("❌ Prediction failed")

    def update_prediction_display(self, top_indices, top_confidences):
        """Update prediction display with enhanced visualization"""

        # Update top prediction
        if len(top_indices) > 0:
            top_char = self.class_names[top_indices[0]]
            top_conf = top_confidences[0]

            self.top_prediction_var.set(top_char)
            self.confidence_var.set(f"{top_conf:.1%}")

            # Update color based on confidence with more nuanced colors
            if top_conf > 0.9:
                color = "#27ae60"  # Green - Very confident
            elif top_conf > 0.7:
                color = "#2ecc71"  # Light green - Confident
            elif top_conf > 0.5:
                color = "#f39c12"  # Orange - Moderate
            elif top_conf > 0.3:
                color = "#e67e22"  # Dark orange - Low
            else:
                color = "#e74c3c"  # Red - Very low

            self.prediction_label.config(fg=color)

        # Clear and update top 5 predictions
        for item in self.predictions_tree.get_children():
            self.predictions_tree.delete(item)

        rank_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

        for i, (idx, conf) in enumerate(zip(top_indices, top_confidences)):
            char = self.class_names[idx]
            confidence_str = f"{conf:.2%}"
            rank_str = f"{rank_emojis[i]} #{i+1}"

            # Add colored tags based on confidence
            tags = ()
            if conf > 0.7:
                tags = ('high_conf',)
            elif conf > 0.4:
                tags = ('med_conf',)
            else:
                tags = ('low_conf',)

            self.predictions_tree.insert('', 'end', 
                                        values=(rank_str, char, confidence_str),
                                        tags=tags)

        # Configure tag colors
        self.predictions_tree.tag_configure('high_conf', foreground='#27ae60')
        self.predictions_tree.tag_configure('med_conf', foreground='#f39c12')
        self.predictions_tree.tag_configure('low_conf', foreground='#e74c3c')

    def open_drawing_window(self):
        """Open a window for drawing characters"""

        # Create drawing window
        self.draw_window = tk.Toplevel(self.root)
        self.draw_window.title("✏️ Draw Character")
        self.draw_window.geometry("500x600")
        self.draw_window.configure(bg="#f8f9fa")
        self.draw_window.resizable(False, False)

        # Make window modal
        self.draw_window.transient(self.root)
        self.draw_window.grab_set()

        # Instructions
        instruction_frame = tk.Frame(self.draw_window, bg="#3498db", height=60)
        instruction_frame.pack(fill="x")
        instruction_frame.pack_propagate(False)

        tk.Label(
            instruction_frame, 
            text="Draw a character below using your mouse",
            font=('Segoe UI', 14, 'bold'), 
            bg="#3498db",
            fg="white"
        ).pack(expand=True)

        # Canvas for drawing with better styling
        canvas_frame = tk.Frame(self.draw_window, bg="#f8f9fa")
        canvas_frame.pack(pady=20)

        canvas_container = tk.Frame(canvas_frame, bg="#2c3e50", padx=5, pady=5)
        canvas_container.pack()

        self.drawing_canvas = tk.Canvas(
            canvas_container, 
            width=350, 
            height=350, 
            bg="white", 
            cursor="pencil",
            relief="flat"
        )
        self.drawing_canvas.pack()

        # Bind drawing events
        self.drawing_canvas.bind("<Button-1>", self.start_drawing)
        self.drawing_canvas.bind("<B1-Motion>", self.draw)
        self.drawing_canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.drawing = False
        self.last_x = 0
        self.last_y = 0

        # Control buttons with better styling
        button_frame = tk.Frame(self.draw_window, bg="#f8f9fa")
        button_frame.pack(pady=20)

        # First row
        row1 = tk.Frame(button_frame, bg="#f8f9fa")
        row1.pack(pady=(0, 10))

        clear_btn = ttk.Button(
            row1, 
            text="🗑️ Clear Canvas", 
            command=self.clear_drawing_canvas,
            style='Custom.TButton'
        )
        clear_btn.pack(side="left", padx=10)

        predict_btn = ttk.Button(
            row1, 
            text="🔍 Predict", 
            command=self.predict_from_canvas,
            style='Custom.TButton'
        )
        predict_btn.pack(side="left", padx=10)

        # Second row
        row2 = tk.Frame(button_frame, bg="#f8f9fa")
        row2.pack()

        save_btn = ttk.Button(
            row2, 
            text="💾 Save Drawing", 
            command=self.save_drawing,
            style='Custom.TButton'
        )
        save_btn.pack(side="left", padx=10)

        close_btn = ttk.Button(
            row2, 
            text="❌ Close", 
            command=self.close_drawing_window,
            style='Custom.TButton'
        )
        close_btn.pack(side="left", padx=10)

    def start_drawing(self, event):
        """Start drawing on canvas"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        """Draw on canvas with improved brush"""
        if self.drawing:
            self.drawing_canvas.create_oval(
                event.x - 4, event.y - 4, event.x + 4, event.y + 4,
                fill="black", outline="black"
            )

            # Draw line for smooth strokes
            if abs(event.x - self.last_x) < 50 and abs(event.y - self.last_y) < 50:
                self.drawing_canvas.create_line(
                    self.last_x, self.last_y, event.x, event.y,
                    width=8, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE
                )

            self.last_x = event.x
            self.last_y = event.y

    def stop_drawing(self, event):
        """Stop drawing on canvas"""
        self.drawing = False

    def clear_drawing_canvas(self):
        """Clear drawing canvas"""
        if hasattr(self, 'drawing_canvas'):
            self.drawing_canvas.delete("all")

    def predict_from_canvas(self):
        """Predict character from drawn canvas"""

        if not self.loaded_model:
            messagebox.showwarning("No Model", "Please load a model first.")
            return

        try:
            # Convert canvas to image
            canvas_image = self.canvas_to_image()
            if canvas_image:
                self.current_image = canvas_image
                self.display_image(canvas_image)
                self.image_info_var.set("Source: Hand-drawn\nSize: 350x350 pixels")

                # Close drawing window and predict
                self.close_drawing_window()
                self.predict_character()

        except Exception as e:
            messagebox.showerror("Drawing Error", f"Failed to process drawing: {str(e)}")

    def canvas_to_image(self):
        """Convert drawing canvas to PIL Image with improved quality"""

        if not hasattr(self, 'drawing_canvas'):
            return None

        try:
            # Get canvas dimensions
            canvas_width = self.drawing_canvas.winfo_width()
            canvas_height = self.drawing_canvas.winfo_height()

            # Create high-quality image from canvas
            image = Image.new("RGB", (canvas_width, canvas_height), "white")
            draw = ImageDraw.Draw(image)

            # Get all canvas items and draw them
            for item in self.drawing_canvas.find_all():
                item_type = self.drawing_canvas.type(item)
                coords = self.drawing_canvas.coords(item)

                if item_type == "line" and len(coords) >= 4:
                    draw.line(coords, fill="black", width=8)
                elif item_type == "oval" and len(coords) >= 4:
                    draw.ellipse(coords, fill="black")

            return image

        except Exception as e:
            print(f"Error converting canvas to image: {e}")
            return None

    def save_drawing(self):
        """Save the current drawing"""
        if hasattr(self, 'drawing_canvas'):
            canvas_image = self.canvas_to_image()
            if canvas_image:
                filename = filedialog.asksaveasfilename(
                    title="Save Drawing",
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
                )
                if filename:
                    canvas_image.save(filename)
                    messagebox.showinfo("Saved", f"Drawing saved as {filename}")

    def close_drawing_window(self):
        """Close drawing window"""
        if hasattr(self, 'draw_window'):
            self.draw_window.destroy()

    def clear_canvas(self):
        """Clear current image and predictions"""
        self.current_image = None
        self.image_label.config(
            image="", 
            text="📁 No image loaded\n\nClick 'Load Image' or 'Draw Character'\nto get started"
        )
        self.image_label.image = None

        # Clear image info
        self.image_info_var.set("No image loaded")

        # Clear predictions
        self.top_prediction_var.set("None")
        self.confidence_var.set("0.00%")
        self.prediction_label.config(fg="#7f8c8d")

        # Clear predictions tree
        for item in self.predictions_tree.get_children():
            self.predictions_tree.delete(item)

        self.predict_btn.config(state="disabled")
        self.status_var.set("Canvas cleared - Ready for new image")

    def reset_all(self):
        """Reset everything"""
        self.clear_canvas()
        self.loaded_model = None
        self.model_var.set("")
        self.model_status_var.set("No model loaded")
        self.prediction_history = []
        self.status_var.set("Application reset - Load a model to begin")

    def save_results(self):
        """Save current prediction results"""

        if self.top_prediction_var.get() == "None":
            messagebox.showwarning("No Results", "No predictions to save.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                if filename.endswith('.json'):
                    # Save as JSON
                    results = {
                        'timestamp': datetime.now().isoformat(),
                        'model': self.model_var.get(),
                        'top_prediction': self.top_prediction_var.get(),
                        'confidence': self.confidence_var.get(),
                        'top_5_predictions': []
                    }

                    for item in self.predictions_tree.get_children():
                        values = self.predictions_tree.item(item)['values']
                        results['top_5_predictions'].append({
                            'rank': values[0],
                            'character': values[1],
                            'confidence': values[2]
                        })

                    with open(filename, 'w') as f:
                        json.dump(results, f, indent=2)

                else:
                    # Save as text
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("🔤 Character Recognition Results\n")
                        f.write("=" * 50 + "\n")
                        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Model: {self.model_var.get()}\n\n")

                        f.write(f"🏆 Top Prediction: {self.top_prediction_var.get()}\n")
                        f.write(f"📊 Confidence: {self.confidence_var.get()}\n\n")

                        f.write("📋 Top 5 Predictions:\n")
                        f.write("-" * 30 + "\n")

                        for item in self.predictions_tree.get_children():
                            values = self.predictions_tree.item(item)['values']
                            f.write(f"{values[0]} {values[1]}: {values[2]}\n")

                messagebox.showinfo("Saved", f"Results saved to {filename}")
                self.status_var.set(f"Results saved to {os.path.basename(filename)}")

            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save results: {str(e)}")

    def save_prediction_history(self):
        """Save complete prediction history"""
        if not self.prediction_history:
            messagebox.showwarning("No History", "No prediction history to save.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Prediction History",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.prediction_history, f, indent=2)
                messagebox.showinfo("Saved", f"Prediction history saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save history: {str(e)}")

    def show_model_info(self):
        """Show detailed model information"""
        if not self.loaded_model:
            messagebox.showwarning("No Model", "No model loaded.")
            return

        try:
            # Get model information
            params = self.loaded_model.count_params()
            layers = len(self.loaded_model.layers)
            input_shape = self.loaded_model.input_shape
            output_shape = self.loaded_model.output_shape

            info_text = f"""Model Information:

📊 Architecture:
   • Total Parameters: {params:,}
   • Number of Layers: {layers}
   • Input Shape: {input_shape}
   • Output Shape: {output_shape}

🎯 Output Classes: {len(self.class_names)}
📝 Model: {self.model_var.get()}

🔧 Capabilities:
   • Character Recognition: 0-9, A-Z, a-z
   • Image Input: 128x128 pixels
   • Preprocessing: Adaptive thresholding
   • Output: Probability distribution"""

            messagebox.showinfo("Model Information", info_text)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to get model info: {str(e)}")

    def show_class_names(self):
        """Show all supported character classes"""

        # Group classes
        digits = self.class_names[:10]
        uppercase = self.class_names[10:36]
        lowercase = self.class_names[36:62] if len(self.class_names) >= 62 else []

        class_text = f"""Supported Characters ({len(self.class_names)} classes):

🔢 Digits: {' '.join(digits)}

🔤 Uppercase: {' '.join(uppercase)}

🔡 Lowercase: {' '.join(lowercase) if lowercase else 'Not available'}

Total: {len(self.class_names)} character classes"""

        messagebox.showinfo("Character Classes", class_text)

    def show_instructions(self):
        """Show detailed instructions"""

        instructions = """🔤 Handwritten Character Recognition - Instructions

🚀 Getting Started:
   1. Load a model using the dropdown or 'Load Custom Model'
   2. Load an image using 'Load Image' or draw using 'Draw Character'
   3. Click 'Predict Character' to get results

📁 Loading Images:
   • Supported formats: PNG, JPEG, GIF, BMP, TIFF, WebP
   • Best results: Clear, well-lit character images
   • Recommended: Characters should be centered and large

✏️ Drawing Characters:
   • Use the drawing window to hand-draw characters
   • Draw large, clear characters for best results
   • Use the entire canvas area

🧠 Models:
   • ResNet50: Pretrained model with high accuracy
   • Deep CNN: Custom model optimized for characters
   • Both support 62 classes: 0-9, A-Z, a-z

📊 Results:
   • View top prediction with confidence score
   • Check top-5 alternatives in the table
   • Green = High confidence, Orange = Medium, Red = Low

💾 Saving:
   • Save individual results as TXT or JSON
   • Export complete prediction history
   • Save drawn characters as PNG images

⌨️ Keyboard Shortcuts:
   • Ctrl+O: Load Image
   • Ctrl+S: Save Results
   • F5: Refresh Models
   • Ctrl+Q: Quit"""

        # Create custom dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Instructions")
        dialog.geometry("600x700")
        dialog.configure(bg="#f8f9fa")
        dialog.resizable(False, False)

        # Make modal
        dialog.transient(self.root)
        dialog.grab_set()

        # Add text with scrollbar
        text_frame = tk.Frame(dialog, bg="#f8f9fa")
        text_frame.pack(fill="both", expand=True, padx=20, pady=20)

        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=('Segoe UI', 10),
            bg="#ffffff",
            fg="#2c3e50",
            padx=15,
            pady=15
        )

        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        text_widget.insert(tk.END, instructions)
        text_widget.config(state="disabled")

        # Close button
        close_btn = ttk.Button(dialog, text="Close", command=dialog.destroy)
        close_btn.pack(pady=10)

    def show_shortcuts(self):
        """Show keyboard shortcuts"""

        shortcuts = """⌨️ Keyboard Shortcuts:

File Operations:
   • Ctrl+O    Load Image
   • Ctrl+S    Save Results
   • Ctrl+Q    Quit Application

View Operations:
   • F5        Refresh Models

Window Operations:
   • Escape    Close dialogs
   • Enter     Confirm actions"""

        messagebox.showinfo("Keyboard Shortcuts", shortcuts)

    def show_about(self):
        """Show about dialog with enhanced information"""

        about_text = """🔤 Handwritten Character Recognition

Version: 2.0
Build: Professional Edition

🧠 AI Models:
   • ResNet50 (Pretrained Transfer Learning)
   • Deep CNN (Custom Architecture)

🎯 Capabilities:
   • 62 Character Classes (0-9, A-Z, a-z)
   • Real-time Prediction
   • Interactive Drawing
   • Professional GUI Interface

🛠️ Technology Stack:
   • TensorFlow/Keras for Deep Learning
   • OpenCV for Image Processing
   • Tkinter for GUI Interface
   • PIL for Image Handling

📊 Performance:
   • Target Accuracy: >90%
   • Real-time Inference: <50ms
   • Support for Multiple Model Types

🎨 Features:
   • Load and test images
   • Draw characters manually
   • View top-5 predictions with confidence
   • Save results and prediction history
   • Professional model comparison

Built for research and educational purposes.
Optimized for handwritten character recognition tasks."""

        messagebox.showinfo("About", about_text)

def run_gui():
    """Main function to launch the GUI application"""

    # Configure high DPI awareness
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    root = tk.Tk()
    app = CharacterRecognitionGUI(root)

    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    # Set minimum size
    root.minsize(1200, 800)

    print("🚀 Launching Character Recognition GUI...")
    print("✅ GUI loaded successfully!")

    root.mainloop()

if __name__ == "__main__":
    run_gui()
