import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import json
import os

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

class ImageLabelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Label Master")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2B2B2B")
        ctk.set_appearance_mode("dark")  # Set dark mode
        ctk.set_default_color_theme("blue")  # Set color theme

        # Title Label
        self.title_label = ctk.CTkLabel(self.root, text="Label Master", font=("Helvetica", 28, "bold"))
        self.title_label.pack(pady=20)

        # Navbar Frame
        self.navbar_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.navbar_frame.pack(fill="x", padx=20, pady=10)

        # Buttons in Navbar
        self.load_button = ctk.CTkButton(self.navbar_frame, text="Load Image", command=self.load_image, width=100)
        self.load_button.pack(side="left", padx=10)

        self.save_button = ctk.CTkButton(self.navbar_frame, text="Save Labels", command=self.save_labels, width=100)
        self.save_button.pack(side="left", padx=10)

        self.search_button = ctk.CTkButton(self.navbar_frame, text="Search Object", command=self.on_search_button_click, width=100)
        self.search_button.pack(side="left", padx=10)

        # Frame for image display
        self.image_frame = ctk.CTkFrame(self.root, width=640, height=480, corner_radius=10)
        self.image_frame.pack(pady=20)

        # Image Label (for showing loaded image)
        self.image_panel = ctk.CTkLabel(self.image_frame)
        self.image_panel.pack()

        # Image-related variables
        self.cv_img = None
        self.labels = []
        self.image_path = None
        self.results = None

        # Variables for drawing bounding box
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.drawing_box = False

        self.image_panel.bind("<Button-1>", self.on_mouse_down)
        self.image_panel.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_panel.bind("<ButtonRelease-1>", self.on_mouse_up)

    def load_image(self):
        # File dialog to select an image
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", ".jpg;.jpeg;.png;.bmp")])
        if not file_path:
            return

        self.image_path = file_path
        self.cv_img = cv2.imread(file_path)
        if self.cv_img is None:
            messagebox.showerror("Error", "Could not load image.")
            return

        self.cv_img = cv2.resize(self.cv_img, (640, 480))
        self.display_image()

    def detect_objects(self, img, x1, y1, x2, y2):
        results = model(img)

        self.labels = []  # Clear previous labels
        self.results = results

        for result in results.xyxy[0]:
            x, y, w, h, conf, cls = result
            label = model.names[int(cls)]
            self.labels.append({
                'label': label,
                'coordinates': (int(x), int(y), int(x+w), int(y+h))
            })

        if not self.labels:
            messagebox.showwarning("No Objects", "No objects detected in the image area.")
        else:
            print(f"Labels detected: {self.labels}")

        self.edit_labels(self.labels)

    def crop_image(self, img, x1, y1, x2, y2):
        return img[y1:y2, x1:x2]

    def display_image(self):
        img_rgb = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(pil_img)

        self.image_panel.configure(image=tk_img)  # Use configure instead of config
        self.image_panel.image = tk_img

    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.drawing_box = True

    def on_mouse_drag(self, event):
        if self.drawing_box:
            self.end_x = event.x
            self.end_y = event.y

            temp_img = self.cv_img.copy()
            cv2.rectangle(temp_img, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 0, 0), 2)
            self.display_image_with_rect(temp_img)

    def on_mouse_up(self, event):
        self.drawing_box = False
        self.end_x = event.x
        self.end_y = event.y

        cropped_img = self.crop_image(self.cv_img, self.start_x, self.start_y, self.end_x, self.end_y)
        self.detect_objects(cropped_img, self.start_x, self.start_y, self.end_x, self.end_y)

    def display_image_with_rect(self, img_with_rect):
        img_rgb = cv2.cvtColor(img_with_rect, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(pil_img)

        self.image_panel.configure(image=tk_img)  # Use configure instead of config
        self.image_panel.image = tk_img

    def edit_labels(self, detected_objects):
        current_labels = ', '.join([obj['label'] for obj in detected_objects])
        new_labels = ctk.CTkInputDialog(text=f"Detected labels: {current_labels}\nEnter new labels (comma-separated):").get_input()
        if new_labels:
            new_label_list = [label.strip() for label in new_labels.split(',')]
            for i, obj in enumerate(detected_objects):
                if i < len(new_label_list):
                    obj['label'] = new_label_list[i]

        self.labels = detected_objects
        self.save_labels()

    def save_labels(self):
        if self.labels and self.image_path:
            data = {
                'image_path': self.image_path,
                'objects': self.labels
            }
            json_file_path = 'image_labels.json'

            if os.path.exists(json_file_path):
                if os.stat(json_file_path).st_size > 0:
                    with open(json_file_path, 'r') as json_file:
                        try:
                            existing_data = json.load(json_file)
                        except json.JSONDecodeError:
                            existing_data = []
                else:
                    existing_data = []
            else:
                existing_data = []

            existing_data.append(data)
            with open(json_file_path, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)

            messagebox.showinfo("Saved", "Labels and coordinates saved to image_labels.json")

    def on_search_button_click(self):
        messagebox.showinfo("Search Mode", "Draw a box over the object you want to search.")

if __name__ == "__main__":  
    root = ctk.CTk()
    app = ImageLabelApp(root)
    root.mainloop()
