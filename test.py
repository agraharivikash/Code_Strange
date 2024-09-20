import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import simpledialog
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
        self.root.title("Object Recognition and Labeling Tool")

        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill=tk.X)

        # Buttons to load images and save labels
        self.load_button = ttk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(button_frame, text="Save Labels", command=self.save_labels)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Frame for image display
        self.image_panel = tk.Label(self.root)
        self.image_panel.pack(pady=1)

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

        # Run object detection and store results for later use
        self.results = model(self.cv_img)  # Store results without drawing boxes

    def display_image(self):
        img_rgb = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(pil_img)

        self.image_panel.config(image=tk_img)
        self.image_panel.image = tk_img

    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.drawing_box = True

    def on_mouse_drag(self, event):
        if self.drawing_box:
            self.end_x = event.x
            self.end_y = event.y

            # Draw rectangle on the image
            temp_img = self.cv_img.copy()
            cv2.rectangle(temp_img, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 0, 0), 2)
            self.display_image_with_rect(temp_img)

    def on_mouse_up(self, event):
        self.drawing_box = False
        self.end_x = event.x
        self.end_y = event.y

        # Get objects inside the drawn box
        self.get_objects_in_box()

    def display_image_with_rect(self, img_with_rect):
        img_rgb = cv2.cvtColor(img_with_rect, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(pil_img)

        self.image_panel.config(image=tk_img)
        self.image_panel.image = tk_img

    def get_objects_in_box(self):
        if not self.results:
            return

        user_box = (self.start_x, self.start_y, self.end_x, self.end_y)
        objects_in_box = []

        for result in self.results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result
            obj_box = (x1, y1, x2, y2)

            # Check if the object is inside the user-drawn box
            if self.is_box_inside(user_box, obj_box):
                label = model.names[int(cls)]
                objects_in_box.append((label, (int(x1), int(y1), int(x2), int(y2))))  # Store label and coordinates

        if objects_in_box:
            # Create a string to display found objects
            found_objects = ', '.join([f"{label} (Coords: {coords})" for label, coords in objects_in_box])
            messagebox.showinfo("Objects Found", f"Objects inside the box: {found_objects}")
        else:
            messagebox.showinfo("No Objects", "No objects found inside the box.")

    def is_box_inside(self, user_box, obj_box):
        ux1, uy1, ux2, uy2 = user_box
        ox1, oy1, ox2, oy2 = obj_box

        # Check if the object box is inside the user-drawn box
        return (ux1 <= ox1 and uy1 <= oy1 and ux2 >= ox2 and uy2 >= oy2)

    def save_labels(self):
        if self.image_path:
            # Create a dictionary for saving
            data = {
                'image_path': self.image_path,
                'detected_objects': []
            }

            # Collect labels and coordinates for detected objects
            for result in self.results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = result
                label = model.names[int(cls)]
                data['detected_objects'].append({
                    'label': label,
                    'coordinates': (int(x1), int(y1), int(x2), int(y2))
                })

            json_file_path = 'image_labels.json'

            # Check if the file already exists and load existing data
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as json_file:
                    existing_data = json.load(json_file)
            else:
                existing_data = []

            existing_data.append(data)  # Add new data

            # Save all data back to JSON file
            with open(json_file_path, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)

            messagebox.showinfo("Saved", "Labels saved to image_labels.json")
            print(f"Labels saved for {self.image_path}")
        else:
            messagebox.showerror("Error", "No image loaded.")
            print("No image to save labels for.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelApp(root)
    root.mainloop()
