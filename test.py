import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import torch
import json
import os
import tkinter as tk
import time
import numpy as np


# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s' , force_reload=True)

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection Selector")
        self.root.geometry("400x300")
        self.root.configure(bg="#1a1a2e")

        # Title Label
        self.title_label = ctk.CTkLabel(self.root, text="Select Detection Mode", font=("Arial", 20), fg_color="#162447", text_color="white")
        self.title_label.pack(pady=30)

        # Buttons for image and video detection
        self.image_button = ctk.CTkButton(self.root, text="Detect Object in Image", command=self.open_image_detection, width=200)
        self.image_button.pack(pady=10)

        self.video_button = ctk.CTkButton(self.root, text="Detect Object in Video", command=self.open_video_detection, width=200)
        self.video_button.pack(pady=10)

    def open_image_detection(self):
        self.new_window = tk.Toplevel(self.root)
        self.app = ImageLabelApp(self.new_window)

    def open_video_detection(self):
        self.new_window = tk.Toplevel(self.root)
        self.app = VideoLabelApp(self.new_window)
        
        

class ImageLabelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Label Master - Image Detection")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2B2B2B")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Title Label
        self.title_label = ctk.CTkLabel(self.root, text="Label Master", font=("Helvetica", 28, "bold"))
        self.title_label.pack(pady=20)

        # Navbar
        self.navbar_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.navbar_frame.pack(fill="x", padx=20, pady=10)

        # Buttons in Navbar
        self.load_button = ctk.CTkButton(self.navbar_frame, text="Load Image", command=self.load_image, width=100)
        self.load_button.pack(side="left", padx=10)

        self.save_button = ctk.CTkButton(self.navbar_frame, text="Save Labels", command=self.save_labels, width=100)
        self.save_button.pack(side="left", padx=10)

        self.search_button = ctk.CTkButton(self.navbar_frame, text="Search Object", command=self.on_search_button_click, width=100)
        self.search_button.pack(side="left", padx=10)

        # Image Display Frame
        self.image_frame = ctk.CTkFrame(self.root, width=640, height=480, corner_radius=10)
        self.image_frame.pack(pady=20)

        # Image Panel
        self.image_panel = ctk.CTkLabel(self.image_frame)
        self.image_panel.pack()

        # Image and labels variables
        self.cv_img = None
        self.labels = []
        self.image_path = None

        # Drawing Bounding Box
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.drawing_box = False

        self.image_panel.bind("<Button-1>", self.on_mouse_down)
        self.image_panel.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_panel.bind("<ButtonRelease-1>", self.on_mouse_up)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", ".jpg;.jpeg;.png;.bmp")])
        if not file_path:
            return
        self.image_path = file_path
        self.cv_img = cv2.imread(file_path)
        self.cv_img = cv2.resize(self.cv_img, (640, 480))
        self.display_image()

    def detect_objects(self, img, x1, y1, x2, y2):
        results = model(img)
        self.labels = []  # Clear previous labels
        for result in results.xyxy[0]:
            x, y, w, h, conf, cls = result
            label = model.names[int(cls)]
            self.labels.append({
                'label': label,
                'coordinates': (int(x), int(y), int(x + w), int(y + h))
            })

        if not self.labels:
            messagebox.showwarning("No Objects", "No objects detected in the image area.")
        else:
            print(f"Labels detected: {self.labels}")

        self.edit_labels()

    def crop_image(self, img, x1, y1, x2, y2):
        return img[y1:y2, x1:x2]

    def display_image(self):
        if self.cv_img is not None:
            img_rgb = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # Maintain aspect ratio
            img_width, img_height = pil_img.size
            max_width, max_height = 640, 480

            aspect_ratio = min(max_width / img_width, max_height / img_height)
            new_width = int(img_width * aspect_ratio)
            new_height = int(img_height * aspect_ratio)

            # Resize the image with aspect ratio
            pil_img_resized = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Create a new blank image with black background and paste the resized image
            new_img = Image.new("RGB", (max_width, max_height), (0, 0, 0))
            new_img.paste(pil_img_resized, ((max_width - new_width) // 2, (max_height - new_height) // 2))

            tk_img = ImageTk.PhotoImage(new_img)

            self.image_panel.configure(image=tk_img)
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
        self.image_panel.configure(image=tk_img)
        self.image_panel.image = tk_img

    def edit_labels(self):
        if not self.labels:  # Check if there are no labels
            messagebox.showinfo("No Objects Detected", "No objects were detected to label.")
            return  # Exit the function if no objects are detected

        current_labels = ', '.join([obj['label'] for obj in self.labels])
        new_labels = ctk.CTkInputDialog(text=f"Detected labels: {current_labels}\nEnter new labels (comma-separated):").get_input()

        if new_labels:
            new_label_list = [label.strip() for label in new_labels.split(',')]
            for i, obj in enumerate(self.labels):
                if i < len(new_label_list):
                    obj['label'] = new_label_list[i]
    
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
    
class VideoLabelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Label Master - Video Detection")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2B2B2B")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Title Label
        self.title_label = ctk.CTkLabel(self.root, text="Label Master - Video", font=("Helvetica", 28, "bold"))
        self.title_label.pack(pady=20)

        # Navbar
        self.navbar_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.navbar_frame.pack(fill="x", padx=20, pady=10)

        # Buttons in Navbar
        self.load_button = ctk.CTkButton(self.navbar_frame, text="Load Video", command=self.load_video, width=100)
        self.load_button.pack(side="left", padx=10)

        self.play_button = ctk.CTkButton(self.navbar_frame, text="Play", command=self.play_video, width=100)
        self.play_button.pack(side="left", padx=10)

        self.pause_button = ctk.CTkButton(self.navbar_frame, text="Pause", command=self.pause_video, width=100)
        self.pause_button.pack(side="left", padx=10)

        self.save_button = ctk.CTkButton(self.navbar_frame, text="Save Labels", command=self.save_labels, width=100)
        self.save_button.pack(side="left", padx=10)

        self.search_button = ctk.CTkButton(self.navbar_frame, text="Search Object", command=self.on_search_button_click, width=100)
        self.search_button.pack(side="left", padx=10)

        # Video Display Frame
        self.video_frame = ctk.CTkFrame(self.root, width=640, height=480, corner_radius=10)
        self.video_frame.pack(pady=20)

        # Video Panel
        self.video_panel = ctk.CTkLabel(self.video_frame)
        self.video_panel.pack()

        # Video and labels variables
        self.cap = None
        self.frame = None
        self.video_path = None
        self.playing = False
        self.labels = []

        # Drawing Bounding Box
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.drawing_box = False

        self.video_panel.bind("<Button-1>", self.on_mouse_down)
        self.video_panel.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_panel.bind("<ButtonRelease-1>", self.on_mouse_up)

    def load_video(self):
        file_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
        if not file_path:
            return
        self.video_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        self.playing = False
        self.display_frame()

    def play_video(self):
        self.playing = True
        self.update_frame()

    def pause_video(self):
        self.playing = False

    def update_frame(self):
        if self.playing and self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if ret:
                self.display_frame()
                self.root.after(30, self.update_frame)  # Update frame every 30ms (about 33 FPS)
            else:
                self.playing = False

    def display_frame(self):
        if self.frame is not None:
            # Get frame dimensions
            frame_height, frame_width = self.frame.shape[:2]

            # Set max dimensions (you can adjust this as per your UI requirements)
            max_width = 640
            max_height = 480

            # Maintain aspect ratio
            aspect_ratio = min(max_width / frame_width, max_height / frame_height)
            new_width = int(frame_width * aspect_ratio)
            new_height = int(frame_height * aspect_ratio)

            # Resize the frame with aspect ratio
            frame_resized = cv2.resize(self.frame, (new_width, new_height))

            # Create a new blank frame with black background and paste the resized frame
            blank_frame = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        
            # Calculate offsets for centering the resized frame
            y_offset = (max_height - new_height) // 2
            x_offset = (max_width - new_width) // 2
            blank_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = frame_resized

            # Convert BGR to RGB and display the frame
            img_rgb = cv2.cvtColor(blank_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tk_img = ImageTk.PhotoImage(pil_img)

            self.video_panel.configure(image=tk_img)
            self.video_panel.image = tk_img




    def on_mouse_down(self, event):
        if self.frame is not None:
            self.start_x = event.x
            self.start_y = event.y
            self.drawing_box = True

    def on_mouse_drag(self, event):
        if self.drawing_box and self.frame is not None:
            self.end_x = event.x
            self.end_y = event.y
            temp_frame = self.frame.copy()
            cv2.rectangle(temp_frame, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 0, 0), 2)
            self.display_frame_with_rect(temp_frame)

    def on_mouse_up(self, event):
        if self.frame is not None:
            self.drawing_box = False
            self.end_x = event.x
            self.end_y = event.y
            cropped_frame = self.crop_frame(self.frame, self.start_x, self.start_y, self.end_x, self.end_y)
            self.detect_objects(cropped_frame, self.start_x, self.start_y, self.end_x, self.end_y)

    def crop_frame(self, frame, x1, y1, x2, y2):
        return frame[y1:y2, x1:x2]

    def display_frame_with_rect(self, frame_with_rect):
        frame_resized = cv2.resize(frame_with_rect, (640, 480))
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.video_panel.configure(image=tk_img)
        self.video_panel.image = tk_img

    def detect_objects(self, frame, x1, y1, x2, y2):
        results = model(frame)
        self.labels = []  # Clear previous labels
        for result in results.xyxy[0]:
            x, y, w, h, conf, cls = result
            label = model.names[int(cls)]
            self.labels.append({
                'label': label,
                'coordinates': (int(x), int(y), int(x + w), int(y + h)),
                'timestamp': self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Timestamp in seconds
            })

        if not self.labels:
            messagebox.showwarning("No Objects", "No objects detected in the video frame.")
        else:
            print(f"Labels detected: {self.labels}")
        self.edit_labels()

    def edit_labels(self):
        if not self.labels:  # Check if there are no labels
            messagebox.showinfo("No Objects Detected", "No objects were detected to label.")
            return  # Exit the function if no objects are detected

        current_labels = ', '.join([obj['label'] for obj in self.labels])
        new_labels = ctk.CTkInputDialog(text=f"Detected labels: {current_labels}\nEnter new labels (comma-separated):").get_input()

        if new_labels:
            new_label_list = [label.strip() for label in new_labels.split(',')]
            for i, obj in enumerate(self.labels):
                if i < len(new_label_list):
                    obj['label'] = new_label_list[i]
    
        self.save_labels()

    def save_labels(self):
        if self.labels and self.video_path:
            data = {
                'video_path': self.video_path,
                'objects': self.labels
            }
            json_file_path = 'video_labels.json'
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

            messagebox.showinfo("Saved", "Labels and coordinates saved to video_labels.json")

    def on_search_button_click(self):
        messagebox.showinfo("Search Mode", "Pause the video and draw a box over the object you want to search.")

if __name__ == "__main__":
    root = tk.Tk()
    main_app = MainApp(root)
    root.mainloop()


