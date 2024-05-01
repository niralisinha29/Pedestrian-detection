
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

class PedestrianDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pedestrian Detection System")
        
        # Create HOG descriptor object
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # GUI Layout
        self.frame = tk.Frame(self.root)
        self.frame.pack(pady=70)

        self.btn_load_video = tk.Button(self.frame, text="Load Video", command=self.load_video)
        self.btn_load_video.pack(side=tk.LEFT, padx=20)

        self.label = tk.Label(self.frame, text="No video loaded")
        self.label.pack(side=tk.LEFT, padx=20)

    def load_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.label.configure(text="Video Loaded: " + file_path)
            self.process_video(file_path)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Detect pedestrians in the frame
                frame = self.detect_pedestrians(frame)
                # Display the frame
                cv2.imshow('Pedestrian Detection', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    def detect_pedestrians(self, frame):
        # Resize frame to improve detection speed and accuracy
        frame = cv2.resize(frame, (640, 480))
        # Detect pedestrians in the frame
        rects, weights = self.hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        
        # Draw rectangles around detected pedestrians
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame

root = tk.Tk()
app = PedestrianDetectionApp(root)
root.mainloop()


