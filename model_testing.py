import os
import cv2
import torch
import numpy as np

from ultralytics import YOLO


# Pre-defined variables
path = "/Scouting_Report"
video_path = os.path.join(path, "input_videos", "Test_vid.mp4")
model_path = os.path.join(path, "runs", "detect", "train", "weights","best.pt")
confidence_thresh = .7

# Set device to gpu
device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0)

# Model path
model = YOLO(model_path)

class Tracker:
    # Constructor
    def __init__(self, model, video_path, confidence_thresh):
        super().__init__()
        self.model = model
        self.video_path = video_path
        self.confidence_thresh = confidence_thresh

    # Process the video results
    def player_tracking(self):
        results = model.predict(self.video_path, save=True, device='0')
        return results



t1 = Tracker(model, video_path, confidence_thresh)

print(t1.player_tracking()[0])
print('====================')

for box in t1.player_tracking()[0].boxes:
    print(box)

#Cleanup and close window
#cv2.destroyAllWindows()
