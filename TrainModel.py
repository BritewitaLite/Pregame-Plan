
from ultralytics import YOLO
import torch

# print(torch.cuda.is_available()) gpu is now being found in vscode not pycharm

def main():
    model = YOLO('yolov8l.yaml')

# Train the model using the 'coco8.yaml' dataset for 250 epochs
    results = model.train(data=r"C:\Scouting_Report_Clone\Pregame-Plan\data.yaml", epochs=250, imgsz=640, device='0')
    
    return results


# Create a new YOLO model from scratch
if __name__ ==  '__main__':
    main()