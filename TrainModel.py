
from ultralytics import YOLO
import torch

# print(torch.cuda.is_available()) gpu is now being found in vscode not pycharm

def main():
    model = YOLO("yolo11l-pose.pt")

# Train the model using the 'coco8.yaml' dataset for 250 epochs
    results = model.train(data=r"C:\Users\bryce\Documents\GitHub\Pregame-Plan\data.yaml", epochs=200, imgsz=1280, device='0')
    
    return results


# Create a new YOLO model from scratch
if __name__ ==  '__main__':
    main()