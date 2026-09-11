import cv2
import os
import numpy as np


VIDEO_DIR = 'input_videos'
OUT_DIR = 'runs/predict'
OUT_PATH = os.path.join(OUT_DIR, 'verify_colors.png')

# Try to read first frame of first video
video_files = [os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
frame = None
if video_files:
    cap = cv2.VideoCapture(video_files[0])
    ret, frame = cap.read()
    cap.release()

if frame is None:
    # Create a dummy green field image
    frame = 255 * np.ones((720, 1280, 3), dtype='uint8')
    frame[:, :, 1] = 200  # greenish

# Create synthetic tracks: one player per class
classes = ["C","DB","DLine","FB","G","LB","PreSnap","QB","RB","Ref","S","T","TE","WR"]
tracks = {}
for cls in classes:
    # single frame list with one bbox per class
    tracks[cls] = [{0: {"bbox": [50 + 80 * classes.index(cls), 200, 120 + 80 * classes.index(cls), 320]}}]

def draw_ellipse(frame, bbox, color, track_id=None):
    y2 = int(bbox[3])
    x_center = int((bbox[0] + bbox[2]) / 2)
    width = int(bbox[2] - bbox[0])
    cv2.ellipse(frame, center=(x_center, y2), axes=(int(width), int(0.35*width)), angle=0, startAngle=45, endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4)
    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x_center - rectangle_width//2
    x2_rect = x_center + rectangle_width//2
    y1_rect = (y2 - rectangle_height//2) + 15
    if track_id is not None:
        cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y1_rect + rectangle_height)), color, cv2.FILLED)
        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10
        cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return frame

# Role mapping provided by user
ROLE_MAP = {
    "C": "offense",
    "DB": "defense",
    "DLine": "defense",
    "FB": "offense",
    "G": "offense",
    "LB": "defense",
    "PreSnap": "neutral",
    "QB": "offense",
    "RB": "offense",
    "Ref": "neutral",
    "S": "defense",
    "T": "offense",
    "TE": "offense",
    "WR": "offense",
}

COLOR_BY_ROLE = {
    "offense": (50, 100, 220),
    "defense": (220, 50, 50),
    "neutral": (255, 255, 255),
}

# Draw using synthetic tracks
annotated = frame.copy()
for class_name, frames_list in tracks.items():
    obj_dict = frames_list[0]
    role = ROLE_MAP.get(class_name, "neutral")
    color = COLOR_BY_ROLE.get(role, (255,255,255))
    for track_id, obj in obj_dict.items():
        annotated = draw_ellipse(annotated, obj["bbox"], color, track_id)

os.makedirs(OUT_DIR, exist_ok=True)
cv2.imwrite(OUT_PATH, annotated)
print('Wrote', OUT_PATH)
