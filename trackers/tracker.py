from ultralytics import YOLO
import supervision as sv
import cv2
import pickle
import os
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width
from utils import save_stub, read_stub

class Tracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        #creating batch to limimt memory usage
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf= 0.1, device='0')
            #adding the detections to the list
            detections += detections_batch
        return detections

    '''this function will be returned to later when we create a dictionary to store intial calssifications over first 10 frames and then keep that 
    classification for the rest of the frames'''
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
    
        tracks = read_stub(read_from_stub, stub_path)

        #keeps us from running this again if the stub_path exists
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)

        #going to keep track of the number of presnap confs above .9
        presnap_conf_count = 0
        presnap_locked = False
        locked_classes = {} #tracking the classes that are locked in for each player


        #initializing empty list for each class 'C', 'DB', 'DLine', 'FB', 'G', 'LB', 'PostSnap', 'PreSnap', 'QB', 'RB', 'Ref', 'S', 'T', 'TE', 'WR'
        tracks={
            "C": [],
            "DB": [],
            "DLine": [],
            "FB": [],
            "G": [],
            "LB": [],
            "PreSnap": [],
            "QB": [],
            "RB": [],
            "Ref": [],
            "S": [],
            "T": [],
            "TE": [],
            "WR": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            # inversing the class names to read Skill:1 instead of 1:Skill
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            #we shouldn't need this as we are not eliminating a class from appearing but will be doing something similar
            '''for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]'''

            # adds tracker object to the detection
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
        
            # adding the detection to the tracks dictionary
            # going to contain the track_id and bbox for each frame for each individual player who has that class value
            tracks["C"].append({})
            tracks["DB"].append({})
            tracks["DLine"].append({})
            tracks["FB"].append({})
            tracks["G"].append({})
            tracks["LB"].append({})
            tracks["PreSnap"].append({})
            tracks["QB"].append({})
            tracks["RB"].append({})
            tracks["Ref"].append({})
            tracks["S"].append({})
            tracks["T"].append({})
            tracks["TE"].append({})
            tracks["WR"].append({})

            # doing it separately to keep track of the center and qb because there is only 1 of each of these players
            for i in range(len(detection_supervision)):
                bbox = detection_supervision.xyxy[i].tolist()
                class_id = int(detection_supervision.class_id[i])
                confidence = float(detection_supervision.confidence[i])

                if class_id == cls_names_inv['PreSnap'] and confidence > 0.92:
                    # we are only adding the presnap bbox if the confidence is above a certain threshold
                    # this is to avoid false positives
                    tracks["PreSnap"][frame_num][0] = {"bbox":bbox}
                    presnap_conf_count += 1
        
            # if we have 40 presnap confs above .9 then we lock in the class for that player
            if not presnap_locked and presnap_conf_count >= 30:
                print(f"[INFO] PreSnap confidence exceeded threshold at frame {frame_num}. Locking class assignments.")
                presnap_locked = True

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                # it is 3 because the class_id is the third value given in the output
                class_id = frame_detection[3]
                # it is 4 because the track_id is the fourth value given in the output
                track_id = frame_detection[4]

                if presnap_locked:
                    if len(locked_classes) >= 25 and track_id not in locked_classes:
                        print(f"[WARNING] Skipping new track_id {track_id} due to 25-track limit.")
                        continue
                    if track_id in locked_classes:
                        class_id = locked_classes[track_id]
                    elif len(locked_classes) < 25:
                        locked_classes[track_id] = class_id
                        print(f"[INFO] Locked class {class_id} for track_id {track_id} at frame {frame_num}.")
                    else:
                        print(f"[WARNING] Skipped locking for track_id {track_id} due to 25-class limit.")

                # Add detection to the appropriate class track
                for class_name, class_index in cls_names_inv.items():
                    if class_id == class_index and class_name in tracks:
                        tracks[class_name][frame_num][track_id] = {"bbox": bbox}
                        break

        save_stub(stub_path, tracks)


        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id = None):
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, center = (x_center,y2), axes = (int(width), int(0.35*width)), angle=0, startAngle=45, endAngle=235 , color = color, thickness = 2, lineType = cv2.LINE_4)

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Role mapping provided by user: offense/defense/neutral
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

            # BGR colors: offense = blue, defense = red, neutral = white
            COLOR_BY_ROLE = {
                "offense": (50, 100, 220),   # blue-ish BGR
                "defense": (220, 50, 50),    # red BGR
                "neutral": (255, 255, 255),  # white
            }

            # Iterate all classes generically and draw with role color
            for class_name, frames_list in tracks.items():
                if frame_num >= len(frames_list):
                    continue
                obj_dict = frames_list[frame_num]
                role = ROLE_MAP.get(class_name, "neutral")
                color = COLOR_BY_ROLE.get(role, (255, 255, 255))
                for track_id, obj in obj_dict.items():
                    frame = self.draw_ellipse(frame, obj["bbox"], color, track_id)

            output_video_frames.append(frame)

        return output_video_frames