from ultralytics import YOLO
import supervision as sv
import cv2
import pickle
import os
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

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
    
        #keeps us from running this again if the stub_path exists
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        #initializing empty list for each class
        tracks={
            "Line": [],
            "Ref": [],
            "center": [],
            "qb": [],
            "skill": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            # inversing the class names to read Skill:1 instead of 1:Skill
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            #we shouldn't need this as we are not eliminating a class from appearing but will be doing something similar
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # adds tracker object to the detection
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
        
            # adding the detection to the tracks dictionary
            #going to contain the track_id and bbox for each frame for each individual player who has that class value
            tracks["Line"].append({})
            tracks["Ref"].append({})
            tracks["center"].append({})
            tracks["qb"].append({})
            tracks["skill"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                # it is 3 because the class_id is the third value given in the output
                class_id = frame_detection[3]
                # it is 4 because the class_id is the fourth value given in the output
                track_id = frame_detection[4]
                
                #frame_num will be the list index for the Line list, adding bounding box into a dictionary for that player
                if class_id == cls_names_inv['Line']:
                    tracks["Line"][frame_num][track_id] = {"bbox":bbox}

                if class_id == cls_names_inv['Ref']:
                    tracks["Ref"][frame_num][track_id] = {"bbox":bbox}

                if class_id == cls_names_inv['skill']:
                    tracks["skill"][frame_num][track_id] = {"bbox":bbox}
                

            # doing it separately to keep track of the center and qb because there is only 1 of each of these players
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == cls_names_inv['center']:
                    tracks["center"][frame_num][00] = {"bbox":bbox}

                if class_id == cls_names_inv['qb']:
                    tracks["qb"][frame_num][99] = {"bbox":bbox}
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)


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

            line_dict = tracks["Line"][frame_num]
            ref_dict = tracks["Ref"][frame_num]
            center_dict = tracks["center"][frame_num]
            qb_dict = tracks["qb"][frame_num]
            skill_dict = tracks["skill"][frame_num]

            # draw players
            for track_id, line in line_dict.items():
                frame = self.draw_ellipse(frame, line["bbox"], (0,255,255), track_id)

            for track_id, ref in ref_dict.items():
                frame = self.draw_ellipse(frame, ref["bbox"], (0,0,255), track_id)

            for track_id, center in center_dict.items():
                frame = self.draw_ellipse(frame, center["bbox"], (0,255,0), track_id)

            for track_id, qb in qb_dict.items():
                frame = self.draw_ellipse(frame, qb["bbox"], (255,255,0), track_id)

            for track_id, skill in skill_dict.items():
                frame = self.draw_ellipse(frame, skill["bbox"], (255,0,255), track_id)

            output_video_frames.append(frame)

        return output_video_frames