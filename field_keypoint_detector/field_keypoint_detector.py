from ultralytics import YOLO
import supervision as sv
import numpy as np
import sys
import os
import pickle
sys.path.append('../')
from utils import read_stub,save_stub

class FieldKeypointDetector:
        
    def __init__(self, model_path):
        self.model=YOLO(model_path)

    def get_field_keypoints(self,frames,read_from_stub=False,stub_path=None):

        field_keypoints = read_stub(read_from_stub,stub_path)

        if field_keypoints is not None:
            if len(field_keypoints) == len(frames):
                return field_keypoints

        batch_size = 20
        field_keypoints=[]

        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size],conf=0.65)
            for detection in detections_batch:
                field_keypoints.append(detection.keypoints)

        save_stub(stub_path, field_keypoints)

        return field_keypoints

class FieldKeypointAnnotator:
    def __init__(self):
        self.keypoints_color = "#ff2c2c"

    def draw(self, frames, field_keypoints):
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoints_color),
            radius=6
        )
        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoints_color),
            text_color=sv.Color.WHITE,
            text_scale =0.5,
            text_thickness=1
        )

        output_frames = []
        for index,frame in enumerate(frames):
            annotated_frame = frame.copy()

            keypoints = field_keypoints[index]
            annotate_frame = vertex_annotator.annotate(scene=annotated_frame, key_points=keypoints)

            keypoints_numpy = keypoints.cpu().numpy()
            annotate_frame = vertex_label_annotator.annotate(scene=annotated_frame, key_points=keypoints_numpy)
            output_frames.append(annotated_frame)

        return output_frames
