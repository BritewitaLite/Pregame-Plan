import cv2
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from camera_movement_est.camera_mov_est import CameraMovementEstimator
from field_keypoint_detector import FieldKeypointDetector, FieldKeypointAnnotator

def main():
    # read video
    video_frames = read_video('input_videos/test2.mp4')

    #initialize tracker
    tracker = Tracker('models/player_detector.pt')

    #initialize field keypoint detector
    field_keypoint_detector = FieldKeypointDetector('models/field_keypoints_detector_v1.pt')
    field_keypoint_detector1 = FieldKeypointAnnotator()


    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs2.pk1')
    field_keypoints = field_keypoint_detector.get_field_keypoints(video_frames, read_from_stub=True, stub_path='stubs/field_keypoints_stubs2.pk1')

    #camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stubs2.pk1')

    '''# save cropped image of a player, don't need this now but can be used later trackig by jersey color or other things (not fully complete here just wanted it as a reminder)
    for track_id, line in tracks['Line'][0].items():
        bbox = line['bbox']
        frame = video_frames[0]
        #cropped bbox from frame
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        #save cropped image
        cv2.imwrite(f'output_videos/cropped_img.jpg', cropped_image)
        break'''

    #draw object tracked
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    #draw field keypoints
    output_video_frames = field_keypoint_detector1.draw(output_video_frames, field_keypoints)

    #draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # save video
    save_video(output_video_frames, 'output_videos/output_video1.avi')
    

if __name__ == '__main__':
    main()