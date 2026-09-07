import cv2
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from camera_movement_est.camera_mov_est import CameraMovementEstimator

# ── Drop-in replacement for FieldKeypointDetector ──────────────────────────
from .hough_field_homography import build_hough_mapper, HoughFieldAnnotator
# ───────────────────────────────────────────────────────────────────────────

def main():
    # Read video
    video_frames = read_video('input_videos/test2.mp4')

    # Initialize player tracker (unchanged)
    tracker = Tracker('models/player_detector.pt')
    tracks  = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path='stubs/track_stubs2.pk1'
    )

    # ── Hough-based field homography (no ML model needed) ──────────────────
    mapper, homographies = build_hough_mapper(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/hough_homography.pkl',
    )
    annotator = HoughFieldAnnotator()
    # ───────────────────────────────────────────────────────────────────────

    # Camera movement estimator (unchanged)
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stubs2.pk1'
    )

    # Draw tracked objects
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Draw Hough field grid + homography status
    output_video_frames = annotator.draw(output_video_frames, homographies, draw_grid=True)

    # Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )

    # ── Optional: map a tracked player pixel → field yards ─────────────────
    # Example: take the first player detection in frame 0 and map its feet to field coords
    frame0_players = tracks.get('players', [{}])[0]
    for track_id, info in frame0_players.items():
        bbox = info['bbox']
        foot_x = (bbox[0] + bbox[2]) / 2   # centre-bottom of bounding box
        foot_y = bbox[3]
        field_pos = mapper.image_to_field(foot_x, foot_y)
        if field_pos:
            x_yd, y_yd = field_pos
            print(f"Player {track_id}: ({foot_x:.0f}, {foot_y:.0f}) px  →  "
                  f"({x_yd:.1f} yd, {y_yd:.1f} yd) on field")
        break   # just show one example
    # ───────────────────────────────────────────────────────────────────────

    # Save output
    save_video(output_video_frames, 'output_videos/output_video1.avi')


if __name__ == '__main__':
    main()
