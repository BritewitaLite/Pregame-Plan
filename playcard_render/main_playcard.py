import os
import sys

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

import cv2
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from camera_movement_est.camera_mov_est import CameraMovementEstimator

from hough_detector.hough_field_homography import build_hough_mapper, HoughFieldAnnotator
from playcard_render.playcard_renderer import PlaycardRenderer, save_playcard_snapshot


def main():
    # ── 1. Read video ────────────────────────────────────────────────────────
    video_frames = read_video('input_videos/test2.mp4')

    # ── 2. Player tracker ────────────────────────────────────────────────────
    tracker = Tracker('models/player_detector.pt')
    tracks  = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs2.pk1',
    )

    # ── 3. Hough field homography (no ML model) ──────────────────────────────
    mapper, homographies = build_hough_mapper(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/hough_homography.pkl',
    )

    # ── 4. Camera movement estimator ─────────────────────────────────────────
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stubs2.pk1',
    )

    # ── 5. Build the play card renderer ─────────────────────────────────────
    renderer = PlaycardRenderer(
        team_a_color=(50, 100, 220),    # blue – assign to offense or home team
        team_b_color=(220, 50, 50),     # red  – defense / away team
        ball_color  =(  0, 200, 255),   # yellow-ish ball dot
        player_radius=9,
        trail_length=25,                # how many frames of movement trail
    )

    # ── 6a. Annotated camera view (Hough lines + tracker boxes) ─────────────
    annotated_frames = tracker.draw_annotations(video_frames, tracks)
    annotated_frames = HoughFieldAnnotator().draw(
        annotated_frames, homographies, draw_grid=True
    )
    annotated_frames = camera_movement_estimator.draw_camera_movement(
        annotated_frames, camera_movement_per_frame
    )

    # ── 6b. Play card composite video ───────────────────────────────────────
    #   mode options:
    #     "composite" → side-by-side (original | play card)
    #     "minimap"   → original frame with play card inset bottom-right
    #     "playcard"  → play card only (bird's-eye field diagram)
    
    # Combine all position groups into 'players', 'Ref' becomes 'referees'
    combined_tracks = {'players': [], 'referees': [], 'ball': []}
    n = len(video_frames)
    for i in range(n):
        frame_players = {}
        frame_refs = {}
        frame_ball = {}
        
        # Combine all position groups except 'Ref' into players
        for pos_key in tracks.keys():
            if pos_key == 'Ref':
                frame_refs.update(tracks[pos_key][i] if i < len(tracks[pos_key]) else {})
            else:
                frame_players.update(tracks[pos_key][i] if i < len(tracks[pos_key]) else {})
        
        combined_tracks['players'].append(frame_players)
        combined_tracks['referees'].append(frame_refs)
        combined_tracks['ball'].append(frame_ball)  # Empty for now
    
    playcard_frames = renderer.render_video(
        video_frames,
        homographies,
        combined_tracks,
        mode="composite",           # ← change to "minimap" or "playcard" as needed
    )

    # ── 7. Save videos ───────────────────────────────────────────────────────
    save_video(annotated_frames,  'output_videos/output_annotated.avi')
    save_video(playcard_frames,   'output_videos/output_playcard.avi')

    # ── 8. Save a still snapshot of the first good frame ────────────────────
    first_good = next(
        (i for i, H in enumerate(homographies) if H is not None), 0
    )
    frame_players = combined_tracks['players'][first_good]
    frame_refs    = combined_tracks['referees'][first_good]
    frame_ball    = combined_tracks['ball'][first_good]

    save_playcard_snapshot(
        frame=video_frames[first_good],
        H=homographies[first_good],
        frame_tracks={
            'players':  frame_players,
            'referees': frame_refs,
            'ball':     frame_ball,
        },
        out_path='output_videos/playcard_snapshot.jpg',
        mode='composite',
    )

    # ── 9. Console field-position report for first frame ────────────────────
    print("\n── Player Field Positions (Frame {}) ──".format(first_good))
    for tid, info in frame_players.items():
        bbox  = info['bbox']
        foot_x = (bbox[0] + bbox[2]) / 2
        foot_y = bbox[3]
        pos = mapper.image_to_field(foot_x, foot_y)
        if pos:
            x_yd, y_yd = pos
            # Convert BEV y to field yard (0=back of EZ, 10=goal line, 60=50yd)
            field_yd = y_yd - 10.0   # offset past near end zone
            side     = "L" if x_yd < FIELD_W_YD / 2 else "R"
            hash_pos = abs(x_yd - FIELD_W_YD / 2)
            print(f"  Player {tid:>3}: {field_yd:5.1f} yd  |  hash {side} ({hash_pos:.1f} yd from centre)")
    print()


FIELD_W_YD = 53.3   # local reference for the console report


if __name__ == '__main__':
    main()
