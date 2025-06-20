from utils import read_video, save_video
from trackers import Tracker

def main():
    # read video
    video_frames = read_video('input_videos/Test_vid.mp4')

    #initialize tracker
    tracker = Tracker('runs/detect/train/weights/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pk1')

    #draw output
    #draw object tracked
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # save video
    save_video(output_video_frames, 'output_videos/output_video1.avi')
    

if __name__ == '__main__':
    main()