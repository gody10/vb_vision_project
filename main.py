from utils import (read_video, save_video) 
from trackers import PlayerTracker, BallTracker

def main():
    # Read video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Initialize player tracker
    player_tracker = PlayerTracker(model_path= "yolov8x")

    # Detect players in the video
    player_detections = player_tracker.detect_frames(frames= video_frames, read_from_stub= True, stub_path= "tracker_stubs/player_detections.pkl")

    # Initialize ball tracker
    ball_tracker = BallTracker(model_path= "models/yolo11_last.pt")

    # Detect balls in the video
    ball_detections = ball_tracker.detect_frames(frames= video_frames, read_from_stub= True, stub_path= "tracker_stubs/ball_detections.pkl")

    # Draw output

    ## Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    ## Draw ball bounding boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()