from utils import (read_video, save_video) 
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2

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

    # Interpolate ball positions
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Detect court lines
    court_line_detector = CourtLineDetector(model_path= "models/keypoints_model.pt")

    # Detect court lines in the video
    court_keypoints= court_line_detector.predict(video_frames[0])

    # Choose and filter players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections, n_of_players= 2)

    # Draw output

    ## Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    ## Draw ball bounding boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw court lines
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()