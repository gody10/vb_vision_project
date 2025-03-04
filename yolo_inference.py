from ultralytics import YOLO

# Load classic model
model = YOLO('yolo11n')

# Load ball fine tuned model
#model = YOLO('models/yolo11_last.pt')

# Inference
#result = model.predict('input_videos/input_video.mp4', conf= 0.2, save= True)
result = model.track('input_videos/input_video.mp4', conf= 0.2, save= True)

# print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)