import torch
import torchvision.transforms as transforms
from torchvision import models
import cv2

class CourtLineDetector:
    def __init__(self, model_path):
        # Define the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")

        self.model = models.resnet101(pretrained= False).to(self.device)
        # Change the number of output features to 28 (14 keypoints * 2 coordinates)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2).to(self.device)
        # Load the model weights
        self.model.load_state_dict(torch.load(model_path, map_location= self.device))

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transforms(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor).to(self.device)

        keypoints= outputs.squeeze().cpu().numpy()
        original_height, original_width = img_rgb.shape[:2]

        keypoints[0::2] *= (original_width/224.0)
        keypoints[1::2] *= (original_height/224.0)

        return keypoints
    
    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints),2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])

            cv2.putText(image, str(i//2), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []

        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)

        return output_video_frames