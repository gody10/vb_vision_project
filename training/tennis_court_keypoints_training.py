import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import json
import cv2
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class KeypointsDataset(Dataset):

    def __init__(self, img_dir, data_file):
        self.img_dir = img_dir

        with open(data_file, "r") as f:
            self.data = json.load(f)
        
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img = cv2.imread(f"{self.img_dir}/{item['id']}.png")
        h,w = img.shape[:2]

        # CV2 reads images in BGR format, convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transforms(img)
        kps = np.array(item['kps']).flatten()
        kps = kps.astype(np.float32)

        # Since we resized the image to 224x224, we need to scale the keypoints
        kps[0::2] *= (224.0/w)
        kps[1::2] *= (224.0/h)

        return img, kps
    

# Change directories to the location of the dataset
os.chdir("C:/Users/odiamant/Desktop/Work/my_stuff/vb_vision_project")

train_dataset = KeypointsDataset(
    img_dir= "training/tennis_court_det_dataset/data/images", 
    data_file= "training/tennis_court_det_dataset/data/data_train.json"
    )

valid_dataset = KeypointsDataset(
    img_dir= "training/tennis_court_det_dataset/data/images", 
    data_file= "training/tennis_court_det_dataset/data/data_val.json"
    )

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True)

model = models.resnet101(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 14*2) # Replaces the last layer with a new one that outputs 14*2 values (x,y for each keypoint)

model = model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs= 30

for epoch in range(epochs):
    for i, (img,kps) in enumerate(train_loader): 
        img= img.to(device)
        kps = kps.to(device)

        optimizer.zero_grad()
        pred= model(img)
        loss = criterion(pred, kps)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")  

torch.save(model.state_dict(), "keypoints_model.pth")