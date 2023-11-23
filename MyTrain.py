import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

# 数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, excel_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_excel(excel_path)  # 读取 Excel 文件
        self.image_paths = [os.path.join(root_dir, f) for f in self.df['图片名称']]
        self.anxiety_scores = self.df['焦虑分数'].tolist()
        self.depression_scores = self.df['抑郁分数'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        anxiety_score = self.anxiety_scores[idx]
        depression_score = self.depression_scores[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, anxiety_score, depression_score

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# 数据集和 DataLoader
dataset = CustomDataset("Pics", "得分.xlsx", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 神经网络模型
class EmotionPredictor(nn.Module):
    def __init__(self):
        super(EmotionPredictor, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 2)  # 2个输出节点，对应焦虑分数和抑郁分数
        )

    def forward(self, x):
        x = self.model1(x)
        return x

# 训练模型（以焦虑分数为例）
emotion_predictor = EmotionPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(emotion_predictor.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, anxiety_targets, _ = data  # 注意这里只使用了焦虑分数，你也可以同时使用抑郁分数
        outputs = emotion_predictor(imgs)
        result_loss = criterion(outputs[:, 0], anxiety_targets.float())  # 只考虑第一个输出节点，即焦虑分数
        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()
        running_loss += result_loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

# 预测
# 假设 test_image 是一张测试图像，需要进行相应的预处理
test_image = transform(Image.open("Pics/test_image.png").convert("RGB")).unsqueeze(0)
predicted_scores = emotion_predictor(test_image)

print(f"Predicted Anxiety Score: {predicted_scores[0, 0].item()}")
print(f"Predicted Depression Score: {predicted_scores[0, 1].item()}")
