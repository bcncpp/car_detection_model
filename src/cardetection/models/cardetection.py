import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageCarDetection(nn.Module):
    def __init__(self, num_classes=4):
        super(ImageCarDetection, self).__init__()
        self.conv1 = nn.Conv2d(3, 17, kernel_size=3, padding=1)  # Assuming input has 3 channels (RGB)
        self.conv2 = nn.Conv2d(17, 39, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(39, 39, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(39, 50, kernel_size=3, padding=1)
        # doing pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.flattened_size = 50 * 16 * 9 
        self.fc1 = nn.Linear(self.flattened_size, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
