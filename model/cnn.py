import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        #zmniejzsa o polowe
        self.pool = nn.MaxPool2d(2, 2)

        # input 64 kanały * 16x16 ( obraz 128x128 po 3 maxpoolingach)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 klasy: kot, pies

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (3,128,128) -> (16,64,64)
        x = self.pool(F.relu(self.conv2(x)))  # (16,64,64) -> (32,32,32)
        x = self.pool(F.relu(self.conv3(x)))  # (32,32,32) -> (64,16,16)

        x = x.view(-1, 64 * 16 * 16)  # (batch_size, cechy)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # TODO aktywacja lub nie

        return x
