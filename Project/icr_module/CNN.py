import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes=40, batch_size = 4):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

#
        self.fc1 = nn.Linear(25568 * batch_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 25568) # 16 * 5 * 5
        x = F.relu(self.fc1(x))
        embedding_layer = F.relu(self.fc2(x))
        x = self.fc3(embedding_layer)
        return x, embedding_layer