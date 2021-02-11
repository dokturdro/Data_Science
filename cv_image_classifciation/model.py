import torch
import torch.nn as nn
import torch.nn.functional as F

class Flower_Net_1(nn.Module):
    def __init__(self):
        super(Flower_Net_1, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.AvgPool2d(kernel_size=3, padding=1))
        self.layer2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.MaxPool2d(kernel_size=3, padding=1))
        self.layer3 = nn.Sequential(nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.MaxPool2d(kernel_size=3, padding=1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        out = self.flatten(x)
        return out


class Flower_Net_2(nn.Module):
    def __init__(self):
        super(Flower_Net_2, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.AvgPool2d(kernel_size=3, padding=1))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.MaxPool2d(kernel_size=3, padding=1))
        self.layer3 = nn.Sequential(nn.Conv2d(32, 8, kernel_size=3, padding=1), nn.MaxPool2d(kernel_size=3, padding=1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        out = self.flatten(x)
        return out


class Flower_Net_3(nn.Module):
    def __init__(self):
        super(Flower_Net_3, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.AvgPool2d(kernel_size=3, padding=1))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 8, kernel_size=3, padding=1), nn.MaxPool2d(kernel_size=3, padding=1))
        self.layer3 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, padding=1), nn.MaxPool2d(kernel_size=3, padding=1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        out = self.flatten(x)
        return out


class ensemble_Net(nn.Module):
    def __init__(self):
        super(ensemble_Net, self).__init__()
        f1 = Flower_Net_1()
        f2 = Flower_Net_2()
        f3 = Flower_Net_3()
        self.e1 = f1
        self.e2 = f2
        self.e3 = f3
        self.avgpool = nn.AvgPool1d(kernel_size=1)
        self.fc1 = nn.Linear(216, 30)
        self.fc2 = nn.Linear(30, 5)

    def forward(self, x):
        o1 = self.e1(x)

        o2 = self.e2(x)
        o3 = self.e3(x)
        x = torch.cat((o1, o2, o3), dim=1)
        x = self.fc1(x)
        out = self.fc2(x)

        return out