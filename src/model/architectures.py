import torch
import torch.nn as nn

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = norm(in_planes)
        self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = norm(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        return out + x

class CNNClassifier_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True)
        )
        self.features = nn.Sequential(ResBlock(64, 64))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        h = self.down(x)
        h = self.features(h) # 64 channels, 13x13 spatial dimensions
        h = self.pool(h)
        feat = self.flatten(h)   # penultimate feature vector (size 64)
        out = self.fc(feat)
        return out, feat

class CNNClassifier_SVHN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True)
        )
        self.features = nn.Sequential(ResBlock(64, 64))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        h = self.down(x)
        h = self.features(h)
        h = self.pool(h)
        feat = self.flatten(h)   # penultimate feature vector (size 64)
        out = self.fc(feat)
        return out, feat
