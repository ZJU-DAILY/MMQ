import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, N):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=16,  
                kernel_size=3,  
                stride=1,  
                padding=1,  
            ),  
            nn.ReLU(), 
            nn.MaxPool2d(
                kernel_size=2, 
                stride=2,  
            ),  
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 6 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, N)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output