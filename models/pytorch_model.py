import torch
import torch.nn as nn

class ADEncoder(nn.Module):

    def __init__(self):
        super(ADEncoder,self).__init__()

        # Encoder 1024x1024 greyscale normalized to 0..1 as input
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=2,
                      padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU())
    
    def forward(self,x):
        return self.forward(x)


model = ADEncoder()
x = torch.randn(1,1,1024,1024)
output = model.conv1(x)
output = model.conv2(output)
print(output.shape)