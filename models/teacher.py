import torch
import torch.nn as nn

class TeacherCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,kernel_size = 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,kernel_size = 3, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),


            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,1)
        return self.classifier(x)

# # Test
# model = TeacherCNN()
# x = torch.randn(4,1,28,28)
# y = model(x)
# print("Input: ",x.shape)
# print("Output: ",y.shape)
# print("Parameters:", sum(p.numel() for p in model.parameters()))