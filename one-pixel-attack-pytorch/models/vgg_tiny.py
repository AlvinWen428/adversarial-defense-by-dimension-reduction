import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.layer1 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # 1-2 conv layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            # 1 Pooling layer
            nn.AvgPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # 1-2 conv layer
            nn.Conv2d(32, 8, kernel_size=3, padding=1))

        # self.layer6 = nn.Sequential(
        #
        #     # 6 Fully connected layer
        #     nn.Linear(6272, 128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU())
        #
        # self.layer7 = nn.Sequential(
        #     # 7 Fully connected layer
        #     # Dropout layer omitted since batch normalization is used.
        #     nn.Linear(128, output_channel))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        features = out.view(out.shape[0], -1)
        return features


# class Fc(nn.Module):
#     def __init__(self, input_channel, output_channel):
#         super(Fc, self).__init__()
#         self.layer = nn.Linear(input_channel, output_channel)
#
#     def forward(self, x):
#         out = self.layer(x)
#         return out


class Fc(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Fc, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_channel, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, output_channel))
        self.initialization()


    def forward(self, x):
        out = self.layer(x)
        return out
    
    def initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_normal(m.weight)

class vgg_tiny_complete(nn.Module):
    def __init__(self):
        super(vgg_tiny_complete, self).__init__()
        self.layer1 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # 1-2 conv layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            # 1 Pooling layer
            nn.AvgPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # 1-2 conv layer
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8))

        
        self.layer3 = nn.Sequential(
            # nn.Linear(1568, 64),
            # nn.BatchNorm1d(64),
            # nn.LeakyReLU(),

            # nn.Linear(64, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(1568, 10))
        self.initialization()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        features = x.view(x.shape[0], -1)
        out = self.layer3(features)
        return out
    
    def initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_normal(m.weight)
