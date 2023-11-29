from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

import torch.nn as nn


class CNN(Module):
    def __init__(self, numChannels, classes):
        super().__init__()
        self.input = None
        self.y = None
        self.classes = classes
        # call the parent constructor
        super(CNN, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=32,
                            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=32, out_channels=50,
                            kernel_size=(3, 3))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=1800, out_features=500)
        self.relu3 = ReLU()
        # initialize our softmax classifier
        self.last_layer = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        self.input = x
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.last_layer(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output

    def reInitializeFinalLayer(self):
        # freeze all layers except the last and reset its parameters
        self.last_layer = Linear(in_features=500, out_features=self.classes)
        for name, layer in self.named_parameters():
            if 'last' in name:
                layer.requires_grad = True
            else:
                layer.requires_grad = False


class VGG13(Module):
    def __init__(self, numChannels, classes):
        super(VGG13, self).__init__()
        self.features = Sequential(
            Conv2d(numChannels, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 128, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(128, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(256, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = AdaptiveAvgPool2d((7, 7))  # Adaptive pooling to (7, 7)
        self.classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = flatten(x, 1)  # Flatten the tensor
        x = self.classifier(x)
        return x

    def reInitializeFinalLayer(self):
        # freeze all layers except the last and reset its parameters
        self.last_layer = Linear(in_features=500, out_features=self.classes)
        for name, layer in self.named_parameters():
            if 'last' in name:
                layer.requires_grad = True
            else:
                layer.requires_grad = False


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.input = None
        self.y = None
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.last_layer = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.input = x
        x = self.conv1(x)
        # print(f"Conv1 output size: {x.size()}")

        x = self.maxpool(x)
        # print(f"MaxPool output size: {x.size()}")

        x = self.layer0(x)
        # print(f"Layer0 output size: {x.size()}")

        x = self.layer1(x)
        # print(f"Layer1 output size: {x.size()}")

        x = self.layer2(x)
        # print(f"Layer2 output size: {x.size()}")

        x = self.layer3(x)
        # print(f"Layer3 output size: {x.size()}")

        x = self.avgpool(x)
        # print(f"AvgPool output size: {x.size()}")

        x = x.view(x.size(0), -1)
        output = self.last_layer(x)
        # print(f"Last layer output size: {x.size()}")

        return output

    def reInitializeFinalLayer(self):
        # freeze all layers except the last and reset its parameters
        self.last_layer = nn.Linear(512, 10)
        for name, layer in self.named_parameters():
            if 'last' in name:
                layer.requires_grad = True
            else:
                layer.requires_grad = False
