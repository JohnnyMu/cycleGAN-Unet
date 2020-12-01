import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init


class ConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, dropout=0.0):
        super(ConvBlock, self).__init__()

        self.dropout = dropout
        self.interChannel = 4 * output_channel
        self.batchnorm1 = nn.BatchNorm2d(input_channel)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(input_channel, self.interChannel, 1)
        self.dropoutlayer = nn.Dropout(dropout)

        self.batchnorm2 = nn.BatchNorm2d(self.interChannel)
        self.conv2 = nn.Conv2d(self.interChannel, output_channel, 3, padding=1)

    def forward(self, x):
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv1(x)
        if self.dropout:
            x = self.dropoutlayer(x)
        x = self.conv2(self.relu(self.batchnorm2(x)))
        if self.dropout:
            x = self.dropoutlayer(x)
        return x


class DenseBlock5(torch.nn.Module):
    def __init__(self, nb_filter, growth_rate, nb_block=4, dropout=0.0, grow_nb_filters=True):
        super(DenseBlock5, self).__init__()

        self.grow_nb_filters = grow_nb_filters
        self.nb_block = nb_block
        self.dropout = dropout
        self.nb_filter = nb_filter
        self.nb_filter = nb_filter
        self.growth_rate = growth_rate
        self.denselayer1 = ConvBlock(self.nb_filter, self.growth_rate, self.dropout)
        self.denselayer2 = ConvBlock(self.nb_filter + self.growth_rate, self.growth_rate, self.dropout)
        self.denselayer3 = ConvBlock(self.nb_filter + self.growth_rate * 2, self.growth_rate, self.dropout)
        self.denselayer4 = ConvBlock(self.nb_filter + self.growth_rate * 3, self.growth_rate, self.dropout)
        self.denselayer5 = ConvBlock(self.nb_filter + self.growth_rate * 4, self.growth_rate, self.dropout)
        # self.denselayer6 = ConvBlock(self.nb_filter + self.growth_rate * 5, self.growth_rate, self.dropout)
    def forward(self, x):
        shortcut = x
        # nb_filter = self.nb_filter
        x = self.denselayer1(x)
        shortcut = torch.cat((shortcut, x), dim=1)
        x = self.denselayer2(shortcut)
        shortcut = torch.cat((shortcut, x), dim=1)
        x = self.denselayer3(shortcut)
        shortcut = torch.cat((shortcut, x), dim=1)
        x = self.denselayer4(shortcut)
        shortcut = torch.cat((shortcut, x), dim=1)
        x = self.denselayer5(shortcut)
        shortcut = torch.cat((shortcut, x), dim=1)
        # x = self.denselayer6(shortcut)
        # shortcut = torch.cat((shortcut, x), dim=1)

        return shortcut #, self.nb_filter + self.growth_rate * 4


class TransitionBlock(torch.nn.Module):
    def __init__(self, nChannels, nOutChannels, maxpool=True, dropout=0.0):
        super(TransitionBlock, self).__init__()

        self.batchnorm = nn.BatchNorm2d(nChannels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(nChannels, nOutChannels, 1)
        self.dropout = dropout
        self.dropoutlayer = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(self.relu(self.batchnorm(x)))


class GeneratorDenseNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorDenseNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        # for _ in range(num_residual_blocks):
        #     model += [ResidualBlock(out_features)]
        model += [DenseBlock5(out_features, 128), TransitionBlock(out_features + 128 * 5, out_features)]
        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


