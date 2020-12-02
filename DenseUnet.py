import torch
import torch.nn as nn


class DenseLayer(torch.nn.Module):
    def __init__(self, input_channel, output_channel, dropout=0.0):
        super(DenseLayer, self).__init__()

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


class DenseBlock(torch.nn.Module):
    def __init__(self, nb_filter, growth_rate, dropout=0.0, grow_nb_filters=True):
        super(DenseBlock, self).__init__()

        self.grow_nb_filters = grow_nb_filters
        self.dropout = dropout
        self.nb_filter = nb_filter
        self.nb_filter = nb_filter
        self.growth_rate = growth_rate
        self.denselayer1 = DenseLayer(self.nb_filter, self.growth_rate, self.dropout)
        self.denselayer2 = DenseLayer(self.nb_filter + self.growth_rate, self.growth_rate, self.dropout)
        # self.denselayer3 = DenseLayer(self.nb_filter + self.growth_rate * 2, self.growth_rate, self.dropout)
        # self.denselayer4 = DenseLayer(self.nb_filter + self.growth_rate * 3, self.growth_rate, self.dropout)
        # self.denselayer5 = DenseLayer(self.nb_filter + self.growth_rate * 4, self.growth_rate, self.dropout)
        # self.denselayer6 = DenseLayer(self.nb_filter + self.growth_rate * 5, self.growth_rate, self.dropout)
    def forward(self, x):
        shortcut = x
        # nb_filter = self.nb_filter
        x = self.denselayer1(x)
        shortcut = torch.cat((shortcut, x), dim=1)
        x = self.denselayer2(shortcut)
        shortcut = torch.cat((shortcut, x), dim=1)
        # x = self.denselayer3(shortcut)
        # shortcut = torch.cat((shortcut, x), dim=1)
        # x = self.denselayer4(shortcut)
        # shortcut = torch.cat((shortcut, x), dim=1)
        # x = self.denselayer5(shortcut)
        # shortcut = torch.cat((shortcut, x), dim=1)
        # x = self.denselayer6(shortcut)
        # shortcut = torch.cat((shortcut, x), dim=1)

        return shortcut #, self.nb_filter + self.growth_rate * 4


class DownBlock(torch.nn.Module):
    def __init__(self, nb_filter, growth_rate):
        super(DownBlock, self).__init__()

        self.denseblock1 = DenseBlock(nb_filter, growth_rate)
        self.denseblock2 = DenseBlock(nb_filter + growth_rate * 3, growth_rate)
        self.maxpooling = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.denseblock1(x)
        x = self.denseblock2(x)
        x1 = self.maxpooling(x)
        return x, x1


class UpBlock(torch.nn.Module):
    def __init__(self, nb_filter,):
        super(UpBlock, self).__init__()

        self.upsampling = nn.Upsample(scale_factor=2)


class UNet(torch.nn.Module):
    def __init__(self, input_channel, output_channel, growth_rate):
        super(UNet, self).__init__()

