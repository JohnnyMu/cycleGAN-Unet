import torch
import torch.nn as nn
import torch.functional as F


class DenseBlock(torch.nn.Module):
    def __init__(self, nb_filter, growth_rate, nb_block=4, dropout=0.0, grow_nb_filters=True):
        super(DenseBlock, self).__init__()

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
        # self.denselayer5 = ConvBlock(self.nb_filter + self.growth_rate * 4, self.growth_rate, self.dropout)
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
        # x = self.denselayer5(shortcut)
        # shortcut = torch.cat((shortcut, x), dim=1)
        # x = self.denselayer6(shortcut)
        # shortcut = torch.cat((shortcut, x), dim=1)

        return shortcut #, self.nb_filter + self.growth_rate * 4
        # for i in range(self.nb_block):
        #     dense_layer = self._make_dense_layer(nb_filter, self.growth_rate, self.dropout)
        #     x = dense_layer(x)
        #     shortcut = torch.cat((shortcut, x), dim=1)
        #
        #     if self.grow_nb_filters:
        #         nb_filter += self.growth_rate
        # return shortcut, nb_filter

    # def _make_dense_layer(self, nb_filter, nb_growth_rate, dropout):
    #     return ConvBlock(nb_filter, nb_growth_rate, dropout)


class ConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, dropout=0.0):
        super(ConvBlock, self).__init__()

        self.dropout = dropout
        self.interChannel = 4 * output_channel
        self.batchnorm1 = nn.BatchNorm2d(input_channel)
        self.relu = nn.LeakyReLU(0.01)

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


class TransitionBlock(torch.nn.Module):
    def __init__(self, nb_filter, maxpool=True, dropout=0.0):
        super(TransitionBlock, self).__init__()

        self.batchnorm = nn.BatchNorm2d(nb_filter)
        self.relu = nn.LeakyReLU(0.01)
        self.conv = nn.Conv2d(nb_filter, nb_filter, 1)
        self.dropout = dropout
        self.dropoutlayer = nn.Dropout(dropout)

        self.maxpool = maxpool
        self.maxpooling = nn.MaxPool2d(2, 2)
        self.convdown = nn.Conv2d(nb_filter, nb_filter, 3, 2, 1)

    def forward(self, x):
        x = self.conv(self.relu(self.batchnorm(x)))
        if self.dropout:
            x = self.dropoutlayer(x)
        if self.maxpool:
            x = self.maxpooling(x)
        else:
            x = self.convdown(x)
        return x


class DenseUnet(torch.nn.Module):
    def __init__(self, input_channel, output_channel, nb_layer=4, growth_rate=48, dropout=0.0, maxpool=True):
        super(DenseUnet, self).__init__()
        self.maxpool = maxpool
        self.nb_layer = nb_layer
        self.nb_filter = 96
        self.growth_rate = growth_rate
        self.dropout = dropout

        self.conv1 = nn.Conv2d(input_channel, self.nb_filter, 7, 2, 3)
        self.batchnorm1 = nn.BatchNorm2d(self.nb_filter)
        self.maxpooling1 = nn.MaxPool2d(2, 2)
        self.convdown = nn.Conv2d(self.nb_filter, self.nb_filter, 3, 2, 1)

        self.relu = nn.LeakyReLU(0.01)
        self.upsampling = nn.Upsample(scale_factor=2)

        self.upsamplingconv1 = torch.nn.ConvTranspose2d(self.nb_filter + self.growth_rate * 4 * 4,
                                                       self.nb_filter + self.growth_rate * 4 * 4, kernel_size=4, stride=2,
                                                       padding=1)
        self.convup1 = nn.Conv2d(self.nb_filter + self.growth_rate * 4 * 3, self.nb_filter + self.growth_rate * 4 * 4, 1)
        self.convup2 = nn.Conv2d(self.nb_filter + self.growth_rate * 4 * 4, self.nb_filter + self.growth_rate * 4 * 2, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(self.nb_filter + self.growth_rate * 4 * 2)
        self.upsamplingconv2 = torch.nn.ConvTranspose2d(self.nb_filter + self.growth_rate * 4 * 2,
                                                        self.nb_filter + self.growth_rate * 4 * 2, kernel_size=4,
                                                        stride=2,
                                                        padding=1)
        self.convup3 = nn.Conv2d(self.nb_filter + self.growth_rate * 4 * 2, self.nb_filter + self.growth_rate * 4, 3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(self.nb_filter + self.growth_rate * 4)
        self.upsamplingconv3 = torch.nn.ConvTranspose2d(self.nb_filter + self.growth_rate * 4,
                                                        self.nb_filter + self.growth_rate * 4, kernel_size=4,
                                                        stride=2,
                                                        padding=1)
        self.convup4 = nn.Conv2d(self.nb_filter + self.growth_rate * 4, 96, 3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(96)
        self.upsamplingconv4 = torch.nn.ConvTranspose2d(96, 96, kernel_size=4,
                                                        stride=2,
                                                        padding=1)
        self.convup5 = nn.Conv2d(96, 96, 3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(96)
        self.upsamplingconv5 = torch.nn.ConvTranspose2d(96, 96, kernel_size=4,
                                                        stride=2,
                                                        padding=1)
        self.convup6 = nn.Conv2d(96,64, 3, padding=1)
        self.dropout6 = nn.Dropout(0.3)
        self.batchnorm6 = nn.BatchNorm2d(64)
        self.convout = nn.Conv2d(64, output_channel, 1)

        self.denseblock1 = DenseBlock(self.nb_filter, self.growth_rate, self.nb_layer, self.dropout)
        self.transition1 = TransitionBlock(self.nb_filter + self.growth_rate * 4, maxpool=maxpool)
        self.denseblock2 = DenseBlock(self.nb_filter + self.growth_rate * 4, self.growth_rate, self.nb_layer, self.dropout)
        self.transition2 = TransitionBlock(self.nb_filter + self.growth_rate * 4 * 2, maxpool=maxpool)
        self.denseblock3 = DenseBlock(self.nb_filter + self.growth_rate * 4 * 2, self.growth_rate, self.nb_layer, self.dropout)
        self.transition3 = TransitionBlock(self.nb_filter + self.growth_rate * 4 * 3, maxpool=maxpool)
        self.denseblock4 = DenseBlock(self.nb_filter + self.growth_rate * 4 * 3, self.growth_rate, self.nb_layer, self.dropout)
        self.batchnormdown1 = nn.BatchNorm2d(self.nb_filter + self.growth_rate * 4 * 4)
        self.tanh = nn.Tanh()
    def forward(self, x):
        box = []
        ####### down
        x = self.relu(self.batchnorm1(self.conv1(x)))
        box.append(x) #64*64*96
        if self.maxpool:
            x = self.maxpooling1(x)
        else:
            x = self.convdown(x)
        nb_filter = self.nb_filter
        x = self.denseblock1(x)
        box.append(x) #32*32*288
        x = self.transition1(x)
        x = self.denseblock2(x)
        box.append(x) #16*16*480
        x = self.transition2(x)
        x = self.denseblock3(x)
        box.append(x) #  8*8*672
        x = self.transition3(x)
        x = self.denseblock4(x)
        x = self.batchnormdown1(x)
        # for i in range(self.nb_layer - 1):
        #     denseblock = self._make_dense_block(nb_filter, self.growth_rate, self.nb_layer, self.dropout)
        #     x, nb_filter = denseblock(x)
        #     box.append(x)
        #     x = self._make_transition_layer(nb_filter, self.maxpool)#4*4*674
        #
        # denseblocklast = self._make_dense_block(nb_filter,self.growth_rate, self.nb_layer, self.dropout)
        # x, nb_filter = denseblocklast(x)
        # batchlast = self._make_batchnorm(nb_filter)
        # x = batchlast(x)
        x = self.relu(x)
        box.append(x)#4*4*864

        #######  up
    # if self.maxpool:
        up0 = self.upsampling(x) #8*8*864
    # else:
    #     up0 = self.upsamplingconv1(x)
        line0 = self.convup1(box[3])
        up0sum = torch.add(up0, line0)
        conv_up0 = self.convup2(up0sum)
        conv_b_up0 = self.batchnorm2(conv_up0)
        conv_b_r_up0 = self.relu(conv_b_up0)

    # if self.maxpool:
        up1 = self.upsampling(conv_b_r_up0)
    # else:
    #     up1 = self.upsamplingconv2(conv_b_r_up0)
        up1sum = torch.add(up1, box[2])
        conv_up1 = self.convup3(up1sum)
        conv_b_up1 = self.batchnorm3(conv_up1)
        conv_b_r_up1 = self.relu(conv_b_up1)

    # if self.maxpool:
        up2 = self.upsampling(conv_b_r_up1)
    # else:
    #     up2 = self.upsamplingconv3(conv_b_r_up1)
        up2sum = torch.add(up2, box[1])
        conv_up2 = self.convup4(up2sum)
        conv_b_up2 = self.batchnorm4(conv_up2)
        conv_b_r_up2 = self.relu(conv_b_up2)

    # if self.maxpool:
        up3 = self.upsampling(conv_b_r_up2)
    # else:
    #     up3 = self.upsamplingconv4(conv_b_r_up2)
        up3sum = torch.add(up3, box[0])
        conv_up3 = self.convup5(up3sum)
        conv_b_up3 = self.batchnorm5(conv_up3)
        conv_b_r_up3 = self.relu(conv_b_up3)

    # if self.maxpool:
        up4 = self.upsampling(conv_b_r_up3)
    # else:
    #     up4 = self.upsamplingconv5(conv_b_r_up3)
        conv_up4 = self.convup6(up4)
        conv_d_up4 = self.dropout6(conv_up4)
        conv_b_up4 = self.batchnorm6(conv_d_up4)
        conv_b_r_up4 = self.relu(conv_b_up4)

        x = self.convout(conv_b_r_up4)
        x = self.tanh(x)

        return x





    # def _make_dense_block(self, nb_filter, growth_rate, nb_block, dropout):
    #     return DenseBlock(nb_filter, growth_rate, nb_block, dropout)
    #
    # def _make_transition_layer(self, nb_filter, maxpool):
    #     return TransitionBlock(nb_filter, maxpool=maxpool)
    #
    # def _make_batchnorm(self,nb_filter):
    #     return nn.BatchNorm2d(nb_filter)

