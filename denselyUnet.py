import torch
import torch.nn as nn


class denselyUnet(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(denselyUnet, self).__init__()

        filter = [64, 128, 256, 512, 1024]

        self.conv1_1 = nn.Conv2d(input_channel, filter[0], 3, 1, 1)
        self.BatchNorm1_1 = nn.BatchNorm2d(filter[0])
        self.relu = nn.ReLU()
        self.conv1_2 = nn.Conv2d(filter[0], filter[0], 3, 1, 1)
        self.drop1_2 = nn.Dropout(0)
        # Merge1 = merge([conv1_1, drop1_2], mode='concat', concat_axis=3)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(filter[0] * 2, filter[1], 3, 1, 1)
        self.BatchNorm2_1 = nn.BatchNorm2d(filter[1])
        self.ReLU2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(filter[1], filter[1], 3, 1, 1)
        self.drop2_2 = nn.Dropout(0)
        # Merge2 = merge([conv2_1, drop2_2], mode='concat', concat_axis=3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(filter[1] * 2, filter[2], 3, 1, 1)
        self.BatchNorm3_1 = nn.BatchNorm2d(filter[2])
        self.ReLU3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(filter[2], filter[2], 3, 1, 1)
        self.drop3_2 = nn.Dropout(0)
        # Merge3 = merge([conv3_1, drop3_2], mode='concat', concat_axis=3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4_1 = nn.Conv2d(filter[2] * 2, filter[3], 3, 1, 1)
        self.BatchNorm4_1 = nn.BatchNorm2d(filter[3])
        self.ReLU4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(filter[3], filter[3], 3, 1, 1)
        self.drop4_2 = nn.Dropout(0)
        # Merge4 = merge([conv4_1, drop4_2], mode='concat', concat_axis=3)
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5_1 = nn.Conv2d(filter[3] * 2, filter[4], 3, 1, 1)
        self.BatchNorm5_1 = nn.BatchNorm2d(filter[4])
        self.ReLU5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(filter[4], filter[4], 3, 1, 1)
        self.drop5_2 = nn.Dropout(0)
        # Merge5 = merge([conv5_1, drop5_2], mode='concat', concat_axis=3)
        self.drop5 = nn.Dropout(0.5)

        self.upsampling6 = nn.Upsample(scale_factor=2)

        self.up6 = nn.Conv2d(filter[4] * 2, filter[3], 3, 1, 1)
        # self.merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        self.conv6_1 = nn.Conv2d(filter[3] * 3, filter[3], 3, 1, 1)
        self.BatchNorm6_1 = nn.BatchNorm2d(filter[3])
        self.ReLU6_1 = nn.ReLU()
        self.conv6_2 = nn.Conv2d(filter[3], filter[3], 3, 1, 1)
        self.drop6_2 = nn.Dropout(0)
        # Merge6 = merge([conv6_1, drop6_2], mode='concat', concat_axis=3)

        self.upsampling7 = nn.Upsample(scale_factor=2)
        self.up7 = nn.Conv2d(filter[3] * 2, filter[2], 3, 1, 1)
        # merge7 = merge([Merge3, up7], mode='concat', concat_axis=3)
        self.conv7_1 = nn.Conv2d(filter[2] * 3, filter[2], 3, 1, 1)
        self.BatchNorm7_1 = nn.BatchNorm2d(filter[2])
        self.ReLU7_1 = nn.ReLU()
        self.conv7_2 = nn.Conv2d(filter[2], filter[2], 3, 1, 1)
        self.drop7_2 = nn.Dropout(0)
        # Merge7 = merge([conv7_1, drop7_2], mode='concat', concat_axis=3)

        self.upsampling8 = nn.Upsample(scale_factor=2)
        self.up8 = nn.Conv2d(filter[2] * 2, filter[1], 3, 1, 1)
        # merge8 = merge([Merge2, up8], mode='concat', concat_axis=3)
        self.conv8_1 = nn.Conv2d(filter[1] * 3, filter[1], 3, 1, 1)
        self.BatchNorm8_1 = nn.BatchNorm2d(filter[1])
        self.ReLU8_1 = nn.ReLU()
        self.conv8_2 = nn.Conv2d(filter[1], filter[1], 3, 1, 1)
        self.drop8_2 = nn.Dropout(0)
        # Merge8 = merge([conv8_1, drop8_2], mode='concat', concat_axis=3)

        self.upsampling9 = nn.Upsample(scale_factor=2)
        self.up9 = nn.Conv2d(filter[1] * 2, filter[0], 3, 1, 1)
        # merge9 = merge([Merge1, up9], mode='concat', concat_axis=3)
        self.conv9_1 = nn.Conv2d(filter[0] * 3, filter[0], 3, 1, 1)
        self.BatchNorm9_1 = nn.BatchNorm2d(filter[0])
        self.ReLU9_1 = nn.ReLU()
        self.conv9_2 = nn.Conv2d(filter[0], filter[0], 3, 1, 1)
        self.drop9_2 = nn.Dropout(0)
        # Merge9 = merge([conv9_1, drop9_2], mode='concat', concat_axis=3)

        self.conv9 = nn.Conv2d(filter[0] * 2, 2, 3, 1, 1)
        self.conv10 = nn.Conv2d(2, output_channel, 3, 1, 1)  # sigmoid
        self.Tanh10 = nn.Tanh()

    def forward(self, x):
        x1_1 = self.conv1_1(x)
        x1_2 = self.BatchNorm1_1(x1_1)
        x1_2 = self.relu(x1_2)
        x1_2 = self.conv1_2(x1_2)
        x1_2 = self.drop1_2(x1_2)

        m1 = torch.cat((x1_2, x1_1), dim=1)

        pool1 = self.pool1(m1)

        x2_1 = self.conv2_1(pool1)
        x2_2 = self.BatchNorm2_1(x2_1)
        x2_2 = self.relu(x2_2)
        x2_2 = self.conv2_2(x2_2)
        x2_2 = self.drop2_2(x2_2)

        m2 = torch.cat((x2_1, x2_2), dim=1)

        pool2 = self.pool2(m2)

        x3_1 = self.conv3_1(pool2)
        x3_2 = self.BatchNorm3_1(x3_1)
        x3_2 = self.relu(x3_2)
        x3_2 = self.conv3_2(x3_2)
        x3_2 = self.drop3_2(x3_2)

        M3 = torch.cat((x3_1, x3_2), dim=1)

        pool3 = self.pool3(M3)

        x4_1 = self.conv4_1(pool3)
        x4_2 = self.BatchNorm4_1(x4_1)
        x4_2 = self.relu(x4_2)
        x4_2 = self.conv4_2(x4_2)
        x4_2 = self.drop4_2(x4_2)

        M4 = torch.cat((x4_2, x4_1), dim=1)

        x4_2 = self.drop4(M4)
        pool4 = self.pool4(x4_2)

        x5_1 = self.conv5_1(pool4)
        x5_2 = self.BatchNorm5_1(x5_1)
        x5_2 = self.relu(x5_2)
        x5_2 = self.conv5_2(x5_2)
        x5_2 = self.drop5_2(x5_2)

        M5 = torch.cat((x5_1, x5_2), dim=1)

        x5_2 = self.drop5(M5)
        pool5 = self.upsampling6(x5_2)

        up6 = self.up6(pool5)
        m6 = torch.cat((up6, x4_2), dim=1)
        x6_1 = self.conv6_1(m6)
        x6_2 = self.BatchNorm6_1(x6_1)
        x6_2 = self.relu(x6_2)
        x6_2 = self.conv6_2(x6_2)
        x6_2 = self.drop6_2(x6_2)

        M6 = torch.cat((x6_1, x6_2), dim=1)

        pool6 = self.upsampling7(M6)
        up7 = self.up7(pool6)
        m7 = torch.cat((M3, up7), dim=1)
        x7_1 = self.conv7_1(m7)
        x7_2 = self.conv7_2(x7_1)
        x7_2 = self.BatchNorm7_1(x7_2)
        x7_2 = self.relu(x7_2)
        x7_2 = self.conv7_2(x7_2)
        x7_2 = self.drop7_2(x7_2)

        M7 = torch.cat((x7_1, x7_2), dim=1)

        pool7 = self.upsampling8(M7)
        up8 = self.up8(pool7)
        m8 = torch.cat((m2, up8), dim=1)
        x8_1 = self.conv8_1(m8)
        x8_2 = self.BatchNorm8_1(x8_1)
        x8_2 = self.relu(x8_2)
        x8_2 = self.conv8_2(x8_2)
        x8_2 = self.drop8_2(x8_2)
        M8 = torch.cat((x8_1, x8_2), dim=1)

        pool8 = self.upsampling8(M8)
        up9 = self.up9(pool8)
        m9 = torch.cat((m1, up9), dim=1)
        x9_1 = self.conv9_1(m9)
        x9_2 = self.BatchNorm9_1(x9_1)
        x9_2 = self.relu(x9_2)
        x9_2 = self.conv9_2(x9_2)
        x9_2 = self.drop9_2(x9_2)
        M9 = torch.cat((x9_1, x9_2), dim=1)

        x10 = self.conv9(M9)
        out = self.conv10(x10)
        out = self.Tanh10(out)

        return out







