import torch
import torch.nn as nn
from collections import OrderedDict


class MCN(nn.Module):
    def __init__(self, args):
        super(MCN, self).__init__()
        num_init_features = args.num_init_features
        num_classes = args.num_classes

        # First convolution
        self.features0 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     1,
                     num_init_features,
                     kernel_size=3,  # 7
                     stride=(1, 1, 1),  # 2
                     padding=(1, 1, 1),  # 3
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))

        self.features1 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     1,
                     num_init_features,
                     kernel_size=5,  # 7
                     stride=(1, 1, 1),  # 2
                     padding=(2, 2, 2),  # 3
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))
        self.features2 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     1,
                     num_init_features,
                     kernel_size=7,  # 7
                     stride=(1, 1, 1),  # 2
                     padding=(3, 3, 3),  # 3
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))

        self.transeferblock = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     num_init_features * 3,
                     num_init_features * 2,
                     kernel_size=1,
                     stride=1,
                     padding=0,
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features * 2)),
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1',
                 nn.Conv3d(
                     num_init_features * 2,
                     num_init_features,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     bias=False)),
            ]))

        num_features = num_init_features

        self.features_end = nn.Sequential(
            OrderedDict([
                ('conv0', nn.Conv3d(num_features*4, num_features* 3, kernel_size=3, stride=1, padding=1, bias=False)),
                # 352 -- 176
                ('norm0', nn.BatchNorm3d(num_features* 3)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2)),
                ('conv1', nn.Conv3d(num_init_features * 3, num_init_features*2, kernel_size=5, stride=1, padding=2, bias=False)),
                ('norm1', nn.BatchNorm3d(num_features * 2)),
                ('relu1', nn.ReLU(inplace=True)),
                # ('pool1', nn.MaxPool3d(kernel_size=3, stride=2)),
                ('conv2', nn.Conv3d(num_init_features * 2, num_init_features * 1, kernel_size=7, stride=1, padding=3,
                                    bias=False)),
                ('norm2', nn.BatchNorm3d(num_features * 1)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool3d(kernel_size=(4, 5, 4), stride=2)),

            ]))

        # Linear layer
        self.classifier1 = nn.Linear(8*10* 8* num_features, 256)   #[1, 128, 8, 10, 8]

        self.dropout = nn.Dropout(0.5)

        self.classifier2 = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        features_mri_1 = self.features0(x)
        features_mri_2 = self.features1(x)
        features_mri_3 = self.features2(x)

        mri1 = torch.cat([features_mri_1, features_mri_2, features_mri_3], 1)
        mri_2 = self.transeferblock(mri1)
        mri_3 = torch.cat([mri1, mri_2], 1)

        output = self.features_end(mri_3)
        output = output.view(output.size(0), -1)
        output = self.classifier1(output)
        output = self.dropout(output)
        output = self.classifier2(output)

        return output