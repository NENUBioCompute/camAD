import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import math
import copy
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

class ResNet_Concat_b(nn.Module):
    def __init__(self, args):
        super(ResNet_Concat_b, self).__init__()
        self.args = args

        self.relu = nn.ReLU()
        self.act_f = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.mid_dim = args.mid_dim
        self.resnet1 = ResNet(self.args, mid_dim=self.mid_dim)
        # self.resnet2 = ResNet(self.args, mid_dim=self.mid_dim//2)

        self.fc = nn.Linear(self.mid_dim*2, args.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        feature_x = self.resnet1(x)
        feature_y = self.resnet1(y)
        feature_concat = torch.cat([feature_x, feature_y], 1)

        output = nn.AvgPool3d(kernel_size=(feature_concat.size(2), feature_concat.size(3), feature_concat.size(2)), stride=1)(feature_concat)  # [2, 128, 1, 1, 229]

        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

class ResNet_Concat_f(nn.Module):
    def __init__(self, args):
        super(ResNet_Concat_f, self).__init__()
        self.args = args

        self.relu = nn.ReLU()
        self.act_f = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.mid_dim = args.mid_dim
        self.resnet = ResNet(self.args, in_channel=2)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):

        feature_concat = torch.cat([x, y], 1)
        output = self.resnet(feature_concat)

        return output


class ResNet_Mean(nn.Module):
    def __init__(self, args):
        super(ResNet_Mean, self).__init__()
        self.args = args

        self.relu = nn.ReLU()
        self.act_f = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.resnet1 = ResNet(self.args)
        # self.resnet2 = ResNet(self.args, mid_dim=self.mid_dim//2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        feature_x = self.resnet1(x)
        feature_y = self.resnet1(y)
        # print(feature_x.shape, feature_y.shape)
        output = torch.add(feature_x, feature_y)
        # print(output.shape)
        return output

class ResNet_CrossAttenion(nn.Module):
    def __init__(self, args):
        super(ResNet_CrossAttenion, self).__init__()
        self.args = args

        self.relu = nn.ReLU()
        self.act_f = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.mid_dim = args.mid_dim
        self.num_heads = args.crossAtt_num_heads

        self.resnet1 = ResNet(self.args, mid_dim=self.mid_dim)
        self.cross_attention_3d = _CrossAttention3D(self.mid_dim, self.num_heads)
        # self.resnet2 = ResNet(self.args)
        self.fc1 = nn.Linear(27*self.mid_dim, self.mid_dim//2)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(self.mid_dim//2, args.num_classes)  # [1, 128, 8, 10, 8]

    def forward(self, x, y):
        feature_x = self.resnet1(x)
        feature_y = self.resnet1(y)
        feature_com = self.cross_attention_3d(feature_x, feature_y)
        feature_com = feature_com.view(feature_com.size(0), -1)
        output = self.fc1(feature_com)
        output = self.dropout(output)
        output = self.fc2(output)
        # output = self.cross_attention(feature_x, feature_x)
        return output


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3), stride=(stride, stride, stride),
                     padding=(1, 1, 1), bias=False)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1), bias=False)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=(stride, stride, stride), padding=(1, 1, 1), bias=False)
        self.bn2 = nn.InstanceNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * Bottleneck.expansion, kernel_size=(1, 1, 1), bias=False)
        self.bn3 = nn.InstanceNorm3d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, args=None, mid_dim=False, in_channel=1, bottleneck=False):
        super(ResNet, self).__init__()
        self.args = args
        self.mid_dim = mid_dim
        blocks = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
        if self.args:
            self.depth = self.args.resnet_depth
            self.inplanes = self.args.inplanes
            self.num_classes = self.args.num_classes
        else:
            self.depth = 18
            self.inplanes = 64
            self.num_classes = 3
        assert layers[self.depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'
        self.f_out = [self.inplanes, self.inplanes * 2, self.inplanes * 4, self.inplanes * 8]

        self.conv1 = nn.Conv3d(in_channel, self.inplanes, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.InstanceNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.layer1 = self._make_layer(blocks[self.depth], self.f_out[0], layers[self.depth][0])
        self.layer2 = self._make_layer(blocks[self.depth], self.f_out[1], layers[self.depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[self.depth], self.f_out[2], layers[self.depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[self.depth], self.f_out[3], layers[self.depth][3], stride=2)

        class_dim  = self.num_classes if not self.mid_dim else self.mid_dim
        self.fc = nn.Linear(self.inplanes, class_dim)
        # self.classifier2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=(1, 1, 1), stride=(stride, stride, stride), bias=False),
                nn.InstanceNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)

        if not self.mid_dim:
            x = nn.AvgPool3d(kernel_size=(x.size(2), x.size(3), x.size(2)), stride=1)(x)
            x = x.view(x.size(0), -1)
            x = x.permute(1, 0) if x.shape[-1] == 1 else x
            x = self.fc(x)  
        return x
