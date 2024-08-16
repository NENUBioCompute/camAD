import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class CrossAttention3D(nn.Module):
    def __init__(self, input_channels, num_heads):
        """
        3D 交叉注意力模块。
        参数:
        - input_channels: 输入 3D 图像的通道数。
        - num_heads: 多头注意力的头数。
        """
        super(CrossAttention3D, self).__init__()
        self.num_heads = num_heads
        self.input_channels = input_channels

        # 定义线性层用于生成查询、键和值矩阵
        self.query_linear = nn.Conv3d(input_channels, input_channels, kernel_size=1)
        self.key_linear = nn.Conv3d(input_channels, input_channels, kernel_size=1)
        self.value_linear = nn.Conv3d(input_channels, input_channels, kernel_size=1)

        # 定义输出线性层
        self.output_linear = nn.Conv3d(input_channels, input_channels, kernel_size=1)

    def forward(self, source, target):
        """
        执行 3D 交叉注意力计算。
        参数:
        - source: 源 3D 图像张量，形状为 (batch_size, channels, depth, height, width)。
        - target: 目标 3D 图像张量，形状为 (batch_size, channels, depth, height, width)。

        返回:
        - 加权和的结果，形状为 (batch_size, channels, depth, height, width)。
        """
        # 计算查询、键和值矩阵
        query = self.query_linear(source)  # 形状为 (batch_size, channels, depth, height, width)
        key = self.key_linear(target)  # 形状为 (batch_size, channels, depth, height, width)
        value = self.value_linear(target)  # 形状为 (batch_size, channels, depth, height, width)

        # 将查询、键和值展开为二维张量
        query_reshaped = query.view(query.size(0), query.size(1), -1)  # (batch_size, channels, depth * height * width)
        key_reshaped = key.view(key.size(0), key.size(1), -1)  # (batch_size, channels, depth * height * width)
        value_reshaped = value.view(value.size(0), value.size(1), -1)  # (batch_size, channels, depth * height * width)

        # 计算注意力得分
        scores = torch.matmul(query_reshaped.transpose(1, 2),
                              key_reshaped)  # (batch_size, depth * height * width, depth * height * width)
        scaling_factor = key.size(1) ** 0.5
        scores /= scaling_factor

        # 使用 softmax 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, depth * height * width, depth * height * width)

        # 计算加权和
        weighted_sum_reshaped = torch.matmul(attention_weights, value_reshaped.transpose(1,
                                                                                         2))  # (batch_size, depth * height * width, channels)

        # 将加权和 reshape 回 3D 张量
        weighted_sum = weighted_sum_reshaped.transpose(1, 2).view(source.size())

        # 输出线性层
        output = self.output_linear(weighted_sum)

        return output

class camAD(nn.Module):
    def __init__(self, args):
        super(camAD, self).__init__()
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
                ('norm0', nn.InstanceNorm3d(num_init_features)),
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
                ('norm0', nn.InstanceNorm3d(num_init_features)),
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
                ('norm0', nn.InstanceNorm3d(num_init_features)),
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
                ('norm0', nn.InstanceNorm3d(num_init_features * 2)),
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
                ('norm0', nn.InstanceNorm3d(num_features* 3)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2)),
                ('conv1', nn.Conv3d(num_init_features * 3, num_init_features*2, kernel_size=5, stride=1, padding=2, bias=False)),
                ('norm1', nn.InstanceNorm3d(num_features * 2)),
                ('relu1', nn.ReLU(inplace=True)),
                # ('pool1', nn.MaxPool3d(kernel_size=3, stride=2)),
                ('conv2', nn.Conv3d(num_init_features * 2, num_init_features * 1, kernel_size=7, stride=1, padding=3,
                                    bias=False)),
                ('norm2', nn.InstanceNorm3d(num_features * 1)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool3d(kernel_size=(4, 5, 4), stride=2)),

            ]))

        self.cross_attention = CrossAttention3D(num_features, 4)

        # Linear layer
        self.classifier1 = nn.Linear(8*10* 8* num_features, 256)   #[1, 128, 8, 10, 8]

        self.dropout = nn.Dropout(args.dropout)   #0.5

        self.classifier2 = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, y):
        features_mri_1 = self.features0(x)
        features_pet_1 = self.features0(y)

        features_mri_2 = self.features1(x)
        features_pet_2 = self.features1(y)

        features_mri_3 = self.features2(x)
        features_pet_3 = self.features2(y)

        mri1 = torch.cat([features_mri_1, features_mri_2, features_mri_3], 1)
        pet1 = torch.cat([features_pet_1, features_pet_2, features_pet_3], 1)

        mri_2 = self.transeferblock(mri1)
        pet_2 = self.transeferblock(pet1)

        mri_3 = torch.cat([mri1, mri_2], 1)
        pet_3 = torch.cat([pet1, pet_2], 1)

        mri_4 = self.features_end(mri_3)
        pet_4 = self.features_end(pet_3)

        output = self.cross_attention(mri_4, pet_4)

        output = output.view(output.size(0), -1)

        output = self.classifier1(output)
        output = self.dropout(output)
        output = self.classifier2(output)

        return output