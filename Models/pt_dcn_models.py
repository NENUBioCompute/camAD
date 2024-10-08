import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math



class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm_1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu_1', nn.ReLU(inplace=True))
        self.add_module('conv_1',
                        nn.Conv3d(
                            num_input_features,
                            bn_size * growth_rate,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('norm_2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu_2', nn.ReLU(inplace=True))
        self.add_module('conv_2',
                        nn.Conv3d(
                            bn_size * growth_rate,
                            growth_rate,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False))
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout3d(self.drop_rate)

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = self.dropout(new_features)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv',
                        nn.Conv3d(
                            num_input_features,
                            num_output_features,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class _Attention(nn.Sequential):
    def __init__(self, num_input_features):
        super(_Attention, self).__init__()

        self.add_module('norm_1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu_1', nn.ReLU(inplace=True))
        self.add_module('conv_1',
                        nn.Conv3d(
                            num_input_features,
                            num_input_features,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('norm_2', nn.BatchNorm3d(num_input_features))
        self.add_module('relu_2', nn.ReLU(inplace=True))
        self.add_module('conv_2',
                        nn.Conv3d(
                            num_input_features,
                            num_input_features,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False))
    def forward(self, x):
        features = super(_Attention, self).forward(x)
        output = nn.Sigmoid()(features)
        return output

class DenseNet(nn.Module):
    def __init__(self,
                 #sample_size,
                 #sample_duration,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000):

        super(DenseNet, self).__init__()


        # First convolution
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     1,
                     num_init_features,
                     kernel_size=3,     #7
                     stride=(1, 1, 1),  #2
                     padding=(1, 1, 1), #3
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))
       
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):                    
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.features.add_module('pool', nn.MaxPool3d(kernel_size=(4, 5, 4), stride=2))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(1*1*1*num_features, num_classes)  #num_features
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out

class PT_DCN(nn.Module):
    
    def __init__(self):
        super(PT_DCN, self).__init__()
        growth_rate=16
        num_layers = 3 
        num_init_features=16
        bn_size=4
        drop_rate=0
        num_classes=2

        # First convolution
        self.features0 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     1,
                     num_init_features,
                     kernel_size=3,     #7
                     stride=(1, 1, 1),  #2
                     padding=(1, 1, 1), #3
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))

        # pathwise transferblock1
        self.transeferblock1 = nn.Sequential(
            OrderedDict([            
                ('conv0',
                 nn.Conv3d(
                     num_init_features * 2,
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

        # denseblock 1 
        num_features = num_init_features * 2    # 32
        block1 = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)
        self.denseblock1 = nn.Sequential(
                                OrderedDict([
                                                ('denseblock1', block1)
                                            ]))         
        num_features = num_features + num_layers * growth_rate  # 32 + 3 * 16 = 80
        trans1 = _Transition(
            num_input_features=num_features,
            num_output_features=num_features // 2)
        self.denseblock1.add_module('transition1', trans1)
        num_features = num_features // 2  # 40

        # pathwise transferblock2
        self.transeferblock2 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     num_features * 2,  # 80
                     num_features * 2,  # 80
                     kernel_size=1,     
                     stride=1,  
                     padding=0,
                     bias=False)),  # 4, 80, 19, 23, 19
                ('norm0', nn.BatchNorm3d(num_features * 2)), # 80
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1',
                 nn.Conv3d(
                     num_features * 2,   # 80
                     num_features,   # 40
                     kernel_size=3,     
                     stride=1,  
                     padding=1,
                     bias=False)), # 4, 40, 19, 23, 19
            ])) 
 
        # denseblock 2 
        num_features = num_features * 2    # 80
        block2 = _DenseBlock(
            num_layers=num_layers, # 3
            num_input_features=num_features,   # 80
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)
        self.denseblock2 = nn.Sequential(
                                OrderedDict([
                                                ('denseblock2', block2)
                                            ]))         
        num_features = num_features + num_layers * growth_rate  # 80 + 3 * 16 = 128
        trans2 = _Transition(
            num_input_features=num_features,  # 128
            num_output_features=num_features // 2)   # 64
        self.denseblock2.add_module('transition2', trans2)
        num_features = num_features // 2  # 64
        
        # pathwise transferblock3
        self.transeferblock3 = nn.Sequential(
            OrderedDict([            
                ('conv0',
                 nn.Conv3d(
                     num_features * 2,  # 128
                     num_features * 2,  # 128
                     kernel_size=1,     
                     stride=1,  
                     padding=0,
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_features * 2)),
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1',
                 nn.Conv3d(
                     num_features * 2,  # 128
                     num_features,  # 64
                     kernel_size=3,     
                     stride=1,  
                     padding=1,
                     bias=False)),
            ])) 

        # denseblock 3 
        num_features = num_features * 2      # 128
        block3 = _DenseBlock(
            num_layers=num_layers,  # 3
            num_input_features=num_features,   # 128
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)
        self.denseblock3 = nn.Sequential(
                                OrderedDict([
                                                ('denseblock3', block3)
                                            ]))         
        num_features = num_features + num_layers * growth_rate   # 128 + 3 * 16 = 176

        self.features_end = nn.Sequential(
                                OrderedDict([
                                                ('conv0', nn.Conv3d(num_features * 2, num_features, kernel_size=3, stride=2, padding=0, bias=False)),   # 352 -- 176
                                                ('norm1', nn.BatchNorm3d(num_features)),
                                                ('relu1', nn.ReLU(inplace=True)),
                                                ('pool1', nn.MaxPool3d(kernel_size=(4, 5, 4), stride=2))
                                            ])) 



        # Linear layer
        self.classifier = nn.Linear(1*1*1*num_features, num_classes) 

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self, x, y):
        #1
        # print("x:{}, y:{}".format(x.shape, y.shape))
        features_mri = self.features0(x)
        features_pet = self.features0(y)
        # print("features_mri: {}, features_pet: {}".format(features_mri.shape, features_pet.shape))

        concat1 = torch.cat([features_mri, features_pet], 1)
        output1 = self.transeferblock1(concat1)
        # print("concat1: {}, output1: {}".format(concat1.shape, output1.shape))

        mri1 = torch.cat([features_mri, output1], 1)
        pet1 = torch.cat([features_pet, output1], 1)
        # print("mri1: {}, pet1: {}".format(mri1.shape, pet1.shape))

        output1_mri = self.denseblock1(mri1)
        output1_pet = self.denseblock1(pet1)
        # print("output1_mri: {}, output1_pet: {}".format(output1_mri.shape, output1_pet.shape))

        #2
        concat2 = torch.cat([output1_mri, output1_pet], 1)
        output2 = self.transeferblock2(concat2)
        # print("concat2: {}, output2: {}".format(concat2.shape, output2.shape))

        mri2 = torch.cat([output1_mri, output2], 1)
        pet2 = torch.cat([output1_pet, output2], 1)
        # print("mri2: {}, pet2: {}".format(mri2.shape, pet2.shape))

        output2_mri = self.denseblock2(mri2)
        output2_pet = self.denseblock2(pet2)
        # print("output2_mri: {}, output2_pet: {}".format(output2_mri.shape, output2_pet.shape))

        #3 
        concat3 = torch.cat([output2_mri, output2_pet], 1)
        output3 = self.transeferblock3(concat3)
        # print("concat3: {}, output3: {}".format(concat3.shape, output3.shape))

        mri3 = torch.cat([output2_mri, output3], 1)
        pet3 = torch.cat([output2_pet, output3], 1)
        # print("mri3: {}, pet3: {}".format(mri3.shape, pet3.shape))

        output3_mri = self.denseblock3(mri3)
        output3_pet = self.denseblock3(pet3)
        # print("output3_mri: {}, output3_pet: {}".format(output3_mri.shape, output3_pet.shape))

        #4
        concat4 = torch.cat([output3_mri, output3_pet], 1)
        output = self.features_end(concat4)
        # print("concat4: {}, output: {}".format(concat4.shape, output.shape))

        output = output.view(output.size(0), -1)

        # print("output_: {}".format(output.shape))
        output = self.classifier(output)
        # print("output: {}".format(output.shape))

        return output


class PT_DCN_no_denseblock(nn.Module):

    def __init__(self):
        super(PT_DCN_no_denseblock, self).__init__()
        growth_rate = 16
        num_layers = 3
        num_init_features = 16
        bn_size = 4
        drop_rate = 0
        num_classes = 2

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

        # pathwise transferblock1
        self.transeferblock1 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     num_init_features * 2,
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

        # denseblock 1
        num_features = num_init_features * 2  # 32
        # block1 = _DenseBlock(
        #     num_layers=num_layers,
        #     num_input_features=num_features,
        #     bn_size=bn_size,
        #     growth_rate=growth_rate,
        #     drop_rate=drop_rate)
        # self.denseblock1 = nn.Sequential(
        #     OrderedDict([
        #         ('denseblock1', block1)
        #     ]))
        # num_features = num_features + num_layers * growth_rate  # 32 + 3 * 16 = 80
        # trans1 = _Transition(
        #     num_input_features=num_features,
        #     num_output_features=num_features // 2)
        # self.denseblock1.add_module('transition1', trans1)
        # num_features = num_features // 2  # 40

        # pathwise transferblock2
        self.transeferblock2 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     num_features * 2,  # 80
                     num_features * 2,  # 80
                     kernel_size=1,
                     stride=1,
                     padding=0,
                     bias=False)),  # 4, 80, 19, 23, 19
                ('norm0', nn.BatchNorm3d(num_features * 2)),  # 80
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1',
                 nn.Conv3d(
                     num_features * 2,  # 80
                     num_features,  # 40
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     bias=False)),  # 4, 40, 19, 23, 19
            ]))

        # denseblock 2
        num_features = num_features * 2  # 80
        # block2 = _DenseBlock(
        #     num_layers=num_layers,  # 3
        #     num_input_features=num_features,  # 80
        #     bn_size=bn_size,
        #     growth_rate=growth_rate,
        #     drop_rate=drop_rate)
        # self.denseblock2 = nn.Sequential(
        #     OrderedDict([
        #         ('denseblock2', block2)
        #     ]))
        # num_features = num_features + num_layers * growth_rate  # 80 + 3 * 16 = 128
        # trans2 = _Transition(
        #     num_input_features=num_features,  # 128
        #     num_output_features=num_features // 2)  # 64
        # self.denseblock2.add_module('transition2', trans2)
        # num_features = num_features // 2  # 64

        # pathwise transferblock3
        self.transeferblock3 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     num_features * 2,  # 128
                     num_features * 2,  # 128
                     kernel_size=1,
                     stride=1,
                     padding=0,
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_features * 2)),
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1',
                 nn.Conv3d(
                     num_features * 2,  # 128
                     num_features,  # 64
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     bias=False)),
            ]))

        # denseblock 3
        num_features = num_features * 2  # 128
        # block3 = _DenseBlock(
        #     num_layers=num_layers,  # 3
        #     num_input_features=num_features,  # 128
        #     bn_size=bn_size,
        #     growth_rate=growth_rate,
        #     drop_rate=drop_rate)
        # self.denseblock3 = nn.Sequential(
        #     OrderedDict([
        #         ('denseblock3', block3)
        #     ]))
        # num_features = num_features + num_layers * growth_rate  # 128 + 3 * 16 = 176

        self.features_end = nn.Sequential(
            OrderedDict([
                ('conv0', nn.Conv3d(num_features * 2, num_features, kernel_size=3, stride=2, padding=0, bias=False)),
                # 352 -- 176
                ('norm1', nn.BatchNorm3d(num_features)),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool3d(kernel_size=(8, 10, 8), stride=2))
            ]))

        # Linear layer
        self.classifier = nn.Linear(1* 1 * 1 * num_features, num_classes)   #[1, 128, 8, 10, 8]

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        # 1
        # print("x:{}, y:{}".format(x.shape, y.shape))
        features_mri = self.features0(x)
        features_pet = self.features0(y)
        # print("features_mri: {}, features_pet: {}".format(features_mri.shape, features_pet.shape))

        concat1 = torch.cat([features_mri, features_pet], 1)
        output1 = self.transeferblock1(concat1)
        # print("concat1: {}, output1: {}".format(concat1.shape, output1.shape))

        mri1 = torch.cat([features_mri, output1], 1)
        pet1 = torch.cat([features_pet, output1], 1)
        # print("mri1: {}, pet1: {}".format(mri1.shape, pet1.shape))

        # output1_mri = self.denseblock1(mri1)
        # output1_pet = self.denseblock1(pet1)
        # print("output1_mri: {}, output1_pet: {}".format(output1_mri.shape, output1_pet.shape))

        # 2
        concat2 = torch.cat([mri1, pet1], 1)
        # concat2 = torch.cat([output1_mri, output1_pet], 1)
        output2 = self.transeferblock2(concat2)
        # print("concat2: {}, output2: {}".format(concat2.shape, output2.shape))

        mri2 = torch.cat([mri1, output2], 1)
        pet2 = torch.cat([pet1, output2], 1)
        # print("mri2: {}, pet2: {}".format(mri2.shape, pet2.shape))

        # output2_mri = self.denseblock2(mri2)
        # output2_pet = self.denseblock2(pet2)
        # print("output2_mri: {}, output2_pet: {}".format(output2_mri.shape, output2_pet.shape))

        # 3
        concat3 = torch.cat([mri2, pet2], 1)
        output3 = self.transeferblock3(concat3)
        # print("concat3: {}, output3: {}".format(concat3.shape, output3.shape))

        mri3 = torch.cat([mri2, output3], 1)
        pet3 = torch.cat([pet2, output3], 1)
        # print("mri3: {}, pet3: {}".format(mri3.shape, pet3.shape))

        # output3_mri = self.denseblock3(mri3)
        # output3_pet = self.denseblock3(pet3)
        # print("output3_mri: {}, output3_pet: {}".format(output3_mri.shape, output3_pet.shape))

        # 4
        concat4 = torch.cat([mri3, pet3], 1)
        # print(concat4.shape)
        output = self.features_end(concat4)
        # print("concat4: {}, output: {}".format(concat4.shape, output.shape))
        # print(output.shape)
        output = output.view(output.size(0), -1)

        # print("output_: {}".format(output.shape))
        output = self.classifier(output)
        # print("output: {}".format(output.shape))

        return output

class PT_DCN_no_denseblock_features_end(nn.Module):

    def __init__(self):
        super(PT_DCN_no_denseblock_features_end, self).__init__()
        growth_rate = 16
        num_layers = 3
        num_init_features = 16
        bn_size = 4
        drop_rate = 0
        num_classes = 2

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

        # pathwise transferblock1
        self.transeferblock1 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     num_init_features * 2,
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

        # denseblock 1
        num_features = num_init_features * 2  # 32
        # block1 = _DenseBlock(
        #     num_layers=num_layers,
        #     num_input_features=num_features,
        #     bn_size=bn_size,
        #     growth_rate=growth_rate,
        #     drop_rate=drop_rate)
        # self.denseblock1 = nn.Sequential(
        #     OrderedDict([
        #         ('denseblock1', block1)
        #     ]))
        # num_features = num_features + num_layers * growth_rate  # 32 + 3 * 16 = 80
        # trans1 = _Transition(
        #     num_input_features=num_features,
        #     num_output_features=num_features // 2)
        # self.denseblock1.add_module('transition1', trans1)
        # num_features = num_features // 2  # 40

        # pathwise transferblock2
        self.transeferblock2 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     num_features * 2,  # 80
                     num_features * 2,  # 80
                     kernel_size=1,
                     stride=1,
                     padding=0,
                     bias=False)),  # 4, 80, 19, 23, 19
                ('norm0', nn.BatchNorm3d(num_features * 2)),  # 80
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1',
                 nn.Conv3d(
                     num_features * 2,  # 80
                     num_features,  # 40
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     bias=False)),  # 4, 40, 19, 23, 19
            ]))

        # denseblock 2
        num_features = num_features * 2  # 80
        # block2 = _DenseBlock(
        #     num_layers=num_layers,  # 3
        #     num_input_features=num_features,  # 80
        #     bn_size=bn_size,
        #     growth_rate=growth_rate,
        #     drop_rate=drop_rate)
        # self.denseblock2 = nn.Sequential(
        #     OrderedDict([
        #         ('denseblock2', block2)
        #     ]))
        # num_features = num_features + num_layers * growth_rate  # 80 + 3 * 16 = 128
        # trans2 = _Transition(
        #     num_input_features=num_features,  # 128
        #     num_output_features=num_features // 2)  # 64
        # self.denseblock2.add_module('transition2', trans2)
        # num_features = num_features // 2  # 64

        # pathwise transferblock3
        self.transeferblock3 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     num_features * 2,  # 128
                     num_features * 2,  # 128
                     kernel_size=1,
                     stride=1,
                     padding=0,
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_features * 2)),
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1',
                 nn.Conv3d(
                     num_features * 2,  # 128
                     num_features,  # 64
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     bias=False)),
            ]))

        # denseblock 3
        num_features = num_features * 4  # 128
        # block3 = _DenseBlock(
        #     num_layers=num_layers,  # 3
        #     num_input_features=num_features,  # 128
        #     bn_size=bn_size,
        #     growth_rate=growth_rate,
        #     drop_rate=drop_rate)
        # self.denseblock3 = nn.Sequential(
        #     OrderedDict([
        #         ('denseblock3', block3)
        #     ]))
        # num_features = num_features + num_layers * growth_rate  # 128 + 3 * 16 = 176

        # self.features_end = nn.Sequential(
        #     OrderedDict([
        #         ('conv0', nn.Conv3d(num_features * 2, num_features, kernel_size=3, stride=2, padding=0, bias=False)),
        #         # 352 -- 176
        #         ('norm1', nn.BatchNorm3d(num_features)),
        #         ('relu1', nn.ReLU(inplace=True)),
        #         ('pool1', nn.MaxPool3d(kernel_size=(8, 10, 8), stride=2))
        #     ]))

        # Linear layer
        self.classifier = nn.Linear(38* 47* 38 * num_features, num_classes)   #[1, 128, 8, 10, 8]

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        # 1
        # print("x:{}, y:{}".format(x.shape, y.shape))
        features_mri = self.features0(x)
        features_pet = self.features0(y)
        # print("features_mri: {}, features_pet: {}".format(features_mri.shape, features_pet.shape))

        concat1 = torch.cat([features_mri, features_pet], 1)
        output1 = self.transeferblock1(concat1)
        # print("concat1: {}, output1: {}".format(concat1.shape, output1.shape))

        mri1 = torch.cat([features_mri, output1], 1)
        pet1 = torch.cat([features_pet, output1], 1)
        # print("mri1: {}, pet1: {}".format(mri1.shape, pet1.shape))

        # output1_mri = self.denseblock1(mri1)
        # output1_pet = self.denseblock1(pet1)
        # print("output1_mri: {}, output1_pet: {}".format(output1_mri.shape, output1_pet.shape))

        # 2
        concat2 = torch.cat([mri1, pet1], 1)
        # concat2 = torch.cat([output1_mri, output1_pet], 1)
        output2 = self.transeferblock2(concat2)
        # print("concat2: {}, output2: {}".format(concat2.shape, output2.shape))

        mri2 = torch.cat([mri1, output2], 1)
        pet2 = torch.cat([pet1, output2], 1)
        # print("mri2: {}, pet2: {}".format(mri2.shape, pet2.shape))

        # output2_mri = self.denseblock2(mri2)
        # output2_pet = self.denseblock2(pet2)
        # print("output2_mri: {}, output2_pet: {}".format(output2_mri.shape, output2_pet.shape))

        # 3
        concat3 = torch.cat([mri2, pet2], 1)
        output3 = self.transeferblock3(concat3)
        # print("concat3: {}, output3: {}".format(concat3.shape, output3.shape))

        mri3 = torch.cat([mri2, output3], 1)
        pet3 = torch.cat([pet2, output3], 1)
        # print("mri3: {}, pet3: {}".format(mri3.shape, pet3.shape))

        # output3_mri = self.denseblock3(mri3)
        # output3_pet = self.denseblock3(pet3)
        # print("output3_mri: {}, output3_pet: {}".format(output3_mri.shape, output3_pet.shape))

        # 4
        concat4 = torch.cat([mri3, pet3], 1)
        # print(concat4.shape)
        # output = self.features_end(concat4)
        # print("concat4: {}, output: {}".format(concat4.shape, output.shape))
        # print(output.shape)
        # print(concat4.shape)
        output = concat4.view(concat4.size(0), -1)

        # print("output_: {}".format(output.shape))
        output = self.classifier(output)
        # print("output: {}".format(output.shape))

        return output


class PT_DCN_no_denseblock_transeferblock(nn.Module):

    def __init__(self):
        super(PT_DCN_no_denseblock_transeferblock, self).__init__()
        growth_rate = 16
        num_layers = 3
        num_init_features = 16
        bn_size = 4
        drop_rate = 0
        num_classes = 2

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

        # pathwise transferblock1
        # self.transeferblock1 = nn.Sequential(
        #     OrderedDict([
        #         ('conv0',
        #          nn.Conv3d(
        #              num_init_features * 2,
        #              num_init_features * 2,
        #              kernel_size=1,
        #              stride=1,
        #              padding=0,
        #              bias=False)),
        #         ('norm0', nn.BatchNorm3d(num_init_features * 2)),
        #         ('relu0', nn.ReLU(inplace=True)),
        #         ('conv1',
        #          nn.Conv3d(
        #              num_init_features * 2,
        #              num_init_features,
        #              kernel_size=3,
        #              stride=1,
        #              padding=1,
        #              bias=False)),
        #     ]))

        # denseblock 1
        num_features = num_init_features * 2  # 32
        # block1 = _DenseBlock(
        #     num_layers=num_layers,
        #     num_input_features=num_features,
        #     bn_size=bn_size,
        #     growth_rate=growth_rate,
        #     drop_rate=drop_rate)
        # self.denseblock1 = nn.Sequential(
        #     OrderedDict([
        #         ('denseblock1', block1)
        #     ]))
        # num_features = num_features + num_layers * growth_rate  # 32 + 3 * 16 = 80
        # trans1 = _Transition(
        #     num_input_features=num_features,
        #     num_output_features=num_features // 2)
        # self.denseblock1.add_module('transition1', trans1)
        # num_features = num_features // 2  # 40

        # pathwise transferblock2
        # self.transeferblock2 = nn.Sequential(
        #     OrderedDict([
        #         ('conv0',
        #          nn.Conv3d(
        #              num_features * 2,  # 80
        #              num_features * 2,  # 80
        #              kernel_size=1,
        #              stride=1,
        #              padding=0,
        #              bias=False)),  # 4, 80, 19, 23, 19
        #         ('norm0', nn.BatchNorm3d(num_features * 2)),  # 80
        #         ('relu0', nn.ReLU(inplace=True)),
        #         ('conv1',
        #          nn.Conv3d(
        #              num_features * 2,  # 80
        #              num_features,  # 40
        #              kernel_size=3,
        #              stride=1,
        #              padding=1,
        #              bias=False)),  # 4, 40, 19, 23, 19
        #     ]))

        # denseblock 2
        num_features = num_features * 2  # 80
        # block2 = _DenseBlock(
        #     num_layers=num_layers,  # 3
        #     num_input_features=num_features,  # 80
        #     bn_size=bn_size,
        #     growth_rate=growth_rate,
        #     drop_rate=drop_rate)
        # self.denseblock2 = nn.Sequential(
        #     OrderedDict([
        #         ('denseblock2', block2)
        #     ]))
        # num_features = num_features + num_layers * growth_rate  # 80 + 3 * 16 = 128
        # trans2 = _Transition(
        #     num_input_features=num_features,  # 128
        #     num_output_features=num_features // 2)  # 64
        # self.denseblock2.add_module('transition2', trans2)
        # num_features = num_features // 2  # 64

        # pathwise transferblock3
        # self.transeferblock3 = nn.Sequential(
        #     OrderedDict([
        #         ('conv0',
        #          nn.Conv3d(
        #              num_features * 2,  # 128
        #              num_features * 2,  # 128
        #              kernel_size=1,
        #              stride=1,
        #              padding=0,
        #              bias=False)),
        #         ('norm0', nn.BatchNorm3d(num_features * 2)),
        #         ('relu0', nn.ReLU(inplace=True)),
        #         ('conv1',
        #          nn.Conv3d(
        #              num_features * 2,  # 128
        #              num_features,  # 64
        #              kernel_size=3,
        #              stride=1,
        #              padding=1,
        #              bias=False)),
        #     ]))

        # denseblock 3
        num_features = num_features * 4  # 128
        # block3 = _DenseBlock(
        #     num_layers=num_layers,  # 3
        #     num_input_features=num_features,  # 128
        #     bn_size=bn_size,
        #     growth_rate=growth_rate,
        #     drop_rate=drop_rate)
        # self.denseblock3 = nn.Sequential(
        #     OrderedDict([
        #         ('denseblock3', block3)
        #     ]))
        # num_features = num_features + num_layers * growth_rate  # 128 + 3 * 16 = 176


        self.features_end = nn.Sequential(
            OrderedDict([
                ('conv0', nn.Conv3d(864, num_features, kernel_size=3, stride=2, padding=0, bias=False)),
                # 352 -- 176
                ('norm1', nn.BatchNorm3d(num_features)),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool3d(kernel_size=(8, 10, 8), stride=2))
            ]))
        # Linear layer
        self.classifier = nn.Linear(6* 7* 6 * num_features, num_classes)   #[1, 128, 8, 10, 8]

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        # 1
        # print("x:{}, y:{}".format(x.shape, y.shape))
        features_mri = self.features0(x)
        features_pet = self.features0(y)
        # print("features_mri: {}, features_pet: {}".format(features_mri.shape, features_pet.shape))

        output1 = torch.cat([features_mri, features_pet], 1)
        # output1 = self.transeferblock1(concat1)
        # print("concat1: {}, output1: {}".format(concat1.shape, output1.shape))

        mri1 = torch.cat([features_mri, output1], 1)
        pet1 = torch.cat([features_pet, output1], 1)
        # print("mri1: {}, pet1: {}".format(mri1.shape, pet1.shape))

        # output1_mri = self.denseblock1(mri1)
        # output1_pet = self.denseblock1(pet1)
        # print("output1_mri: {}, output1_pet: {}".format(output1_mri.shape, output1_pet.shape))

        # 2
        output2 = torch.cat([mri1, pet1], 1)
        # concat2 = torch.cat([output1_mri, output1_pet], 1)
        # output2 = self.transeferblock2(concat2)
        # print("concat2: {}, output2: {}".format(concat2.shape, output2.shape))

        mri2 = torch.cat([mri1, output2], 1)
        pet2 = torch.cat([pet1, output2], 1)
        # print("mri2: {}, pet2: {}".format(mri2.shape, pet2.shape))

        # output2_mri = self.denseblock2(mri2)
        # output2_pet = self.denseblock2(pet2)
        # print("output2_mri: {}, output2_pet: {}".format(output2_mri.shape, output2_pet.shape))

        # 3
        output3 = torch.cat([mri2, pet2], 1)
        # output3 = self.transeferblock3(concat3)
        # print("concat3: {}, output3: {}".format(concat3.shape, output3.shape))

        mri3 = torch.cat([mri2, output3], 1)
        pet3 = torch.cat([pet2, output3], 1)
        # print("mri3: {}, pet3: {}".format(mri3.shape, pet3.shape))

        # output3_mri = self.denseblock3(mri3)
        # output3_pet = self.denseblock3(pet3)
        # print("output3_mri: {}, output3_pet: {}".format(output3_mri.shape, output3_pet.shape))

        # 4
        concat4 = torch.cat([mri3, pet3], 1)
        # print(concat4.shape)
        output = self.features_end(concat4)
        # print("concat4: {}, output: {}".format(concat4.shape, output.shape))
        # print(output.shape)
        # print(concat4.shape)
        output = output.view(output.size(0), -1)

        # print("output_: {}".format(output.shape))
        output = self.classifier(output)
        # print("output: {}".format(output.shape))

        return output

class PT_DCN_no_denseblock_transeferblock_1(nn.Module):

    def __init__(self):
        super(PT_DCN_no_denseblock_transeferblock_1, self).__init__()
        growth_rate = 16
        num_layers = 3
        num_init_features = 16
        bn_size = 4
        drop_rate = 0
        num_classes = 2

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

        num_features = num_init_features *2

        self.features_end = nn.Sequential(
            OrderedDict([
                ('conv0', nn.Conv3d(num_features*3, num_features, kernel_size=3, stride=2, padding=0, bias=False)),
                # 352 -- 176
                ('norm1', nn.BatchNorm3d(num_features)),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool3d(kernel_size=(8, 10, 8), stride=2))
            ]))
        # Linear layer
        self.classifier = nn.Linear(6* 7* 6 * num_features, num_classes)   #[1, 128, 8, 10, 8]

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        # 1
        # print("x:{}, y:{}".format(x.shape, y.shape))
        features_mri = self.features0(x)
        features_pet = self.features0(y)
        # print("features_mri: {}, features_pet: {}".format(features_mri.shape, features_pet.shape))

        output1 = torch.cat([features_mri, features_pet], 1)
        # output1 = self.transeferblock1(concat1)
        # print("concat1: {}, output1: {}".format(concat1.shape, output1.shape))

        mri1 = torch.cat([features_mri, output1], 1)
        pet1 = torch.cat([features_pet, output1], 1)
        # print("mri1: {}, pet1: {}".format(mri1.shape, pet1.shape))

        # output1_mri = self.denseblock1(mri1)
        # output1_pet = self.denseblock1(pet1)
        # print("output1_mri: {}, output1_pet: {}".format(output1_mri.shape, output1_pet.shape))

        # 2
        output2 = torch.cat([mri1, pet1], 1)

        output = self.features_end(output2)
        # print("concat4: {}, output: {}".format(concat4.shape, output.shape))
        # print(output.shape)
        # print(concat4.shape)
        output = output.view(output.size(0), -1)

        # print("output_: {}".format(output.shape))
        output = self.classifier(output)
        # print("output: {}".format(output.shape))

        return output

class PT_DCN_no_denseblock_transeferblock_2(nn.Module):

    def __init__(self):
        super(PT_DCN_no_denseblock_transeferblock_2, self).__init__()
        growth_rate = 16
        num_layers = 3
        num_init_features = 16
        bn_size = 4
        drop_rate = 0
        num_classes = 2

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

        num_features = num_init_features

        self.features_end = nn.Sequential(
            OrderedDict([
                ('conv0', nn.Conv3d(num_features*2, num_features, kernel_size=3, stride=2, padding=0, bias=False)),
                # 352 -- 176
                ('norm1', nn.BatchNorm3d(num_features)),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool3d(kernel_size=(8, 10, 8), stride=2))
            ]))
        # Linear layer
        self.classifier = nn.Linear(6* 7* 6 * num_features, num_classes)   #[1, 128, 8, 10, 8]

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        # 1
        # print("x:{}, y:{}".format(x.shape, y.shape))
        features_mri = self.features0(x)
        features_pet = self.features0(y)
        # print("features_mri: {}, features_pet: {}".format(features_mri.shape, features_pet.shape))

        output1 = torch.cat([features_mri, features_pet], 1)
        # output1 = self.transeferblock1(concat1)
        # print("concat1: {}, output1: {}".format(concat1.shape, output1.shape))

        output = self.features_end(output1)
        # print("concat4: {}, output: {}".format(concat4.shape, output.shape))
        # print(output.shape)
        # print(concat4.shape)
        output = output.view(output.size(0), -1)

        # print("output_: {}".format(output.shape))
        output = self.classifier(output)
        # print("output: {}".format(output.shape))

        return output
