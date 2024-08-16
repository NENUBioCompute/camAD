
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import math
import copy
import numpy as np
import torch.nn.functional as F
from Models.camAD import *
from pt_dcn_models import *
from Models.vision_transformer import *
from Models.transformer import *

class Feature_Combination(nn.Module):
    def __init__(self, args):
        super(Feature_Combination, self).__init__()

        self.args = args
        self.model = self.args.model
        # nonlineraity
        self.relu = nn.ReLU()
        self.act_f = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        self.axial_shape = self.args.axial_shape
        self.sagittal_shape = self.args.sagittal_shape
        self.coronal_shape = self.args.coronal_shape

        # encoding layers
        if self.model == 'camAD':
            self.classifier = camAD(self.args)

            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        elif self.model == 'PT_DCN':
            self.classifier = PT_DCN()

            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        # elif self.model == 'transformer':
        #     self.classifier = Multiview_MEP_Com(self.args)
        #
        # elif self.model == 'VIT':
        #     self.classifier =  VisionTransformer_stage_Com(self.args)
        #

    def forward(self, x, y, al=None):
        if al == None:
            output = self.classifier(x, y)
        else:
            output = self.classifier(x, y, al)

        return output