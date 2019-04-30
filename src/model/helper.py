import torch.nn as nn
import math


def init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.linear):
            m.weight.data.normal_(0,0.01)
            m.bias.data.zero_()


def make_standard_block(feat_in, feat_out, kernel, stride=1, padding=1, use_bn=True):
    layers = []
    layers += [nn.Conv2d(feat_in, feat_out, kernel, stride, padding)]
    if use_bn:
        layers += [nn.BatchNorm2d(feat_out, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)]
    layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)
