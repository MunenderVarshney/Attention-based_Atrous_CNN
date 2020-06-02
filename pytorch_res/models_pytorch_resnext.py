import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")
    if cuda:
        x = x.cuda()
    x = Variable(x)
    return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. 
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing 
    human-level performance on imagenet classification." Proceedings of the 
    IEEE international conference on computer vision. 2015.
    """
  
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
        
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                                dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.convl = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.bnl = norm_layer(512)
        self.relul = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        (_, seq_len, mel_bins) = x.shape
        x = x.view(-1, 1, seq_len, mel_bins)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #print("Input size at layer 3  ", x.size())
        # x = self.layer4(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        x = self.relul(self.bnl(self.convl(x)))
        #print("Input size at layer 3  ", x.size())
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model





def EmbeddingLayers_pooling(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    # return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
    #                **kwargs)             
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)

# class EmbeddingLayers_pooling(nn.Module):
#     def __init__(self):
#         super(EmbeddingLayers_pooling, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
#                                kernel_size=(5, 5), stride=(1, 1),  dilation=1,
#                                padding=(2, 2), bias=False)

#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
#                                kernel_size=(5, 5), stride=(1, 1),  dilation=2,
#                                padding=(4, 4), bias=False)

#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
#                                kernel_size=(5, 5), stride=(1, 1),  dilation=4,
#                                padding=(8, 8), bias=False)

#         self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
#                                kernel_size=(5, 5), stride=(1, 1),  dilation=8,
#                                padding=(16, 16), bias=False)

#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.bn4 = nn.BatchNorm2d(512)
#         self.init_weights()

#     def init_weights(self):
#         init_layer(self.conv1)
#         init_layer(self.conv2)
#         init_layer(self.conv3)
#         init_layer(self.conv4)
#         init_bn(self.bn1)
#         init_bn(self.bn2)
#         init_bn(self.bn3)
#         init_bn(self.bn4)

#     def forward(self, input, return_layers=False):
#         (_, seq_len, mel_bins) = input.shape
#         x = input.view(-1, 1, seq_len, mel_bins)
#         """(samples_num, feature_maps, time_steps, freq_num)"""
#         print("Input size at conv one ", x.size())
#         x = F.relu(self.bn1(self.conv1(x)))
#         print("Input size at conv two ", x.size())
#         x = F.relu(self.bn2(self.conv2(x)))
#         print("Input size at conv three ", x.size())
#         x = F.relu(self.bn3(self.conv3(x)))
#         print("Input size at conv 4 ", x.size())
#         x = F.relu(self.bn4(self.conv4(x)))
#         print("Input size at output ", x.size())
#         return x

class CnnPooling_Max(nn.Module):
    def __init__(self, classes_num):
        super(CnnPooling_Max, self).__init__()
        self.emb = EmbeddingLayers_pooling()
        self.fc_final = nn.Linear(512, classes_num)
        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""
        x = self.emb(input)
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])
        x = F.log_softmax(self.fc_final(x), dim=-1)
        return x

class CnnPooling_Avg(nn.Module):
    def __init__(self, classes_num):
        super(CnnPooling_Avg, self).__init__()
        self.emb = EmbeddingLayers_pooling()
        self.fc_final = nn.Linear(512, classes_num)
        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""
        x = self.emb(input)
        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])
        output = F.log_softmax(self.fc_final(x), dim=-1)
        return output

class CnnPooling_Attention(nn.Module):
    def __init__(self, classes_num):
        super(CnnPooling_Attention, self).__init__()
        self.emb = EmbeddingLayers_pooling()
        self.attention = Attention2d(
            512,
            classes_num,
            att_activation='sigmoid',
            cla_activation='log_softmax')

    def init_weights(self):
        pass

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""
        x = self.emb(input)
        output = self.attention(x)
        # print("Idggfghrf ", output.size())
        return output


class Attention2d(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(Attention2d, self).__init__()
        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        self.att.weight.data.fill_(0.)

    def activate(self, x, activation):
        if activation == 'linear':
            return x
        elif activation == 'relu':
            return F.relu(x)
        elif activation == 'sigmoid':
            return F.sigmoid(x)+0.1
        elif activation == 'log_softmax':
            return F.log_softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, channel, time_steps, freq_bins)
        """
        att = self.att(x)
        # print("atttttttt ", att.size())
        att = self.activate(att, self.att_activation)
        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)
        # print("claaaaaa ", cla.size())
        # (samples_num, channel, time_steps * freq_bins)
        att = att.view(att.size(0), att.size(1), att.size(2) * att.size(3))
        cla = cla.view(cla.size(0), cla.size(1), cla.size(2) * cla.size(3))
        # print("claaaaaa ", cla.size())
        epsilon = 0.1 # 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)
        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)
        Return_heatmap = False
        if Return_heatmap:
            return x, norm_att
        else:
            return x