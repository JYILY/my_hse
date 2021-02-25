import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import copy
import math


class ModifiedResNet(torchvision.models.resnet.ResNet):
    def __init__(self):
        # ResNet50
        super(ModifiedResNet, self).__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class EmbedGuiding(nn.Module):
    def __init__(self, num_classes, init=False):
        super(EmbedGuiding, self).__init__()

        self.fc = nn.Linear(num_classes, 1024)
        self.conv1024 = nn.Conv2d(in_channels=2048 + 1024, out_channels=1024, kernel_size=1)
        self.conv2048 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)

        if init:
            self._init_weights()

    def forward(self, s, f):
        b, c, w, h = f.shape

        s = self.fc(s)
        s = s.repeat(1, w * h)
        s = s.view(b, w, h, -1)
        s = s.permute(0, 3, 2, 1)

        sf = torch.cat((s, f), 1)
        sf = self.conv1024(sf)
        sf = self.tanh(sf)
        sf = self.conv2048(sf)

        b, c, w, h = sf.shape
        sf = sf.view(b * c, w * h)
        sf = self.softmax(sf)
        prior_weights = sf.view(b, c, w, h)

        embed_feature = torch.mul(f, prior_weights)
        embed_feature = self.relu(embed_feature)

        return embed_feature

    def _init_weights(self):
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()

        for m in [self.conv1024, self.conv2048]:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()


class ResNetEmbed(nn.Module):
    def __init__(self, levelTupleList: [], pretrained=True):
        super(ResNetEmbed, self).__init__()

        self.levelTupleList = levelTupleList

        '''
            levelTupleList is the list of levels have the format as [(level_name,num),...]
            and the level goes up from left to right
            
            For example:
            levelTupleList = [("order", 10), ("family", 25), ("genus", 100), ("class", 200)]
        '''
        assert len(levelTupleList) != 0, "the length of levelTupleList could not be 0"

        # Modify from: self.avgpool = nn.AvgPool2d(14, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

        # make trunk
        self.trunk = ModifiedResNet()
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(torchvision.models.resnet.model_urls['resnet50'])
            self.trunk.load_state_dict(state_dict)

        # make branch
        self.l1_layer4 = copy.deepcopy(self.trunk.layer4)
        self.l1_fc = nn.Linear(2048, levelTupleList[0][1])

        self.num_level = len(levelTupleList)

        if self.num_level > 1:
            self.layers = []
            for idx in range(1, self.num_level):
                self.layers.append(copy.deepcopy(self.trunk.layer4))  # layer4_guide
                self.layers.append(copy.deepcopy(self.trunk.layer4))  # layer4_raw
                self.layers.append(nn.Linear(2048, levelTupleList[idx][1]))  # fc_guide
                self.layers.append(nn.Linear(2048, levelTupleList[idx][1]))  # fc_raw
                self.layers.append(nn.Linear(4096, levelTupleList[idx][1]))  # fc_cat
                self.layers.append(EmbedGuiding(levelTupleList[idx - 1][1]))  # guide
            self.branchs = nn.Sequential(*self.layers)


    def forward(self, x):
        bs = x.shape[0]
        result = []

        f_share = self.trunk(x)

        for i in range(self.num_level):
            if i == 0:
                f = self.l1_layer4(f_share)
                f = self.avgpool(f)
                f = f.view(bs, -1)
                s = self.l1_fc(f)
                result.append(s)
            else:
                f_r = self.layers[(i - 1) * 6 + 1](f_share)  # layer4_raw
                f_r = self.avgpool(f_r)
                f_r = f_r.view(bs, -1)
                s_r = self.layers[(i - 1) * 6 + 3](f_r)  # fc_raw

                last_s = Variable(result[i - 1].data.clone(), requires_grad=False)
                last_s = self.softmax(last_s)
                f_g = self.layers[(i - 1) * 6](f_share)  # layer4_guide
                f_g = self.layers[(i - 1) * 6 + 5](last_s, f_g)  # guide
                f_g = torch.sum(f_g.view(f_g.size(0), f_g.size(1), -1), dim=2)
                f_g = f_g.view(bs, -1)
                s_g = self.layers[(i - 1) * 6 + 2](f_g)  # fc_guide

                f_c = torch.cat((f_r, f_g), dim=1)
                s_c = self.layers[(i - 1) * 6 + 4](f_c)  # fc_cat
                s_avg = (s_r + s_g + s_c) / 3

                result.append(s_avg)

        return result
