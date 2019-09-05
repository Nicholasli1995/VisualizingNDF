import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models

class HierRes(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(HierRes, self).__init__()
        if out_channels % 16 != 0:
            raise NotImplementedError
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, int(out_channels/2), kernel_size=1, padding=0, stride=stride)
        self.bn1 = nn.BatchNorm2d(int(out_channels/2))
        self.relu1 = nn.ReLU(inplace=True)      
        self.conv2 = nn.Conv2d(int(out_channels/2), int(out_channels/4), kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(int(out_channels/4))
        self.relu2 = nn.ReLU(inplace=True)       
        self.conv3 = nn.Conv2d(int(out_channels/4), int(out_channels/8), kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(int(out_channels/8))
        self.relu3 = nn.ReLU(inplace=True) 
        self.conv4 = nn.Conv2d(int(out_channels/8), int(out_channels/8), kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(int(out_channels/8))        
        self.relu4 = nn.ReLU(inplace=True) 
        self.in_num = in_channels
        self.out_num = out_channels
        if in_channels != out_channels or stride != 1:
            self.map = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) 
            self.bn_map = nn.BatchNorm2d(out_channels)
            
    def forward(self, x):
        if self.in_num != self.out_num or self.stride != 1:
            origin = self.bn_map(self.map(x))
        else:
            origin = x
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = self.relu2(out2)
        out3 = self.conv3(out2)
        out3 = self.bn3(out3)
        out3 = self.relu3(out3)
        out4 = self.conv4(out3)
        out4 = self.bn4(out4)
        out4 = self.relu4(out4)        
        out = torch.cat((out1, out2, out3, out4), dim=1) + origin
        return out

class Inception(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(Inception, self).__init__()
        if out_channels % 16 != 0:
            raise NotImplementedError
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, int(out_channels/2), kernel_size=1, padding=0, stride=stride)
        self.bn1 = nn.BatchNorm2d(int(out_channels/2))
        self.relu1 = nn.ReLU(inplace=True)      
        self.conv2 = nn.Conv2d(in_channels, int(out_channels/4), kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(int(out_channels/4))
        self.relu2 = nn.ReLU(inplace=True)       
        self.conv3 = nn.Conv2d(in_channels, int(out_channels/4), kernel_size=5, padding=2, stride=stride)
        self.bn3 = nn.BatchNorm2d(int(out_channels/4))
        self.relu3 = nn.ReLU(inplace=True) 
        self.in_num = in_channels
        self.out_num = out_channels
        if in_channels != out_channels or stride != 1:
            self.map = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) 
            self.bn_map = nn.BatchNorm2d(out_channels)
            
    def forward(self, x):
        if self.in_num != self.out_num or self.stride != 1:
            origin = self.bn_map(self.map(x))
        else:
            origin = x
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(x)))
        out3 = F.relu(self.bn3(self.conv3(x)))
        out = torch.cat((out1, out2, out3), dim=1) + origin
        return out
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.mp(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 224 by 224 input, the output size is 7 by 7
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_output):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_output)

def ResNet34(num_output, pretrained = False):
    model = ResNet(Inception, [6,8,12,6], num_classes=num_output)
    # copy the first convolution kernel from a model pre-trained on Imagenet
    if pretrained:
        model.conv1.weight.data = get_first_conv_layer()
    return model

def ResNet50(num_output):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_output)

def ResNet101(num_output):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_output)

def ResNet152(num_output):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_output)

def get_first_conv_layer():
    # get the first convlution layer from resnet18 pre-trained on ImageNet
    temp_model = models.resnet18(pretrained=True)
    weight = temp_model.conv1.weight.data.clone()
    del temp_model
    return weight

def test():
    net = ResNet34(128)
    y = net(torch.randn(1,3,224,224))
    print(y.size())

def normalize(tensor):
    return (tensor/(tensor.max()-tensor.min()) + 1)/2

def visualize_filters(weight):
    assert len(weight.shape) == 4
    assert weight.shape[1] == 3
    col_num = 10
    row_num = int(weight.shape[0]/col_num) + 1
    for filter_idx in range(weight.shape[0]):
        plt.subplot(row_num, col_num, filter_idx+1)
        plt.imshow(normalize(weight[filter_idx,:]).numpy().transpose((1,2,0)))
    
#-----------------------------------------------------------------------------#
class Hybrid(nn.Module):
    def __init__(self, block, num_blocks = [6,8,12,6], replace = [False, False, 
                 False, False], num_classes=10, attention=False):
        # a resnet with optional simple attention
        super(Hybrid, self).__init__()
        self.sub_model = models.resnet50(pretrained=True)        
        # repalce some layers
        self.in_planes = 64
        # first block
        if replace[0]:
            del self.sub_model.layer1
            self.sub_model.layer1 = self._make_layer(block, 64, num_blocks[0], 
                                                     stride=1)
        # second block
        if replace[1]:
            del self.sub_model.layer2
            self.sub_model.layer2 = self._make_layer(block, 128, num_blocks[1], 
                                                     stride=2)
        # third block
        if replace[2]:
            self.in_planes = 128
            del self.sub_model.layer3
            self.sub_model.layer3 = self._make_layer(block, 256, num_blocks[2], 
                                                     stride=2)
        # fourth block
        if replace[3]:
            self.in_planes = 256
            del self.sub_model.layer4
            self.sub_model.layer4 = self._make_layer(block, 512, num_blocks[3], 
                                                     stride=2)
        # FC layer
        del self.sub_model.fc
        self.attention = attention
        
        # a two-layer fully-connected module
        self.sub_model.fc = nn.Sequential(                    
                    nn.Linear(2048, 2048),
                    nn.ReLU(True),
                    nn.Dropout(0.5),
                    nn.Linear(2048, num_classes)) 
        
        if attention:
            self.gamma1 = 0
            self.gamma2 = 0
            self.attention_model = nn.Conv2d(2048, 1, kernel_size=1, stride=1)                 
                    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(len(strides)):
            stride = strides[i]
            # replacing the last X blocks
            if i == 0:
                block_ = BasicBlock
            else:
                block_ = block
            layers.append(block_(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.sub_model.bn1(self.sub_model.conv1(x)))
        out = self.sub_model.maxpool(out)
        out = self.sub_model.layer1(out)
        out = self.sub_model.layer2(out)
        out = self.sub_model.layer3(out)
        out = self.sub_model.layer4(out)
        # 224 by 224 input, the output size is 7 by 7
        reg_loss = 0
        if self.attention:
            # soft attention
#            mask = torch.sigmoid(self.attention_model(out))
#            out = out*mask
            # hard attention
            mask = torch.tanh(self.attention_model(out))
            out = F.relu(out*mask)
            reg_loss = self.gamma1*mask.mean() + self.gamma2*(1 - mask**2).mean()
        out = self.sub_model.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.sub_model.fc(out)
        return out, reg_loss

def Hybridmodel(num_output):
    return Hybrid(HierRes, [6,8,12,5], num_classes=num_output)
