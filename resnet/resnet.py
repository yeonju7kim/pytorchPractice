import collections
from random import random
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from PIL import Image

GlobalParams = collections.namedtuple("GlobalParams", [
    "block", "zero_init_residual",
    "groups", "width_per_group", "replace_stride_with_dilation",
    "norm_layer", "num_classes", "image_size"])

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
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
    __constants__ = ["downsample"]

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        # 1x1 conv로 param 수 줄임
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    # inplanes ch -> width channel ->  planes * self.expansion channel

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


def resnet_params(model_name):

    params_dict = {
        # Coefficients:   block, res
        "resnet18": (BasicBlock, 224),
        "resnet34": (BasicBlock, 224),
        "resnet54": (Bottleneck, 224),
        "resnet101": (Bottleneck, 224),
        "resnet152": (Bottleneck, 224),
    }
    return params_dict[model_name]


def resnet(arch, block, num_classes=1000, zero_init_residual=False,
           groups=1, width_per_group=64, replace_stride_with_dilation=None,
           norm_layer=None, image_size=224):
    """ Creates a resnet_pytorch model. """

    global_params = GlobalParams(
        block=block,
        num_classes=num_classes,
        zero_init_residual=zero_init_residual,
        groups=groups,
        width_per_group=width_per_group,
        replace_stride_with_dilation=replace_stride_with_dilation,
        norm_layer=norm_layer,
        image_size=image_size,
    )

    layers_dict = {
        "resnet18": (2, 2, 2, 2),
        "resnet34": (3, 4, 6, 3),
        "resnet54": (3, 4, 6, 3),
        "resnet101": (3, 4, 23, 3),
        "resnet152": (3, 8, 36, 3),
    }
    layers = layers_dict[arch]

    return layers, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith("resnet"):
        b, s = resnet_params(model_name)
        layers, global_params = resnet(arch=model_name, block=b, image_size=s)
    else:
        raise NotImplementedError(f"model name is not pre-defined: {model_name}")
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return layers, global_params


urls_map = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def load_pretrained_weights(model, model_name, load_fc=True):
    # pretrain 모델 불러오기
    state_dict = model_zoo.load_url(urls_map[model_name])
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == {"fc.weight", "fc.bias"}, "issue loading pretrained weights"
    print(f"Loaded pretrained weights for {model_name}.")

class RandomResize(object):
    def __init__(self, range):
        assert isinstance(range, tuple)
        self.range = range
    def __call__(self, sample):
        image = sample
        h, w = image.shape[1:3]
        short_size = (int) ((self.range[1] - self.range[0]) * random() + self.range[0])
        if h > w:
            new_h, new_w = (int)(short_size * h / w), short_size
        else:
            new_w, new_h = (int)(short_size * w / h), short_size
        pil_img = Image.fromarray(image, "RGB")
        pil_img = pil_img.resize((new_h, new_w))

        return pil_img