import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import List


def constant_init(param, **kwargs):
    """
    Initialize the `param` with constants.

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        param_init.constant_init(linear.weight, value=2.0)
        print(linear.weight.numpy())
        # result is [[2. 2. 2. 2.], [2. 2. 2. 2.]]

    """
    initializer = nn.initializer.Constant(**kwargs)
    initializer(param, param.block)


def normal_init(param, **kwargs):
    """
    Initialize the `param` with a Normal distribution.

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        param_init.normal_init(linear.weight, loc=0.0, scale=1.0)

    """
    initializer = nn.initializer.Normal(**kwargs)
    initializer(param, param.block)


def kaiming_normal_init(param, **kwargs):
    r"""
    Initialize the input tensor with Kaiming Normal initialization.

    This function implements the `param` initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities. In case of Uniform distribution, the range is [-x, x], where
    .. math::
        x = \sqrt{\\frac{6.0}{fan\_in}}
    In case of Normal distribution, the mean is 0 and the standard deviation
    is
    .. math::
        \sqrt{\\frac{2.0}{fan\_in}}

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        # uniform is used to decide whether to use uniform or normal distribution
        param_init.kaiming_normal_init(linear.weight)

    """
    initializer = nn.initializer.KaimingNormal(**kwargs)
    initializer(param, param.block)

def load_matched_state_dict(model, state_dict, print_stats=True):
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.set_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')


class Identity(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2D(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Layer):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.LayerList(modules)

        self.project = nn.Sequential(
            nn.Conv2D(len(self.convs) * out_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = paddle.concat(_res, axis=1)
        return self.project(res)