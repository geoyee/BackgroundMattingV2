import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .encoder import MobileNetV2Encoder, ResNetEncoder
from .decoder import Decoder
from .refiner import Refiner
from .utils import ASPP, load_matched_state_dict


class Base(nn.Layer):
    """
    A generic implementation of the base encoder-decoder network inspired by DeepLab.
    Accepts arbitrary channels for input and output.
    """
    
    def __init__(self, backbone: str, in_channels: int, out_channels: int):
        super().__init__()
        assert backbone in ["resnet50", "resnet101", "mobilenetv2"]
        if backbone in ['resnet50', 'resnet101']:
            self.backbone = ResNetEncoder(in_channels, variant=backbone)
            self.aspp = ASPP(2048, [3, 6, 9])
            self.decoder = Decoder([256, 128, 64, 48, out_channels], [512, 256, 64, in_channels])
        else:
            self.backbone = MobileNetV2Encoder(in_channels)
            self.aspp = ASPP(320, [3, 6, 9])
            self.decoder = Decoder([256, 128, 64, 48, out_channels], [32, 24, 16, in_channels])

    def forward(self, x):
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        return x
    
    def load_pretrained_deeplabv3_state_dict(self, state_dict, print_stats=True):
        # Pretrained DeepLabV3 models are provided by <https://github.com/VainF/DeepLabV3Plus-Pytorch>.
        # This method converts and loads their pretrained state_dict to match with our model structure.
        # This method is not needed if you are not planning to train from deeplab weights.
        # Use load_state_dict() for normal weight loading.
        
        # Convert state_dict naming for aspp module
        state_dict = {k.replace('classifier.classifier.0', 'aspp'): v for k, v in state_dict.items()}

        if isinstance(self.backbone, ResNetEncoder):
            # ResNet backbone does not need change.
            load_matched_state_dict(self, state_dict, print_stats)
        else:
            # Change MobileNetV2 backbone to state_dict format, then change back after loading.
            backbone_features = self.backbone.features
            self.backbone.low_level_features = backbone_features[:4]
            self.backbone.high_level_features = backbone_features[4:]
            del self.backbone.features
            load_matched_state_dict(self, state_dict, print_stats)
            self.backbone.features = backbone_features
            del self.backbone.low_level_features
            del self.backbone.high_level_features


class MattingBase(Base):
    """
    MattingBase is used to produce coarse global results at a lower resolution.
    MattingBase extends Base.
    
    Args:
        backbone: ["resnet50", "resnet101", "mobilenetv2"]
        
    Input:
        src: (B, 3, H, W) the source image. Channels are RGB values normalized to 0 ~ 1.
        bgr: (B, 3, H, W) the background image . Channels are RGB values normalized to 0 ~ 1.
    
    Output:
        pha: (B, 1, H, W) the alpha prediction. Normalized to 0 ~ 1.
        fgr: (B, 3, H, W) the foreground prediction. Channels are RGB values normalized to 0 ~ 1.
        err: (B, 1, H, W) the error prediction. Normalized to 0 ~ 1.
        hid: (B, 32, H, W) the hidden encoding. Used for connecting refiner module.
        
    Example:
        model = MattingBase(backbone='resnet50')
        
        pha, fgr, err, hid = model(src, bgr)    # for training
        pha, fgr = model(src, bgr)[:2]          # for inference
    """
    
    def __init__(self, backbone: str):
        super().__init__(backbone, in_channels=6, out_channels=(1 + 3 + 1 + 32))
        
    def forward(self, src, bgr):
        x = paddle.concat([src, bgr], axis=1)
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        pha = paddle.clip(x[:, 0:1], 0., 1.)
        fgr = paddle.clip(x[:, 1:4].add(src), 0., 1.)
        err = paddle.clip(x[:, 4:5], 0., 1.)
        hid = F.relu(x[:, 5: ])
        return pha, fgr, err, hid