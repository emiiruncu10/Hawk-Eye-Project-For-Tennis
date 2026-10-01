"""TrackNet: VGG-style encoder-decoder that turns a 9-channel (3 x RGB)
frame stack into a 2-D Gaussian heat-map of the ball position.

Two variants are provided behind one class:

* ``tracknet_v1()`` – layer names, block layout and the 256-class softmax
  head follow **yastrebksv/TrackNet** (``BallTrackerNet``), so its
  ``model_best.pt`` checkpoint loads with ``strict=True``. Input 360x640.
* ``tracknet_v2()`` – TrackNetV2-style U-Net with skip connections and a
  3-channel sigmoid head (one heat-map per input frame), as in
  **ChgygLin/TrackNetV2-pytorch**. Input 288x512.

``TrackNet.heatmaps(x)`` always returns values in [0, 1] with shape
(N, K, H, W) regardless of the head, so the detector does not care which
variant is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv3x3 -> ReLU -> BatchNorm (the order used by yastrebksv/TrackNet)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 pad: int = 1, stride: int = 1, bias: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class TrackNet(nn.Module):
    """Encoder-decoder with 18 conv blocks, 3 max-pools and 3 x2 up-samplings."""

    def __init__(self, in_channels: int = 9, out_channels: int = 256,
                 head: str = "softmax256", skip_connections: bool = False,
                 final_1x1: bool = False):
        super().__init__()
        if head not in ("softmax256", "sigmoid"):
            raise ValueError("head must be 'softmax256' or 'sigmoid'")
        if head == "softmax256" and out_channels != 256:
            raise ValueError("softmax256 head requires out_channels == 256")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head = head
        self.skip_connections = skip_connections
        s = skip_connections
        self.final_1x1 = final_1x1

        # encoder
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        # decoder
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(512 + (256 if s else 0), 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(256 + (128 if s else 0), 128)
        self.conv15 = ConvBlock(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(128 + (64 if s else 0), 64)
        self.conv17 = ConvBlock(64, 64)
        # TrackNetV2 (ChgygLin) ends with a plain 1x1 conv; V1 uses a full ConvBlock.
        self.final_1x1 = final_1x1
        self.conv18 = nn.Conv2d(64, out_channels, kernel_size=1) if final_1x1 else ConvBlock(64, out_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits (N, out_channels, H, W). H and W must be multiples of 8."""
        if x.shape[-1] % 8 or x.shape[-2] % 8:
            raise ValueError(f"input H and W must be multiples of 8, got {tuple(x.shape[-2:])}")
        x1 = self.conv2(self.conv1(x))
        x = self.pool1(x1)
        x2 = self.conv4(self.conv3(x))
        x = self.pool2(x2)
        x3 = self.conv7(self.conv6(self.conv5(x)))
        x = self.pool3(x3)
        x = self.conv10(self.conv9(self.conv8(x)))
        x = self.ups1(x)
        if self.skip_connections:
            x = torch.cat([x, x3], dim=1)
        x = self.conv13(self.conv12(self.conv11(x)))
        x = self.ups2(x)
        if self.skip_connections:
            x = torch.cat([x, x2], dim=1)
        x = self.conv15(self.conv14(x))
        x = self.ups3(x)
        if self.skip_connections:
            x = torch.cat([x, x1], dim=1)
        x = self.conv17(self.conv16(x))
        return self.conv18(x)

    def heatmaps(self, x: torch.Tensor) -> torch.Tensor:
        """Probability-like heat-maps in [0, 1], shape (N, K, H, W).

        * sigmoid head: K = out_channels (one map per input frame for V2).
        * softmax256 head: K = 1, the expected grey level / 255 of the
          per-pixel 256-way classification (smooth version of the argmax
          decoding used by yastrebksv/TrackNet).
        """
        logits = self.forward(x)
        if self.head == "sigmoid":
            return torch.sigmoid(logits)
        probs = F.softmax(logits, dim=1)
        levels = torch.arange(self.out_channels, device=x.device, dtype=probs.dtype) / (self.out_channels - 1)
        return (probs * levels.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def tracknet_v1() -> TrackNet:
    """yastrebksv/TrackNet-compatible network (360x640, BGR, newest frame first)."""
    return TrackNet(in_channels=9, out_channels=256, head="softmax256", skip_connections=False)


def tracknet_v2() -> TrackNet:
    """TrackNetV2 network as in ChgygLin/TrackNetV2-pytorch (288x512, RGB, oldest
    frame first, 3 sigmoid heat-maps, 1x1 output conv). Its ``conv2d_N.conv/bn``
    checkpoint keys are mapped onto ``convN.block.0/2`` by :func:`load_weights`."""
    return TrackNet(in_channels=9, out_channels=3, head="sigmoid", skip_connections=True, final_1x1=True)


# arch name -> (constructor, input preset name, index of the heat-map for frame t)
ARCHS = {
    "v1": {"build": tracknet_v1, "preset": "yastrebksv", "output_frame_index": 0},
    "v2": {"build": tracknet_v2, "preset": "tracknet_v2", "output_frame_index": 2},
}


def build_model(arch: str = "v2") -> TrackNet:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch '{arch}', choose from {sorted(ARCHS)}")
    return ARCHS[arch]["build"]()
