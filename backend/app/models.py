def build_compact_unet():
    """Build a small binary 2D U-Net; imported only when segmentation runs."""
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, incoming, outgoing):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
                nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.layers(x)

    class CompactUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = Block(3, 16)
            self.enc2 = Block(16, 32)
            self.enc3 = Block(32, 64)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec2 = Block(64, 32)
            self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dec1 = Block(32, 16)
            self.out = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            a = self.enc1(x)
            b = self.enc2(self.pool(a))
            c = self.enc3(self.pool(b))
            y = self.dec2(__import__("torch").cat([self.up2(c), b], dim=1))
            y = self.dec1(__import__("torch").cat([self.up1(y), a], dim=1))
            return self.out(y)

    return CompactUNet()


def build_attention_unet(base_channels: int = 16):
    """Build a residual Attention U-Net for binary 2D tumor-region masks."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    def norm(channels: int):
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)

    class ResidualBlock(nn.Module):
        def __init__(self, incoming: int, outgoing: int):
            super().__init__()
            self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False)
            self.norm1 = norm(outgoing)
            self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
            self.norm2 = norm(outgoing)
            self.skip = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1, bias=False)
            self.activation = nn.SiLU(inplace=True)

        def forward(self, value):
            residual = self.skip(value)
            value = self.activation(self.norm1(self.conv1(value)))
            value = self.norm2(self.conv2(value))
            return self.activation(value + residual)

    class AttentionGate(nn.Module):
        def __init__(self, skip_channels: int, gate_channels: int, hidden_channels: int):
            super().__init__()
            self.skip_projection = nn.Conv2d(skip_channels, hidden_channels, 1, bias=True)
            self.gate_projection = nn.Conv2d(gate_channels, hidden_channels, 1, bias=True)
            self.score = nn.Sequential(nn.SiLU(inplace=True), nn.Conv2d(hidden_channels, 1, 1), nn.Sigmoid())

        def forward(self, skip, gate):
            if gate.shape[-2:] != skip.shape[-2:]:
                gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            weights = self.score(self.skip_projection(skip) + self.gate_projection(gate))
            return skip * weights

    class DecoderStage(nn.Module):
        def __init__(self, incoming: int, skip_channels: int, outgoing: int):
            super().__init__()
            self.up = nn.ConvTranspose2d(incoming, outgoing, 2, stride=2)
            self.attention = AttentionGate(skip_channels, outgoing, max(outgoing // 2, 8))
            self.block = ResidualBlock(outgoing + skip_channels, outgoing)

        def forward(self, value, skip):
            value = self.up(value)
            weighted_skip = self.attention(skip, value)
            return self.block(torch.cat((value, weighted_skip), dim=1))

    class AttentionUNet(nn.Module):
        def __init__(self):
            super().__init__()
            c1, c2, c3, c4, c5 = (base_channels * scale for scale in (1, 2, 4, 8, 16))
            self.enc1 = ResidualBlock(3, c1)
            self.enc2 = ResidualBlock(c1, c2)
            self.enc3 = ResidualBlock(c2, c3)
            self.enc4 = ResidualBlock(c3, c4)
            self.bottleneck = nn.Sequential(ResidualBlock(c4, c5), nn.Dropout2d(.10))
            self.pool = nn.MaxPool2d(2)
            self.dec4 = DecoderStage(c5, c4, c4)
            self.dec3 = DecoderStage(c4, c3, c3)
            self.dec2 = DecoderStage(c3, c2, c2)
            self.dec1 = DecoderStage(c2, c1, c1)
            self.out = nn.Conv2d(c1, 1, 1)

        def forward(self, value):
            a = self.enc1(value)
            b = self.enc2(self.pool(a))
            c = self.enc3(self.pool(b))
            d = self.enc4(self.pool(c))
            center = self.bottleneck(self.pool(d))
            value = self.dec4(center, d)
            value = self.dec3(value, c)
            value = self.dec2(value, b)
            value = self.dec1(value, a)
            return self.out(value)

    return AttentionUNet()


def predict_segmentation_probability(model, images, horizontal_flip_tta: bool = False):
    """Average native and inverse-flipped probabilities when validation supports TTA."""
    import torch

    native = torch.sigmoid(model(images))
    if not horizontal_flip_tta:
        return native
    flipped = torch.flip(images, dims=(-1,))
    flipped_probability = torch.flip(torch.sigmoid(model(flipped)), dims=(-1,))
    return (native + flipped_probability) * 0.5
