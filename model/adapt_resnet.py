import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_ratio: float,  # Keeping for compatibility but not used
        height: int,
        width: int,
        rank: int = 8,  # Keeping for compatibility but not used
        dropout: float = 0.1,  # Keeping for compatibility but not used
        num_positions: int = 1  # number of ResBlocks in this stage
    ):
        super().__init__()
        self.C = channels
        self.H = height
        self.W = width
        self.num_positions = num_positions

        # 2D Positional embeddings: one 2D embedding per ResBlock position
        # Shape = (num_positions, C, H, W)
        self.pos_emb_2d = nn.Parameter(torch.zeros(num_positions, channels, height, width))
        nn.init.normal_(self.pos_emb_2d, mean=0.0, std=0.02)

        # First Conv1x1 to create intermediate representation
        self.mask_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Initialize conv1x1 layers
        nn.init.kaiming_uniform_(
            self.mask_conv.weight,
            mode='fan_in',
            nonlinearity='relu'
        )

    def forward(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        """
        Forward pass:
        1. Add 2D positional encoding to input feature map
        2. Apply first conv1x1
        3. Apply ReLU activation
        4. Apply second conv1x1 to create mask
        5. Apply sigmoid activation
        """
        B, C, H, W = x.shape
        
        # Ensure input dimensions match expected dimensions
        if C != self.C or H != self.H or W != self.W:
            raise RuntimeError(f"Expected input shape=({self.C},{self.H},{self.W}), got ({C},{H},{W})")

        x_with_pos = x + self.pos_emb_2d[pos]  # (B, C, H, W)

        mask = self.mask_conv(x_with_pos)  # (B, C, H, W)

        return torch.sigmoid(mask)


class AdaptBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mask_fn: nn.Module,  # Now expects a nn.Module rather than lambda
        i_downsample=None,
        stride: int = 1
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.i_downsample = i_downsample
        self.mask_fn = mask_fn  # Now an nn.Module that encapsulates shared_ab + position

        # **NEW**: store the last‐forward mask here
        self.last_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()

        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.relu(self.batch_norm2(self.conv2(out)))
        out = self.batch_norm3(self.conv3(out))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        # ------------- NEW: compute & store mask -------------
        mask = self.mask_fn(identity)        # (B, C', H', W')
        self.last_mask = mask

        # Optimized: use in-place multiplication and addition where possible
        out.add_(mask * identity)
        return self.relu(out)


class MaskWrapper(nn.Module):
    """
    Wraps a shared AdaptiveBlock and a fixed position index so that calling
    this module returns shared_ab(x, position).
    """
    def __init__(self, shared_ab: AdaptiveBlock, position: int):
        super().__init__()
        self.shared_ab = shared_ab
        self.position = position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the shared AdaptiveBlock with the stored position
        return self.shared_ab(x, self.position)


class AdaptResNet(nn.Module):
    def __init__(
        self,
        ResBlock,
        layers: list,
        num_classes: int,
        num_channels: int = 3,
        hidden_ratio: float = 0.25,
        input_size: int = 224
    ):
        super().__init__()
        self.in_channels = 64
        self.hidden_ratio = hidden_ratio

        # Initial convolution + BN + ReLU (no maxpool to keep higher resolution)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # We assume the input's spatial dims are (input_size, input_size).
        spatial = input_size  # current spatial resolution

        # Layer 1: stride = 1 → spatial remains = input_size
        self.layer1 = self._make_layer(
            ResBlock,
            planes=64,
            blocks=layers[0],
            stride=1,
            height=spatial,
            width=spatial
        )
        # After layer1, spatial does not change
        # Layer 2: stride = 2 → spatial //= 2
        spatial = spatial // 2
        self.layer2 = self._make_layer(
            ResBlock,
            planes=128,
            blocks=layers[1],
            stride=2,
            height=spatial,
            width=spatial
        )
        # Layer 3: stride = 2 → spatial //= 2
        spatial = spatial // 2
        self.layer3 = self._make_layer(
            ResBlock,
            planes=256,
            blocks=layers[2],
            stride=2,
            height=spatial,
            width=spatial
        )
        # Layer 4: stride = 2 → spatial //= 2
        spatial = spatial // 2
        self.layer4 = self._make_layer(
            ResBlock,
            planes=512,
            blocks=layers[3],
            stride=2,
            height=spatial,
            width=spatial
        )

        # Final global average pool + linear classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def _make_layer(
        self,
        ResBlock,
        planes: int,
        blocks: int,
        stride: int,
        height: int,
        width: int
    ) -> nn.Sequential:
        layers = []

        # Number of channels feeding into AdaptiveBlock = planes * expansion
        ab_channels = planes * ResBlock.expansion

        # Create one shared AdaptiveBlock per stage, with a positional embedding of size `blocks`
        shared_ab = AdaptiveBlock(
            ab_channels,
            self.hidden_ratio,
            height,
            width,
            rank=8,
            dropout=0.1,
            num_positions=blocks
        )
        # Register the shared AdaptiveBlock so its parameters are tracked
        self.add_module(f"adapt_ab_{planes}", shared_ab)

        # Determine if we need a downsample on the identity for the first block
        downsample = None
        if stride != 1 or self.in_channels != ab_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    ab_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(ab_channels)
            )

        for block_idx in range(blocks):
            # For the first block: use `stride` and `downsample`; subsequent blocks use stride=1, no downsample
            block_stride = stride if block_idx == 0 else 1
            block_downsample = downsample if block_idx == 0 else None

            # Create a MaskWrapper module that captures this block's index
            mask_wrapper = MaskWrapper(shared_ab, block_idx)

            # Now create the ResBlock, passing mask_wrapper
            layers.append(
                ResBlock(
                    in_channels=self.in_channels,
                    out_channels=planes,
                    mask_fn=mask_wrapper,
                    i_downsample=block_downsample,
                    stride=block_stride
                )
            )

            # After the first block, update in_channels to ab_channels, and clear downsample
            if block_idx == 0:
                self.in_channels = ab_channels
                downsample = None

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, list):
        """
        Returns:
          - logits: shape (B, num_classes)
          - masks:  list of Tensors [ (B, C_i, H_i, W_i), ... ] collected from every AdaptBottleneck
        """
        # 1) Initial conv + bn + relu
        x = self.relu(self.batch_norm1(self.conv1(x)))

        all_masks = []

        # 2) Manually run through layer1, layer2, layer3, layer4, collecting each block's .last_mask
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                x = block(x)
                # block.last_mask was set in AdaptBottleneck.forward
                all_masks.append(block.last_mask)

        # 3) Global avgpool + flatten + fc
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)

        return logits, all_masks


def AdaptResNet50(num_classes: int, channels: int = 3, hidden_ratio: float = 0.25, input_size: int = 224):
    return AdaptResNet(
        AdaptBottleneck,
        [3, 4, 6, 3],            # number of ResBlocks in each of the 4 stages
        num_classes,
        num_channels=channels,
        hidden_ratio=hidden_ratio,
        input_size=input_size
    )


def AdaptResNet101(num_classes: int, channels: int = 3, hidden_ratio: float = 0.25, input_size: int = 224):
    return AdaptResNet(
        AdaptBottleneck,
        [3, 4, 23, 3],
        num_classes,
        num_channels=channels,
        hidden_ratio=hidden_ratio,
        input_size=input_size
    )


def AdaptResNet152(num_classes: int, channels: int = 3, hidden_ratio: float = 0.25, input_size: int = 224):
    return AdaptResNet(
        AdaptBottleneck,
        [3, 8, 36, 3],
        num_classes,
        num_channels=channels,
        hidden_ratio=hidden_ratio,
        input_size=input_size
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate AdaptResNet50 on a smaller input (32×32) just for testing
    model = AdaptResNet50(num_classes=10, channels=3, hidden_ratio=0.25, input_size=32).to(device)
    model.eval()

    dummy = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        logits, masks = model(dummy)

    # Print how many parameters each shared AdaptiveBlock has
    print("Trainable parameters in each shared AdaptiveBlock:")
    total_adaptive_params = 0
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveBlock):
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_adaptive_params += param_count
            print(f"{name}: {param_count} parameters")
    print(f"Total across all shared AdaptiveBlocks: {total_adaptive_params}")

    # Check shapes
    print(f"Output logits shape : {logits.shape}")   # (4, 10)
    print(f"Number of masks returned: {len(masks)}")  # e.g. for ResNet50: 3+4+6+3 = 16 masks
    for i, m in enumerate(masks[:3]):
        print(f"  mask {i} shape = {m.shape}")        # each is (4, C_i, H_i, W_i)


if __name__ == "__main__":
    main()
