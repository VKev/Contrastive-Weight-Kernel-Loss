import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveBlock(nn.Module):
    def __init__(
        self,
        max_channels: int,
        hidden_ratio: float,
        max_height: int,
        max_width: int,
        rank: int = 8,
        dropout: float = 0.1,
        total_positions: int = 16  # total number of ResBlocks across all stages
    ):
        super().__init__()
        self.max_C = max_channels
        self.max_H = max_height
        self.max_W = max_width
        self.r = rank
        self.dropout_p = dropout
        self.total_positions = total_positions

        hidden_channels = int(max_channels * hidden_ratio)
        if hidden_channels == 0:
            raise ValueError("hidden_ratio too small → hidden_channels = 0")

        # Positional embeddings: one vector per ResBlock position across all stages
        # Shape = (total_positions, max_channels)
        self.pos_emb = nn.Parameter(torch.zeros(total_positions, max_channels))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        # MLP that maps (B, max_channels) → (B, max_channels), using GELU + Dropout
        self.mlp = nn.Sequential(
            nn.Linear(max_channels, hidden_channels, bias=False),
            nn.GELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(hidden_channels, max_channels, bias=False),
            nn.GELU(),
            nn.Dropout(p=self.dropout_p),
        )

        self.fc_A = nn.Linear(max_channels, max_height * rank, bias=True)
        self.fc_B = nn.Linear(max_channels, rank * max_width, bias=True)
        self.sigmoid = nn.Sigmoid()

        # Kaiming init for all Linear layers in this block
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))

        nn.init.kaiming_uniform_(
            self.fc_A.weight,
            mode='fan_in',
            nonlinearity='relu'
        )
        nn.init.constant_(self.fc_A.bias, 0.0)

        nn.init.kaiming_uniform_(
            self.fc_B.weight,
            mode='fan_in',
            nonlinearity='relu'
        )
        nn.init.constant_(self.fc_B.bias, 0.0)

    def _match_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        If input has fewer than self.max_C channels, pad with zeros.
        If input has more, slice off the extras.
        """
        B, C_in, H, W = x.shape
        if C_in < self.max_C:
            pad = torch.zeros((B, self.max_C - C_in, H, W),
                              device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif C_in > self.max_C:
            x = x[:, :self.max_C, :, :]
        return x

    def forward(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C_in, H, W).
            pos: integer in [0, total_positions), indicating which ResBlock
                 this call corresponds to (for positional embedding).
        Returns:
            attn_map: Tensor of shape (B, C_in, H, W) - same shape as input.
        """
        B, C_in, H_in, W_in = x.shape
        
        # Match channels for processing
        x_padded = self._match_channels(x)
        B, C_padded, H, W = x_padded.shape

        # 1) Global average per‐channel → y ∈ (B, C_padded)
        y = x_padded.mean(dim=[2, 3])  # shape = (B, C_padded)

        # 1.5) Add positional embedding for this ResBlock
        # self.pos_emb[pos] has shape (C_padded,), broadcast to (B, C_padded)
        y = y + self.pos_emb[pos]

        # 2) MLP on the pooled vector → y′ ∈ (B, C_padded)
        y_prime = self.mlp(y)   # shape = (B, C_padded)

        # 3) Build (B*C_in, C_padded) so that we can apply Linear operations
        # Only use the actual input channels for efficiency
        y_rep = y_prime.unsqueeze(1).expand(B, C_in, self.max_C)  # (B, C_in, C_padded)
        y_flat = y_rep.contiguous().view(B * C_in, self.max_C)    # (B*C_in, C_padded)

        # 4) Apply Linear(C_padded → max_H·r) → A_flat: (B*C_in, max_H·r)
        A_flat = self.fc_A(y_flat)  # (B*C_in, max_H·r)
        #    reshape → (B, C_in, max_H, r)
        A = A_flat.view(B, C_in, self.max_H, self.r)  # (B, C_in, max_H, r)

        # 5) Apply Linear(C_padded → r·max_W) → B_flat: (B*C_in, r·max_W)
        B_flat = self.fc_B(y_flat)  # (B*C_in, r·max_W)
        #    reshape → (B, C_in, r, max_W)
        Bv = B_flat.view(B, C_in, self.r, self.max_W)  # (B, C_in, r, max_W)

        # 6) TRUNCATE A AND B TO MATCH INPUT DIMENSIONS BEFORE MATMUL
        A_truncated = A[:, :, :H_in, :]  # (B, C_in, H_in, r)
        Bv_truncated = Bv[:, :, :, :W_in]  # (B, C_in, r, W_in)

        # 7) Perform per‐channel matmul: (H_in, r) @ (r, W_in) → (H_in, W_in)
        #    Result M has shape (B, C_in, H_in, W_in) - matches input exactly
        M = torch.einsum('b c i k, b c k j -> b c i j', A_truncated, Bv_truncated)

        # 8) Sigmoid to confine values to [0, 1]
        attn_map = self.sigmoid(M)
        return attn_map


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
        self.batch_norm1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2   = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.batch_norm3   = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu         = nn.ReLU(inplace=True)
        self.i_downsample = i_downsample
        self.mask_fn      = mask_fn  # Now an nn.Module that encapsulates shared_ab + position

        # **NEW**: store the last‐forward mask here
        self.last_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.relu(self.batch_norm2(self.conv2(out)))
        out = self.batch_norm3(self.conv3(out))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        # ------------- NEW: compute & store mask -------------
        mask = self.mask_fn(identity)        # (B, C', H', W')
        self.last_mask = mask

        out  = out + mask * identity
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
        self.in_channels  = 64
        self.hidden_ratio = hidden_ratio

        # Initial convolution + BN + ReLU (no maxpool to keep higher resolution)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # Calculate total number of blocks across all layers
        total_blocks = sum(layers)
        
        # Calculate max dimensions across all layers
        max_channels = 512 * ResBlock.expansion  # layer4 has the most channels
        max_spatial = input_size  # layer1 has the largest spatial dimensions

        # Create ONE shared AdaptiveBlock for all layers
        self.shared_adaptive_block = AdaptiveBlock(
            max_channels=max_channels,
            hidden_ratio=hidden_ratio,
            max_height=max_spatial,
            max_width=max_spatial,
            rank=8,
            dropout=0.1,
            total_positions=total_blocks
        )

        # We assume the input's spatial dims are (input_size, input_size).
        spatial = input_size  # current spatial resolution
        position_counter = 0  # Track global position across all layers

        # Layer 1: stride = 1 → spatial remains = input_size
        self.layer1, position_counter = self._make_layer(
            ResBlock,
            planes=64,
            blocks=layers[0],
            stride=1,
            height=spatial,
            width=spatial,
            position_start=position_counter
        )
        
        # Layer 2: stride = 2 → spatial //= 2
        spatial = spatial // 2
        self.layer2, position_counter = self._make_layer(
            ResBlock,
            planes=128,
            blocks=layers[1],
            stride=2,
            height=spatial,
            width=spatial,
            position_start=position_counter
        )
        
        # Layer 3: stride = 2 → spatial //= 2
        spatial = spatial // 2
        self.layer3, position_counter = self._make_layer(
            ResBlock,
            planes=256,
            blocks=layers[2],
            stride=2,
            height=spatial,
            width=spatial,
            position_start=position_counter
        )
        
        # Layer 4: stride = 2 → spatial //= 2
        spatial = spatial // 2
        self.layer4, position_counter = self._make_layer(
            ResBlock,
            planes=512,
            blocks=layers[3],
            stride=2,
            height=spatial,
            width=spatial,
            position_start=position_counter
        )

        # Final global average pool + linear classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * ResBlock.expansion, num_classes)

    def _make_layer(
        self,
        ResBlock,
        planes: int,
        blocks: int,
        stride: int,
        height: int,
        width: int,
        position_start: int
    ) -> tuple:
        layers = []

        # Number of channels feeding into AdaptiveBlock = planes * expansion
        ab_channels = planes * ResBlock.expansion

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

        position_counter = position_start
        for block_idx in range(blocks):
            # For the first block: use `stride` and `downsample`; subsequent blocks use stride=1, no downsample
            block_stride     = stride if block_idx == 0 else 1
            block_downsample = downsample if block_idx == 0 else None

            # Create a MaskWrapper module that captures this block's global position
            mask_wrapper = MaskWrapper(self.shared_adaptive_block, position_counter)

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
            
            position_counter += 1

        return nn.Sequential(*layers), position_counter

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

    # Print how many parameters the single shared AdaptiveBlock has
    print("Trainable parameters in the single shared AdaptiveBlock:")
    param_count = sum(p.numel() for p in model.shared_adaptive_block.parameters() if p.requires_grad)
    print(f"shared_adaptive_block: {param_count} parameters")

    # Check shapes
    print(f"Output logits shape : {logits.shape}")   # (4, 10)
    print(f"Number of masks returned: {len(masks)}")  # e.g. for ResNet50: 3+4+6+3 = 16 masks
    for i, m in enumerate(masks[:3]):
        print(f"  mask {i} shape = {m.shape}")        # each is (4, C_i, H_i, W_i)

    # Print total model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params}")


if __name__ == "__main__":
    main()