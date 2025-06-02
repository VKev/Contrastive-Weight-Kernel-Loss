import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class AdaptiveBlock(nn.Module):
    def __init__(self, channels: int, hidden_ratio: float, height: int, width: int, rank: int = 8, dropout: float = 0.1):
        """
        AdaptiveBlock with low-rank factorization:
         - Globally pools each channel to a scalar y[b,c]
         - Runs an MLP on the full (B, C) vector → (B, C) to let channels “attend” to each other
         - Takes y′ ∈ (B, C) and, for each (b,c), replicates y′[b] into (C) then passes
           through two linears:
             A[b,c] = Linear(C → H·rank)  → reshaped to (B, C, H, rank)
             B[b,c] = Linear(C → rank·W)  → reshaped to (B, C, rank, W)
         - Performs a per‐channel matmul: M[b,c] = A[b,c] @ B[b,c] ∈ (H, W)
         - Finally, attn_map[b,c] = sigmoid(M[b,c])

        Args:
            channels (int): Number of channels (C).
            hidden_ratio (float): Hidden‐size multiplier for the MLP.
            height (int): Spatial height (H).
            width (int): Spatial width (W).
            rank (int): Low‐rank dimension (r).
            dropout (float): Dropout probability inside the MLP.
        """
        super().__init__()
        self.C = channels
        self.H = height
        self.W = width
        self.r = rank
        self.dropout_p = dropout

        hidden_channels = int(channels * hidden_ratio)
        if hidden_channels == 0:
            raise ValueError("hidden_ratio too small → hidden_channels = 0")

        # 1) MLP that maps (B, C) → (B, C), using GELU + Dropout
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.GELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.GELU(),
            nn.Dropout(p=self.dropout_p),
        )

        self.fc_A = nn.Linear(channels, height * rank, bias=False)

        self.fc_B = nn.Linear(channels, rank * width, bias=False)

        self.sigmoid = nn.Sigmoid()
        
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        nn.init.kaiming_uniform_(
            self.fc_A.weight,
            mode='fan_in',
            nonlinearity='relu'
        )
        nn.init.kaiming_uniform_(
            self.fc_B.weight,
            mode='fan_in',
            nonlinearity='relu'
        )


    def _match_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        If input has fewer than self.C channels, pad with zeros.
        If input has more, slice off the extras.
        """
        B, C_in, H, W = x.shape
        if C_in < self.C:
            pad = torch.zeros((B, self.C - C_in, H, W),
                              device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif C_in > self.C:
            x = x[:, :self.C, :, :]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C_in, H, W). After channel‐matching,
               it must be (B, self.C, self.H, self.W).

        Returns:
            attn_map: Tensor of shape (B, C, H, W).
        """
        x = self._match_channels(x)
        B, C, H, W = x.shape
        assert C == self.C, f"Got C={C}, expected {self.C}"
        if H != self.H or W != self.W:
            raise RuntimeError(f"Expected spatial=({self.H},{self.W}), got ({H},{W})")

        # 1) Global average per‐channel → y ∈ (B, C)
        y = x.mean(dim=[2, 3])  # shape = (B, C)

        # 2) MLP on the pooled vector → y′ ∈ (B, C)
        y_prime = self.mlp(y)   # shape = (B, C)

        # 3) Build (B*C, C) so that we can apply Linear(C → H·r) and Linear(C → r·W)
        y_rep = y_prime.unsqueeze(1).expand(B, C, C)  # (B, C, C)
        y_flat = y_rep.contiguous().view(B * C, C)    # (B*C, C)

        # 4) Apply Linear(C → H·r) → A_flat: (B*C, H·r)
        A_flat = self.fc_A(y_flat)  # (B*C, H*r)
        #    reshape → (B, C, H, r)
        A = A_flat.view(B, C, self.H, self.r)  # (B, C, H, r)

        # 5) Apply Linear(C → r·W) → B_flat: (B*C, r·W)
        B_flat = self.fc_B(y_flat)  # (B*C, r*W)
        #    reshape → (B, C, r, W)
        Bv = B_flat.view(B, C, self.r, self.W)  # (B, C, r, W)

        # 6) Perform per‐channel matmul: (H, r) @ (r, W) → (H, W)
        #    Result M has shape (B, C, H, W)
        M = torch.einsum('b c i k, b c k j -> b c i j', A, Bv)

        # 7) Sigmoid to confine values to [0, 1]
        attn_map = self.sigmoid(M)

        return attn_map


class AdaptBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mask_fn,                        # ❶ shared AdaptiveBlock.forward
        i_downsample=None,
        stride: int = 1
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3   = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu         = nn.ReLU(inplace=True)
        self.i_downsample = i_downsample
        self.mask_fn      = mask_fn 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        mask = self.mask_fn(identity)        # (B, C', H', W')
        out  = out + mask * identity
        return self.relu(out)


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
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # We assume the input’s spatial dims are (input_size, input_size).
        # After conv1: still (input_size, input_size).
        spatial = input_size

        # Layer 1: stride = 1 → output spatial = spatial // 1 = spatial
        self.layer1 = self._make_layer(
            ResBlock,
            planes=64,
            blocks=layers[0],
            stride=1,
            height=spatial,
            width=spatial
        )
        # Update spatial for next stage:
        # Because first block of layer2 will use stride=2, so identity inside block → spatial // 2
        # Next layer’s “post‐downsample” spatial = spatial // 2.
        spatial = spatial // 1  # still input_size, but we do this step for clarity

        # Layer 2: stride = 2 → output spatial = spatial // 2
        spatial = spatial // 2  # new spatial for layer2
        self.layer2 = self._make_layer(
            ResBlock,
            planes=128,
            blocks=layers[1],
            stride=2,
            height=spatial,
            width=spatial
        )

        # Layer 3: stride = 2 → output spatial = (previous spatial) // 2
        spatial = spatial // 2  # new spatial for layer3
        self.layer3 = self._make_layer(
            ResBlock,
            planes=256,
            blocks=layers[2],
            stride=2,
            height=spatial,
            width=spatial
        )

        # Layer 4: stride = 2 → output spatial = (previous spatial) // 2
        spatial = spatial // 2  # new spatial for layer4
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
        self.fc      = nn.Linear(512 * ResBlock.expansion, num_classes)

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
        # Create one AdaptiveBlock per stage, with shared weights
        shared_ab = AdaptiveBlock(ab_channels, self.hidden_ratio, height, width)
        # Register it as a submodule so its parameters are tracked
        self.add_module(f"adapt_ab_{planes}", shared_ab)

        # Determine if we need a downsample on the identity for the first block
        downsample = None
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * ResBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        # First block in this stage
        layers.append(
            ResBlock(
                in_channels=self.in_channels,
                out_channels=planes,
                mask_fn=shared_ab.forward,
                i_downsample=downsample,
                stride=stride
            )
        )
        self.in_channels = planes * ResBlock.expansion

        for _ in range(blocks - 1):
            layers.append(
                ResBlock(
                    in_channels=self.in_channels,
                    out_channels=planes,
                    mask_fn=shared_ab.forward,
                    i_downsample=None,
                    stride=1
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def AdaptResNet50(num_classes: int, channels: int = 3, hidden_ratio: float = 0.25, input_size: int = 224):
    return AdaptResNet(
        AdaptBottleneck,
        [3, 4, 6, 3],
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
    # Example: 10‐class classification on 224×224 RGB images
    model = AdaptResNet50(num_classes=10, channels=3, hidden_ratio=0.25, input_size=32).to(device)
    model.eval()

    dummy = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        out = model(dummy)
        
    print("Trainable parameters in each AdaptiveBlock:")
    total_adaptive_params = 0
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveBlock):
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_adaptive_params += param_count
            print(f"{name}: {param_count} parameters")    
    
    print(f"Output shape : {out.shape}")  # Expected: (4, 10)


if __name__ == "__main__":
    main()
