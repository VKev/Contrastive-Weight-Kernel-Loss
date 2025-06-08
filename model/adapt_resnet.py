'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


class AdaptiveBlock(nn.Module):
    def __init__(
        self,
        max_channels: int = 64,  # Maximum channels across all layers
        total_positions: int = 18,  # Total number of ResBlocks across all layers
    ):
        super().__init__()
        self.max_channels = max_channels
        self.total_positions = total_positions

        # Learnable position embeddings for each block position
        # We'll dynamically slice these based on input dimensions
        self.pos_emb_1d = nn.Parameter(torch.zeros(total_positions, max_channels))
        nn.init.xavier_normal_(self.pos_emb_1d)

        # Adaptive conv layers that work with variable input sizes
        self.mask_conv = nn.Sequential(
            nn.Conv2d(max_channels, max_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(max_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(max_channels, max_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(max_channels),
            nn.Sigmoid()
        )
        
        for m in self.mask_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Pad channels to max_channels if needed
        if C < self.max_channels:
            x_padded = torch.zeros(B, self.max_channels, H, W, device=x.device, dtype=x.dtype)
            x_padded[:, :C, :, :] = x
        else:
            x_padded = x
        
        # Add position embedding (broadcast across spatial dimensions)
        pos_emb = self.pos_emb_1d[pos]  # (max_channels,)
        pos_emb = pos_emb.view(1, self.max_channels, 1, 1)  # (1, max_channels, 1, 1)
        x_with_pos = x_padded + pos_emb
        
        # Generate mask
        mask_full = self.mask_conv(x_with_pos)  # (B, max_channels, H, W)
        
        # Slice mask to match original input channels
        mask = mask_full[:, :C, :, :]  # (B, C, H, W)

        return mask


class MaskWrapper(nn.Module):
    """
    Wraps a shared AdaptiveBlock and a fixed global position index so that calling
    this module returns shared_ab(x, global_position).
    """
    def __init__(self, shared_ab: AdaptiveBlock, global_position: int):
        super().__init__()
        self.shared_ab = shared_ab
        self.global_position = global_position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the shared AdaptiveBlock with the stored global position
        return self.shared_ab(x, self.global_position)

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', mask_fn=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

        self.mask_fn = mask_fn  # AdaptiveBlock wrapper
        self.last_mask = None  # Store the last forward mask

    def forward(self, x):
        identity = x.clone()
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply shortcut to identity
        identity = self.shortcut(identity)
        
        # Generate and apply adaptive mask if mask_fn is provided
        if self.mask_fn is not None:
            mask = self.mask_fn(identity)  # Generate mask based on identity
            self.last_mask = mask
            out = out + mask * identity  # Apply masked identity
        else:
            out = out + identity  # Standard residual connection
            
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_adaptive_masks=True, input_size=32):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.use_adaptive_masks = use_adaptive_masks

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Create single shared adaptive block for all layers if using adaptive masks
        self.shared_adaptive_block = None
        self.global_block_position = 0  # Track global position across all layers
        
        if self.use_adaptive_masks:
            total_blocks = sum(num_blocks)  # Total number of ResBlocks across all layers
            max_channels = 64  # Maximum channels in layer3
            self.shared_adaptive_block = AdaptiveBlock(
                max_channels=max_channels,
                total_positions=total_blocks
            )
        
        # Calculate spatial dimensions for each layer (CIFAR-10 starts at 32x32)
        spatial = input_size  # 32
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        # After layer1, spatial remains 32
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        # After layer2, spatial becomes 16
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # After layer3, spatial becomes 8
        
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        for block_idx, stride in enumerate(strides):
            # Create mask wrapper if using adaptive masks
            mask_fn = None
            if self.use_adaptive_masks and self.shared_adaptive_block is not None:
                mask_fn = MaskWrapper(self.shared_adaptive_block, self.global_block_position)
                self.global_block_position += 1  # Increment global position
                
            layers.append(block(self.in_planes, planes, stride, mask_fn=mask_fn))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        all_masks = []
        
        if self.use_adaptive_masks:
            # Manually run through layers, collecting masks
            for layer in (self.layer1, self.layer2, self.layer3):
                for block in layer:
                    out = block(out)
                    if block.last_mask is not None:
                        all_masks.append(block.last_mask)
        else:
            # Standard forward pass without mask collection
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        
        if self.use_adaptive_masks:
            return logits, all_masks
        else:
            return logits


def resnet20(num_classes=10, use_adaptive_masks=True, input_size=32):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, use_adaptive_masks=use_adaptive_masks, input_size=input_size)


def resnet32(num_classes=10, use_adaptive_masks=True, input_size=32):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, use_adaptive_masks=use_adaptive_masks, input_size=input_size)


def resnet44(num_classes=10, use_adaptive_masks=True, input_size=32):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, use_adaptive_masks=use_adaptive_masks, input_size=input_size)


def resnet56(num_classes=10, use_adaptive_masks=True, input_size=32):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, use_adaptive_masks=use_adaptive_masks, input_size=input_size)


def resnet110(num_classes=10, use_adaptive_masks=True, input_size=32):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, use_adaptive_masks=use_adaptive_masks, input_size=input_size)


def resnet1202(num_classes=10, use_adaptive_masks=True, input_size=32):
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes, use_adaptive_masks=use_adaptive_masks, input_size=input_size)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    # Test standard ResNets
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(f"Standard {net_name}")
            test(globals()[net_name](use_adaptive_masks=False))
            print()
            
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(f"Adaptive {net_name}")
            test(globals()[net_name](use_adaptive_masks=True))
            print()
    
    # Test adaptive ResNets
    print("="*50)
    print("Testing Adaptive ResNets:")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet20(num_classes=10, use_adaptive_masks=True, input_size=32).to(device)
    
    dummy = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        result = model(dummy)
        if isinstance(result, tuple):
            logits, masks = result
            print(f"Output logits shape: {logits.shape}")
            print(f"Number of masks returned: {len(masks)}")
            for i, m in enumerate(masks):
                print(f"  mask {i} shape = {m.shape}")
        else:
            print(f"Output logits shape: {result.shape}")
    
    # Count adaptive parameters
    print("\nTrainable parameters in shared AdaptiveBlocks:")
    total_adaptive_params = 0
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveBlock):
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_adaptive_params += param_count
            print(f"{name}: {param_count} parameters")
    print(f"Total adaptive parameters: {total_adaptive_params}")
    
    # Total model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params}")
    print(f"Adaptive params ratio: {total_adaptive_params/total_params:.4f}")