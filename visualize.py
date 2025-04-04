import yaml
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_cka import CKA
import numpy as np
from model import ResNet50
from torchvision.models import vgg16
import seaborn as sns
from util import (
    get_kernel_weight_matrix,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visuallize kernel"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Architecture to use (default: resnet50)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"./checkpoint/resnet/resnet50-margin10-mnist-e15.pth",
        help="Path to the model checkpoint",
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar10"],
        default="mnist",
        help="Dataset to use for training ('mnist' or 'cifar10')",
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
            
    return args


def visualize_kernel_similarity(kernels, eps=1e-8):
    """
    Visualize the similarity map of a single set of convolutional kernels.
    
    Args:
        kernels (Tensor): A tensor of shape (num_kernels, size, size).
        margin (float): Margin for the hinge loss.
        eps (float): Small constant to avoid division by zero during normalization.
    """
    n, d, d2 = kernels.shape
    assert d == d2, "Each kernel must be a square matrix."

    # Normalize each kernel by its Frobenius norm.
    norms = kernels.norm(dim=(1, 2), p='fro', keepdim=True) + eps
    kernels_normed = kernels / norms

    # Compute the inverse of each normalized kernel.
    inv_kernels = torch.linalg.inv(kernels_normed)

    # Expand dims to perform pairwise multiplication:
    inv_expanded = inv_kernels.unsqueeze(1)
    kernels_expanded = kernels_normed.unsqueeze(0)
    pairwise_product = torch.matmul(inv_expanded, kernels_expanded)
    # Compute the difference from the identity matrix.
    I = torch.eye(d, device=kernels.device, dtype=kernels.dtype).view(1, 1, d, d)
    diff = I - pairwise_product

    # Compute the Frobenius norm of the difference for each pair.
    diff_norm = torch.linalg.norm(diff, dim=(2, 3))

    # Apply the hinge loss: only penalize pairs for which diff_norm is below margin.
    similarity = torch.clamp(diff_norm, min=0, max=25)

    # Plotting the similarity map.
    plt.figure(figsize=(8, 8))
    sns.heatmap(similarity.cpu().detach().numpy(), annot=False, cmap="coolwarm", cbar=True, square=True)
    plt.title("Kernel Similarity Map")
    plt.xlabel("Kernel Index")
    plt.ylabel("Kernel Index")
    plt.show()
    

def visualize_kernels(kernels):
    if isinstance(kernels, torch.Tensor):
        kernels = kernels.detach().cpu().numpy()
    
    num_kernels = kernels.shape[0]
    grid_size = int(num_kernels**0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        if i < num_kernels:
            kernel = kernels[i]
            ax.imshow(kernel, cmap='gray')
            ax.set_title(f'Kernel {i}')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    channels = 3 if args.dataset == "cifar10" else 1
    if args.model.lower() == "resnet50":
        model = ResNet50(num_classes=10, channels=channels).to(device)
    elif args.model.lower() == "vgg16":
        vgg = vgg16(weights=None)
        if channels == 1:
            vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        vgg.classifier[6] = nn.Linear(4096, 10)
        model = vgg.to(device)
    else:
        raise ValueError(f"Unsupported model architecture: {args.model}")
    
    kernel_list = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            filtered_kernels = get_kernel_weight_matrix(
                module.weight, ignore_sizes=[1]
            )
            if filtered_kernels is not None:
                kernel_list.append(filtered_kernels)
                
    checkpoint = torch.load(args.checkpoint,weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    visualize_kernel_similarity(kernel_list[0])
    # visualize_kernels(kernel_list[0])

    
if __name__ == "__main__":
    main()
