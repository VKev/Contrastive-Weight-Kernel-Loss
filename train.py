import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from util import (
    transform_mnist,
    transform_cifar10_train,
    transform_mnist_224,
    ContrastiveKernelLoss,
    get_kernel_weight_matrix,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from model import ResNet50, LeNet5
from torchvision import models
import random

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet50 on MNIST or CIFAR-10 with combined loss"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--alpha", type=float, default=1, help="Learning rate (default: 1)"
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=8,
        help="Margin for Contrastive Kernel Loss (default: 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Architecture to use (default: resnet50)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=r"",
        help="Path to checkpoint to resume from (default: '')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=r"full-layer",
        help="full-layer or random-sampling",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar10"],
        default="mnist",
        help="Dataset to use for training ('mnist' or 'cifar10')",
    )
    parser.add_argument(
    "--contrastive_kernel_loss",
    action="store_true",
    help="Use contrastive kernel loss (default: False)",
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
            
    return args

def select_random_kernels(kernel_list, k):
    """
    Selects k random kernels from each kernel layer in the kernel list.

    Args:
        kernel_list (list): List of tensors where each tensor has shape (N, H, W).
        k (int): Number of kernels to randomly select from each layer.

    Returns:
        list: A list of tensors where each tensor has shape (k, H, W).
    """
    selected_kernels = []

    for kernels in kernel_list:
        N, H, W = kernels.shape
        # Ensure k does not exceed the number of available kernels
        k = min(k, N)
        
        # Randomly select k indices from the range of available kernels
        selected_indices = random.sample(range(N), k)
        
        # Select the kernels using the chosen indices
        selected = kernels[selected_indices, :, :]
        selected_kernels.append(selected)

    return selected_kernels

def train_epoch(
    model, train_loader, optimizer, cls_criterion, kernel_loss_fn, device, args
):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass for classification.
        outputs = model(images)
        cls_loss = cls_criterion(outputs, labels)

        # Extract conv kernels and compute contrastive kernel loss.
        kernel_list = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                filtered_kernels = get_kernel_weight_matrix(
                    module.weight, ignore_sizes=[1]
                )
                if filtered_kernels is not None:
                    kernel_list.append(filtered_kernels)
        
        # kernel_list = [kernel_list[0], kernel_list[2], kernel_list[4], kernel_list[6]]
        if args.mode.lower() == 'random-sampling':
            kernel_list = select_random_kernels(kernel_list, k = 12)
            kernel_loss = (
            kernel_loss_fn(kernel_list)
            if kernel_list
            else torch.tensor(0.0, device=device)
            )
        else:
            kernel_loss = (
            kernel_loss_fn(kernel_list)
            if kernel_list
            else torch.tensor(0.0, device=device)
        )
        if args.contrastive_kernel_loss:
            total_loss = cls_loss + args.alpha*kernel_loss
        else:
            total_loss = cls_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        progress_bar.set_postfix(
            {
                "Total Loss": total_loss.item(),
                "Cls Loss": cls_loss.item(),
                "Kernel Loss": kernel_loss.item(),
            }
        )
    avg_loss = epoch_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, cls_criterion, kernel_loss_fn, device, args):
    model.eval()
    total_val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            cls_loss = cls_criterion(outputs, labels)

            kernel_list = []
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    filtered_kernels = get_kernel_weight_matrix(
                        module.weight, ignore_sizes=[1]
                    )
                    if filtered_kernels is not None:
                        kernel_list.append(filtered_kernels)
            if args.mode.lower() == 'random-sampling':
                kernel_list = select_random_kernels(kernel_list, k = 12)
                kernel_loss = (
                kernel_loss_fn(kernel_list)
                if kernel_list
                else torch.tensor(0.0, device=device)
                )
            else:
                kernel_loss = (
                kernel_loss_fn(kernel_list)
                if kernel_list
                else torch.tensor(0.0, device=device)
            )
            if args.contrastive_kernel_loss:
                loss = cls_loss + args.alpha*kernel_loss
            else:
                loss = cls_loss
            total_val_loss += loss.item()

            # Compute accuracy.
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_val_loss / len(val_loader)
    accuracy = correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    model.train()
    return avg_loss, accuracy


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "mnist":
        if args.model.lower() == "resnet50" or args.model.lower() == "vgg16" or args.model.lower() == "googlenet":
            transform = transform_mnist_224
        else:
            transform = transform_mnist
        full_dataset = datasets.MNIST(
            root="./data", train=True, transform=transform, download=True
        )
    elif args.dataset == "cifar10":
        full_dataset = datasets.CIFAR10(
            root="./data", train=True, transform=transform_cifar10_train, download=True
        )

    labels = [full_dataset[i][1] for i in range(len(full_dataset))]

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

    for train_idx, val_idx in split.split(range(len(full_dataset)), labels):
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    channels = 3 if args.dataset == "cifar10" else 1

    if args.model.lower() == "resnet50":
        model = ResNet50(num_classes=10, channels=channels).to(device)
    elif args.model.lower() == "vgg16":
        vgg = models.vgg16(weights=None)
        if channels == 1:
            vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        vgg.classifier[6] = nn.Linear(4096, 10)
        model = vgg.to(device)
    elif args.model.lower() == "lenet5":
        if channels == 1:
            model = LeNet5().to(device)
        else:
            raise ValueError(f"{args.model} only support input image 1 channel: {args.model}")
    elif args.model.lower() == "googlenet":
        googlenet = models.googlenet(weights=None, num_classes=10,aux_logits=False)
        if channels == 1:
            googlenet.conv1.conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = googlenet.to(device)
    else:
        raise ValueError(f"Unsupported model architecture: {args.model}")
    print(model)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"{name} is frozen")
    contrastive_loss_fn = ContrastiveKernelLoss(margin=args.margin)
    classification_criterion = nn.CrossEntropyLoss()
    if args.model.lower() == "googlenet" or args.model.lower() == "resnet50":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Resume from checkpoint if provided.
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            print(f"Resuming from epoch {start_epoch + 1}...")
        else:
            print(f"No checkpoint found at {args.resume}. Starting from scratch.")

    # Create checkpoint directory if it doesn't exist.
    checkpoint_dir = "checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    kernel_list = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            filtered_kernels = get_kernel_weight_matrix(
                module.weight, ignore_sizes=[1]
            )
            if filtered_kernels is not None:
                kernel_list.append(filtered_kernels)
    
    print("Conv layers num: ", len(kernel_list))
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            classification_criterion,
            contrastive_loss_fn,
            device,
            args,
        )
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")
        val_loss, val_accuracy = validate(
            model,
            val_loader,
            classification_criterion,
            contrastive_loss_fn,
            device,
            args,
        )

        # Save checkpoint for current epoch.
        if args.contrastive_kernel_loss:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"{args.model}-margin{int(args.margin)}-{args.dataset}-e{epoch+1}.pth"
            )
        else:
            checkpoint_path = os.path.join(
            checkpoint_dir, f"{args.model}-base-{args.dataset}-e{epoch+1}.pth"
            )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "args": vars(args),
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
