import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from util import (
    transform_mnist,
    transform_cifar10,
    ContrastiveKernelLoss,
    get_kernel_weight_matrix,
)
from model import ResNet50


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet50 on MNIST or CIFAR-10 with combined loss"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
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
    return parser.parse_args()


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

        kernel_loss = (
            kernel_loss_fn(kernel_list)
            if kernel_list
            else torch.tensor(0.0, device=device)
        )
        if args.contrastive_kernel_loss:
            total_loss = cls_loss + kernel_loss
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
            kernel_loss = (
                kernel_loss_fn(kernel_list)
                if kernel_list
                else torch.tensor(0.0, device=device)
            )
            if args.contrastive_kernel_loss:
                loss = cls_loss + kernel_loss
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

    # Select dataset based on args.dataset
    if args.dataset == "mnist":
        full_dataset = datasets.MNIST(
            root="./data", train=True, transform=transform_mnist, download=True
        )
    elif args.dataset == "cifar10":
        full_dataset = datasets.CIFAR10(
            root="./data", train=True, transform=transform_cifar10, download=True
        )

    dataset_size = len(full_dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Determine the number of channels based on dataset
    channels = 3 if args.dataset == "cifar10" else 1

    # Model initialization
    model = ResNet50(num_classes=10, channels=channels).to(device)
    print(model)

    contrastive_loss_fn = ContrastiveKernelLoss(margin=args.margin)
    classification_criterion = nn.CrossEntropyLoss()
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
