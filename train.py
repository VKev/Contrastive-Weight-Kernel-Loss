import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from util import transform, ContrastiveKernelLoss, get_kernel_weight_matrix
from model import ResNet50

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 on MNIST with combined loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--margin", type=float, default=10, help="Margin for Contrastive Kernel Loss (default: 10)")
    return parser.parse_args()

def train_epoch(model, train_loader, optimizer, cls_criterion, kernel_loss_fn, device):
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
                filtered_kernels = get_kernel_weight_matrix(module.weight, ignore_sizes=[1])
                if filtered_kernels is not None:
                    kernel_list.append(filtered_kernels)
        
        kernel_loss = kernel_loss_fn(kernel_list) if kernel_list else torch.tensor(0.0, device=device)
        total_loss = cls_loss + kernel_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
        progress_bar.set_postfix({
            "Total Loss": total_loss.item(),
            "Cls Loss": cls_loss.item(),
            "Kernel Loss": kernel_loss.item()
        })
    avg_loss = epoch_loss / len(train_loader)
    return avg_loss

def validate(model, val_loader, cls_criterion, kernel_loss_fn, device):
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
                    filtered_kernels = get_kernel_weight_matrix(module.weight, ignore_sizes=[1])
                    if filtered_kernels is not None:
                        kernel_list.append(filtered_kernels)
            kernel_loss = kernel_loss_fn(kernel_list) if kernel_list else torch.tensor(0.0, device=device)
            loss = cls_loss + kernel_loss
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
    
    # Load full MNIST training dataset.
    full_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

    # subset_size = int(0.08 * len(full_dataset))
    # indices = torch.randperm(len(full_dataset))[:subset_size]
    # full_dataset = torch.utils.data.Subset(full_dataset, indices)

    dataset_size = len(full_dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # MNIST has 10 classes.
    model = ResNet50(num_classes=10, channels=1).to(device)
    
    contrastive_loss_fn = ContrastiveKernelLoss(margin=args.margin)
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create checkpoint directory if it doesn't exist.
    checkpoint_dir = "checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, classification_criterion, contrastive_loss_fn, device)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")
        val_loss, val_accuracy = validate(model, val_loader, classification_criterion, contrastive_loss_fn, device)
        
        # Save checkpoint for current epoch.
        checkpoint_path = os.path.join(checkpoint_dir, f"resnet50_epoch{epoch+1}.pth")
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "args": vars(args)
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
