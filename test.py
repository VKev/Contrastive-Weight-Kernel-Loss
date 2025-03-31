import argparse
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import transform_mnist, transform_cifar10
from model import ResNet50
from torchvision.models import vgg16

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test ResNet50 on MNIST or CIFAR-10 dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"./checkpoint/vgg16-margin2-mnist-e15.pth",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar10"],
        default="mnist",
        help="Dataset to use for testing ('mnist' or 'cifar10')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Architecture to use (default: resnet50)",
    )
    return parser.parse_args()


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0

    progress_bar = tqdm(test_loader, desc="Testing", unit="batch")
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({"Batch Loss": loss.item()})

    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select dataset based on args.dataset
    if args.dataset == "mnist":
        test_dataset = datasets.MNIST(
            root="./data", train=False, transform=transform_mnist, download=True
        )
    elif args.dataset == "cifar10":
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, transform=transform_cifar10, download=True
        )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
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

    checkpoint = torch.load(args.checkpoint,weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from '{args.checkpoint}'")

    test(model, test_loader, device)


if __name__ == "__main__":
    main()
