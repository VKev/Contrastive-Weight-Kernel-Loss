import argparse
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score, f1_score
from util import transform_mnist, transform_cifar10, transform_mnist_224
from model import ResNet50, LeNet5
from torchvision import models


def parse_args():
    parser = argparse.ArgumentParser(description="Test model on MNIST or CIFAR-10 dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing (default: 128)")
    parser.add_argument("--checkpoint", type=str, default=r"./checkpoint/vgg16-margin4-mnist-e15.pth", help="Path to the model checkpoint")
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist", help="Dataset to use for testing ('mnist' or 'cifar10')")
    parser.add_argument("--model", type=str, default="resnet50", help="Architecture to use (default: resnet50)")
    return parser.parse_args()


def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    top5_correct = sum([1 if label in pred else 0 for label, pred in zip(labels, torch.topk(outputs, 5).indices)])
    recall = recall_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
    f1 = f1_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
    return predicted, top5_correct, recall, f1


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    top5_correct = 0
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    all_labels = []
    all_predictions = []

    progress_bar = tqdm(test_loader, desc="Testing", unit="batch")
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            predicted, top5_corr, recall, f1 = calculate_metrics(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            top5_correct += top5_corr

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            progress_bar.set_postfix({"Batch Loss": loss.item()})

    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    top5_accuracy = 100.0 * top5_correct / total
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Top-5 Accuracy: {top5_accuracy:.2f}%, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


def load_model(args, channels, device):
    if args.model.lower() == "resnet50":
        return ResNet50(num_classes=10, channels=channels).to(device)
    elif args.model.lower() == "vgg16":
        vgg = models.vgg16(weights=None)
        if channels == 1:
            vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        vgg.classifier[6] = nn.Linear(4096, 10)
        return vgg.to(device)
    elif args.model.lower() == "lenet5":
        if channels == 1:
            return LeNet5().to(device)
        else:
            raise ValueError(f"{args.model} only supports 1 channel input")
    elif args.model.lower() == "googlenet":
        googlenet = models.googlenet(weights=None, num_classes=10, aux_logits=False)
        if channels == 1:
            googlenet.conv1.conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return googlenet.to(device)
    else:
        raise ValueError(f"Unsupported model architecture: {args.model}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transform_mnist_224 if args.model.lower() in ["resnet50", "vgg16", "googlenet"] else transform_mnist
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True) if args.dataset == "mnist" else datasets.CIFAR10(root="./data", train=False, transform=transform_cifar10, download=True)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    channels = 3 if args.dataset == "cifar10" else 1
    model = load_model(args, channels, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from '{args.checkpoint}'")

    test(model, test_loader, device)


if __name__ == "__main__":
    main()
