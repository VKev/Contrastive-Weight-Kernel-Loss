import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets, transforms, models
import yaml
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import warnings
import shutil
import wandb
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')

from util import (
    transform_mnist,
    transform_cifar10_train,
    transform_cifar10_test,
    transform_mnist_224,
    ContrastiveKernelLoss,
    get_kernel_weight_matrix,
    transform_imagenet_train,
    transform_imagenet_val,
)
from model import ResNet50, LeNet5, AdaptResNet50
from model.diversified.div_resnet import DiversifiedResNet50

warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    print(f"Setting random seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.num_workers = min(16, os.cpu_count() // 2)

    def prepare_data(self):
        if self.dataset == "mnist":
            datasets.MNIST(root="./data", train=True, download=True)
            datasets.MNIST(root="./data", train=False, download=True)
        elif self.dataset == "cifar10":
            datasets.CIFAR10(root="./data", train=True, download=True)
            datasets.CIFAR10(root="./data", train=False, download=True)

    def setup(self, stage=None):
        split = 0.9
        if self.dataset == "mnist":
            train_transform = (
                transform_mnist_224
                if self.args.model.lower()
                in ["resnet50_diversified", "resnet50", "vgg16", "googlenet"]
                else transform_mnist
            )
            test_transform = (
                transform_mnist_224
                if self.args.model.lower()
                in ["resnet50_diversified", "resnet50", "vgg16", "googlenet"]
                else transform_mnist
            )

            full_dataset = datasets.MNIST(
                root="./data", train=True, transform=train_transform
            )
            self.test_dataset = datasets.MNIST(
                root="./data", train=False, transform=test_transform
            )

        elif self.dataset == "cifar10":
            full_dataset = datasets.CIFAR10(
                root="./data", train=True, transform=transform_cifar10_train
            )
            self.test_dataset = datasets.CIFAR10(
                root="./data", train=False, transform=transform_cifar10_test
            )

        elif self.dataset == "cifar100":
            full_dataset = datasets.CIFAR100(
                root="./data",
                train=True,
                transform=transform_cifar10_train,
                download=True,
            )
            self.test_dataset = datasets.CIFAR100(
                root="./data",
                train=False,
                transform=transform_cifar10_test,
                download=True,
            )

        elif self.dataset == "imagenet1k":
            full_dataset = datasets.ImageNet(
                root=r"D:\AI\Dataset\ImageNet1k",
                split="train",
                transform=transform_imagenet_train,
            )
            self.test_dataset = datasets.ImageNet(
                root=r"D:\AI\Dataset\ImageNet1k",
                split="val",
                transform=transform_imagenet_val,
            )
            split = 0.98

        labels = np.array(full_dataset.targets)
        train_idx, val_idx = train_test_split(
            np.arange(len(full_dataset)),
            test_size=1.0 - split,
            stratify=labels,
            random_state=42,
        )

        self.train_dataset = Subset(full_dataset, train_idx)
        self.val_dataset = Subset(full_dataset, val_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        common_kwargs = dict(
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        val_loader = DataLoader(self.val_dataset, **common_kwargs)
        test_loader = DataLoader(self.test_dataset, **common_kwargs)
        return [val_loader, test_loader]

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args

        channels = 3 if args.dataset in ["cifar10", "cifar100", "imagenet1k"] else 1
        self.num_classes = 10
        if args.dataset in ["imagenet1k"]:
            self.num_classes = 1000
        elif args.dataset in ["cifar100"]:
            self.num_classes = 100

        if args.model.lower() == "resnet50":
            self.model = ResNet50(num_classes=self.num_classes, channels=channels)

        elif args.model.lower() == "resnet50_adapt":
            # AdaptResNet50 returns (logits, masks)
            self.model = AdaptResNet50(
                num_classes=self.num_classes,
                channels=channels,
                hidden_ratio=0.25,
                input_size=32 if args.dataset in ["cifar10", "cifar100"] else 224,
            )

        elif args.model.lower() == "resnet50_diversified":
            self.model = DiversifiedResNet50(
                num_classes=self.num_classes, channels=channels
            )

        elif args.model.lower() == "vgg16":
            self.model = models.vgg16(weights=None)
            if channels == 1:
                self.model.features[0] = nn.Conv2d(
                    1, 64, kernel_size=3, stride=1, padding=1
                )
            self.model.classifier[6] = nn.Linear(4096, self.num_classes)

        elif args.model.lower() == "lenet5":
            if channels == 1:
                self.model = LeNet5()
            else:
                raise ValueError(f"{args.model} only supports 1-channel input")

        elif args.model.lower() == "googlenet":
            self.model = models.googlenet(
                weights=None, num_classes=self.num_classes, aux_logits=False
            )
            if channels == 1:
                self.model.conv1.conv = nn.Conv2d(
                    1, 64, kernel_size=7, stride=2, padding=3, bias=False
                )

        else:
            raise ValueError(f"Unsupported model: {args.model}")

        self.cls_criterion = nn.CrossEntropyLoss()
        self.kernel_loss_fn = ContrastiveKernelLoss(margin=args.margin)

        # New: track mask‐penalty weight
        self.mask_penalty_weight = args.mask_penalty_weight

        print(self.model)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.args.model.lower() in ["googlenet"]:
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.args.lr,
                weight_decay=1e-2,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        elif self.args.model.lower() in [
            "resnet50_diversified",
            "resnet50",
            "resnet50_adapt",
        ]:
            optimizer = optim.SGD(
                self.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4
            )

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/epoch_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        else:
            optimizer = optim.Adam(self.parameters(), lr=self.args.lr)

        return optimizer

    def _get_kernel_list(self):
        kernel_list = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                filtered_kernels = get_kernel_weight_matrix(
                    module.weight, ignore_sizes=[1]
                )
                if filtered_kernels is not None:
                    kernel_list.append(filtered_kernels)
        return kernel_list

    def _select_random_kernels(self, kernel_list, k=12):
        selected = []
        for kernels in kernel_list:
            N = kernels.shape[0]
            k = min(k, N)
            selected_indices = random.sample(range(N), k)
            selected.append(kernels[selected_indices])
        return selected

    def training_step(self, batch, batch_idx):
        x, y = batch

        # If using AdaptResNet50, forward() returns (logits, masks)
        if self.args.model.lower() == "resnet50_adapt":
            logits, masks = self.model(x)
        else:
            logits = self.model(x)
            masks = None

        # 1) Classification loss
        cls_loss = self.cls_criterion(logits, y)

        # 2) (Optional) Contrastive kernel loss
        if self.hparams["contrastive_kernel_loss"]:
            kernel_list = self._get_kernel_list()
            if self.hparams["mode"].lower() == "random-sampling":
                kernel_list = self._select_random_kernels(kernel_list, k=24)
            kernel_loss = (
                self.kernel_loss_fn(kernel_list)
                if kernel_list
                else torch.tensor(0.0, device=self.device)
            )
        else:
            kernel_loss = torch.tensor(0.0, device=self.device)

        total_loss = cls_loss
        if self.hparams["contrastive_kernel_loss"]:
            total_loss = total_loss + self.hparams["alpha"] * kernel_loss

        if self.args.model.lower() == "resnet50_adapt":
            per_mask_losses = []
            for mask in masks:
                penalty_mask = F.relu(1 - mask)
                per_mask_mse = (penalty_mask).mean()
                per_mask_losses.append(per_mask_mse)

            if per_mask_losses:
                mask_penalty = torch.stack(per_mask_losses).mean()
            else:
                mask_penalty = torch.tensor(0.0, device=self.device)

            if self.current_epoch < 28:
                total_loss = total_loss + self.mask_penalty_weight * mask_penalty
            self.log(
                "train/mask_penalty", mask_penalty, on_step=True, on_epoch=False
            )

        # 4) Logging & return
        if not hasattr(self, "_train_losses"):
            self._train_losses = {"total_loss": [], "cls_loss": [], "kernel_loss": []}

        self._train_losses["total_loss"].append(total_loss)

        self.log("train/loss", total_loss, on_step=True, on_epoch=False)
        if self.hparams["contrastive_kernel_loss"]:
            self.log("train/cls_loss", cls_loss, on_step=True, on_epoch=False)
            self.log("train/kernel_loss", kernel_loss, on_step=True, on_epoch=False)

        optimizer = self.optimizers()  # returns a list, take the first optimizer
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", current_lr, on_step=True, on_epoch=False)

        return total_loss

    def on_after_backward(self):
        total_norm = torch.norm(
            torch.stack(
                [
                    p.grad.detach().norm(2)
                    for p in self.parameters()
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        self.log(
            "grad_norm/global", total_norm, on_step=True, on_epoch=False, prog_bar=False
        )

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.detach().norm(2)
                self.log(
                    f"grad_norm/{name}",
                    grad_norm,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                )

    def on_train_epoch_end(self):
        avg_total_loss = torch.stack(self._train_losses["total_loss"]).mean()
        self.log("train/epoch_loss", avg_total_loss, prog_bar=True)
        self._train_losses = {"total_loss": []}

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y = batch
        logits = self.model(x) if self.args.model.lower() != "resnet50_adapt" else self.model(x)[0]

        cls_loss = self.cls_criterion(logits, y)
        if self.hparams["contrastive_kernel_loss"]:
            kernel_list = self._get_kernel_list()
            if self.hparams["mode"].lower() == "random-sampling":
                kernel_list = self._select_random_kernels(kernel_list, k=24)
            kernel_loss = (
                self.kernel_loss_fn(kernel_list)
                if kernel_list
                else torch.tensor(0.0, device=self.device)
            )
        else:
            kernel_loss = torch.tensor(0.0, device=self.device)

        total_loss = cls_loss + (
            self.hparams["alpha"] * kernel_loss
            if self.hparams["contrastive_kernel_loss"]
            else 0.0
        )

        preds = torch.argmax(logits, dim=1)
        split = "val" if dataloader_idx == 0 else "test"
        if not hasattr(self, "_stats"):
            self._stats = {s: {"loss": [], "preds": [], "labels": []} for s in ("val", "test")}

        self._stats[split]["loss"].append(total_loss)
        self._stats[split]["preds"].append(preds)
        self._stats[split]["labels"].append(y)

        self.log(f"{split}/loss", total_loss, on_step=True, on_epoch=False)
        return {"loss": total_loss, "split": split}

    def on_validation_epoch_end(self):
        for split, buf in self._stats.items():
            if not buf["loss"]:
                continue
            avg_loss = torch.stack(buf["loss"]).mean()
            acc = (torch.cat(buf["preds"]) == torch.cat(buf["labels"])).float().mean()
            self.log(f"{split}/epoch_loss", avg_loss, prog_bar=False)
            self.log(f"{split}/epoch_acc", acc, prog_bar=True)
            if split == "test":
                self.log("test_acc", acc, prog_bar=False)
            for k in buf:
                buf[k].clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x) if self.args.model.lower() != "resnet50_adapt" else self.model(x)[0]
        loss = self.cls_criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "acc": acc}


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with PyTorch Lightning")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1, help="Alpha parameter")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--margin", type=float, default=8, help="Margin for contrastive loss")
    parser.add_argument("--model", type=str, default="resnet50", help="Model architecture")
    parser.add_argument("--mode", type=str, default="full-layer", help="full-layer or random-sampling")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist", help="Dataset to use")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every n epochs")
    parser.add_argument("--contrastive_kernel_loss", action="store_true", help="Use contrastive kernel loss")
    parser.add_argument("--wandb", action="store_true", help="Use WandB logging")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument(
        "--mask_penalty_weight",
        type=float,
        default=1,
        help="Weight for mask‐penalty term (only used if model=resnet50_adapt)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)
            print(f"Overriding {key}: {getattr(args, key)} -> {value}")

    return args


def build_model(args):
    print("Building model")
    if args.resume:
        print("Resuming from checkpoint:", args.resume)
        return Model.load_from_checkpoint(args.resume, args=args)
    return Model(args)


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        hparams = checkpoint["hyper_parameters"]
        if "wandb_id" in hparams:
            args.wandb_id = hparams["wandb_id"]
        for key, value in hparams.items():
            if key != "resume" and hasattr(args, key) and getattr(args, key) != value:
                print(f"Overriding {key}: {getattr(args, key)} -> {value}")
                setattr(args, key, value)
        checkpoint = None
    else:
        args.wandb_id = wandb.util.generate_id()

    model = build_model(args)
    dm = DataModule(args)

    callbacks = []
    ModelCheckpoint.CHECKPOINT_NAME_LAST = (
        f"{args.model}-{args.dataset}"
        + ("-ckl" if args.contrastive_kernel_loss else "")
        + "-{epoch}-{test_acc:.4f}-last"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoint",
        filename=f"{args.model}-{args.dataset}"
        + ("-ckl" if args.contrastive_kernel_loss else "")
        + "-{epoch}-{test_acc:.4f}",
        monitor="test/epoch_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    if args.early_stopping:
        early_stopping = EarlyStopping(
            monitor="val/epoch_acc", patience=args.patience, mode="max", verbose=True
        )
        callbacks.append(early_stopping)

    logger = None
    if args.wandb:
        logger = WandbLogger(
            project="beyond-contrastive",
            name=f"{args.model}-{args.dataset}",
            id=args.wandb_id,
            resume="allow",
            offline=False,
        )
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        devices=args.device,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        deterministic=False,
    )

    shutil.rmtree("./wandb", ignore_errors=True)
    shutil.rmtree("./lightning_logs", ignore_errors=True)

    trainer.fit(
        model, datamodule=dm, ckpt_path=args.resume if args.resume else None
    )


if __name__ == "__main__":
    main()
