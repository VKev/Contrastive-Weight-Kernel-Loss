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
from model import ResNet50, LeNet5
from model.diversified.div_resnet import DiversifiedResNet50
from model.adapt_resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202

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
    torch.use_deterministic_algorithms(True)
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

        elif args.model.lower() == "resnet20":
            self.model = resnet20(
                num_classes=self.num_classes,
                use_adaptive_masks=False,
                input_size=32 if args.dataset in ["cifar10", "cifar100"] else 224,
            )

        elif args.model.lower() == "resnet32":
            self.model = resnet32(
                num_classes=self.num_classes,
                use_adaptive_masks=False,
                input_size=32 if args.dataset in ["cifar10", "cifar100"] else 224,
            )

        elif args.model.lower() == "resnet44":
            self.model = resnet44(
                num_classes=self.num_classes,
                use_adaptive_masks=False,
                input_size=32 if args.dataset in ["cifar10", "cifar100"] else 224,
            )

        elif args.model.lower() == "resnet56":
            self.model = resnet56(
                num_classes=self.num_classes,
                use_adaptive_masks=False,
                input_size=32 if args.dataset in ["cifar10", "cifar100"] else 224,
            )

        elif args.model.lower() == "resnet110":
            self.model = resnet110(
                num_classes=self.num_classes,
                use_adaptive_masks=False,
                input_size=32 if args.dataset in ["cifar10", "cifar100"] else 224,
            )

        elif args.model.lower() == "resnet1202":
            self.model = resnet1202(
                num_classes=self.num_classes,
                use_adaptive_masks=False,
                input_size=32 if args.dataset in ["cifar10", "cifar100"] else 224,
            )

        elif args.model.lower() == "resnet20_adapt":
            self.model = resnet20(
                num_classes=self.num_classes,
                use_adaptive_masks=True,
                input_size=32 if args.dataset in ["cifar10", "cifar100"] else 224,
            )

        elif args.model.lower() == "resnet32_adapt":
            self.model = resnet32(
                num_classes=self.num_classes,
                use_adaptive_masks=True,
                input_size=32 if args.dataset in ["cifar10", "cifar100"] else 224,
            )

        elif args.model.lower() == "resnet44_adapt":
            self.model = resnet44(
                num_classes=self.num_classes,
                use_adaptive_masks=True,
                input_size=32 if args.dataset in ["cifar10", "cifar100"] else 224,
            )

        elif args.model.lower() == "resnet56_adapt":
            self.model = resnet56(
                num_classes=self.num_classes,
                use_adaptive_masks=True,
                input_size=32 if args.dataset in ["cifar10", "cifar100"] else 224,
            )

        elif args.model.lower() == "resnet110_adapt":
            self.model = resnet110(
                num_classes=self.num_classes,
                use_adaptive_masks=True,
                input_size=32 if args.dataset in ["cifar10", "cifar100"] else 224,
            )

        elif args.model.lower() == "resnet1202_adapt":
            self.model = resnet1202(
                num_classes=self.num_classes,
                use_adaptive_masks=True,
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
            "resnet20",
            "resnet32",
            "resnet44",
            "resnet56",
            "resnet110",
            "resnet1202",
            "resnet20_adapt",
            "resnet32_adapt", 
            "resnet44_adapt",
            "resnet56_adapt",
            "resnet110_adapt",
            "resnet1202_adapt",
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
                    "monitor": "train/loss_epoch",
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

        # If using any adaptive ResNet, forward() returns (logits, masks)
        adaptive_models = ["resnet20_adapt", "resnet32_adapt", 
                          "resnet44_adapt", "resnet56_adapt", "resnet110_adapt", "resnet1202_adapt"]
        if self.args.model.lower() in adaptive_models:
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

        # 3) (Optional) Mask penalty for adaptive models only
        if self.args.model.lower() in adaptive_models and masks is not None:
            per_mask_losses = []
            for i, mask in enumerate(masks):
                penalty_mask = F.relu(1 - mask)
                per_mask_mse = (penalty_mask).mean()
                self.log(
                    f"loss/mask_penalty_{i}", per_mask_mse, on_step=True, on_epoch=False
                )
                per_mask_losses.append(per_mask_mse)

            if per_mask_losses:
                mask_penalty = torch.stack(per_mask_losses).mean()
            else:
                mask_penalty = torch.tensor(0.0, device=self.device)

            if self.current_epoch < 35:
                total_loss = total_loss + self.mask_penalty_weight * mask_penalty
            self.log(
                "train/mask_penalty", mask_penalty, on_step=True, on_epoch=False
            )

        # 4) Compute accuracy and log metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        
        if self.hparams["contrastive_kernel_loss"]:
            self.log("train/cls_loss", cls_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/kernel_loss", kernel_loss, on_step=True, on_epoch=True, prog_bar=False)

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



    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y = batch
        adaptive_models = ["resnet20_adapt", "resnet32_adapt", 
                          "resnet44_adapt", "resnet56_adapt", "resnet110_adapt", "resnet1202_adapt"]
        if self.args.model.lower() in adaptive_models:
            logits, _ = self.model(x)  # Ignore masks in validation
        else:
            logits = self.model(x)

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
        acc = (preds == y).float().mean()
        
        if dataloader_idx == 0:  # Validation
            self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        else:  # Test
            self.log("test/loss", total_loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False) 
            self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)  # For checkpoint monitoring
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, dataloader_idx=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with PyTorch Lightning")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1, help="Alpha parameter")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--margin", type=float, default=8, help="Margin for contrastive loss")
    parser.add_argument("--model", type=str, default="resnet50", help="Model architecture (resnet50, resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202, resnet20_adapt, resnet32_adapt, resnet44_adapt, resnet56_adapt, resnet110_adapt, resnet1202_adapt, etc.)")
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
        monitor="test_acc",
        mode="max",
        save_top_k=-1,
        every_n_epochs=1,
    )
    callbacks.append(checkpoint_callback)

    if args.early_stopping:
        early_stopping = EarlyStopping(
            monitor="val/acc", patience=args.patience, mode="max", verbose=True
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
    trainer_kwargs = dict(
        max_epochs=args.num_epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        devices=args.device,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )

    if torch.cuda.device_count() > 1:
        trainer_kwargs.update(
            strategy="ddp",
            sync_batchnorm=True,
            deterministic=True,
        )
    else:
        trainer_kwargs["deterministic"] = True

    trainer = pl.Trainer(**trainer_kwargs)

    shutil.rmtree("./wandb", ignore_errors=True)
    shutil.rmtree("./lightning_logs", ignore_errors=True)

    trainer.fit(
        model, datamodule=dm, ckpt_path=args.resume if args.resume else None
    )


if __name__ == "__main__":
    main()
