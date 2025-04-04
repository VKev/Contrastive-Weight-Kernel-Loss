# Contrastive Kernel Loss

This repository implements the **Contrastive Kernel Loss** (CKL) for training models on datasets like CIFAR-10 and MNIST. The CKL is designed to improve the performance of models by leveraging kernel-based contrastive loss functions.

## Formula for Contrastive Kernel Loss

The Contrastive Kernel Loss is defined as:

$$
\mathcal{L}_{CK} = \frac{1}{L} \sum_{\ell=1}^{L} \frac{1}{N_{\ell}(N_{\ell} - 1)} \sum_{i,j=1, i \neq j}^{N_{\ell}} \text{max} \left( 0, \text{margin} - \| I_{\ell} - \left( K_{\ell}^{(i)} \right)^{-1} K_{\ell}^{(j)} \|_F \right)
$$

Where:

- L is the number of Layers,
- N is the number of Kernels in Layer
- I is the identity matrix for Layer
- K is Kernel weight matrix
- margin is a hyperparameter that controls the margin between positive and negative samples.

## CIFAR-10 Results

All model train in 60 epoch

| Model                     | Margin | Acc@1      | Acc@5 |
| ------------------------- | ------ | ---------- | ----- |
| resnet50-base-cifar10     | -      | 85.41%     | N/A   |
| resnet50-margin2-cifar10  | 2      | **85.71%** | N/A   |
| resnet50-margin4-cifar10  | 4      | **86.62%** | N/A   |
| resnet50-margin6-cifar10  | 6      | **85.68%** | N/A   |
| resnet50-margin8-cifar10  | 8      | 85.32%     | N/A   |
| resnet50-margin10-cifar10 | 10     | **85.46%** | N/A   |

googlenet-base-cifar10-e60 89.69%

## MNIST Results

All model train in 15 epoch

| Model                   | Margin | Acc@1      |     | Model                | Margin | Acc@1      |
| ----------------------- | ------ | ---------- | --- | -------------------- | ------ | ---------- |
| resnet50-base-mnist     | -      | 99.35%     |     | vgg16-base-mnist     | -      | 99.25%     |
| resnet50-margin2-mnist  | 2      | **99.40%** |     | vgg16-margin2-mnist  | 2      | **99.54%** |
| resnet50-margin4-mnist  | 4      | 99.24%     |     | vgg16-margin4-mnist  | 4      | **99.39%** |
| resnet50-margin6-mnist  | 6      | 99.31%     |     | vgg16-margin6-mnist  | 6      | **99.46%** |
| resnet50-margin8-mnist  | 8      | **99.50%** |     | vgg16-margin8-mnist  | 8      | **99.51%** |
| resnet50-margin10-mnist | 10     | **99.42%** |     | vgg16-margin10-mnist | 10     | **99.30%** |
