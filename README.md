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


googlenet-base-cifar10-e60 85.72%
googlenet-margin2-cifar10-e60 **87.80%** alpha 0.1
googlenet-margin4-cifar10-e60 **87.09%** alpha 0.1
googlenet-margin6-cifar10-e60 <u>**87.89%**</u> alpha 0.1


### VGG16 on MNIST

| Model                | Acc@1      | Margin | Alpha |
|----------------------|------------|--------|-------|
| vgg16-base-mnist     | 99.43%     | -      | -     |
| vgg16-margin2-mnist  | **99.45%** | 2      | 1     |
| vgg16-margin4-mnist  | **99.44%** | 4      | 1     |
| vgg16-margin6-mnist  | 99.41%     | 6      | 1     |
| vgg16-margin8-mnist  | **99.50%** | 8      | 1     |
| vgg16-margin10-mnist | <u>**99.51%**</u> | 10     | 1     |

### LeNet on MNIST

| Model           | Accuracy          | Margin | Alpha |
|-----------------|-------------------|--------|-------|
| LeNet           | 98.85%            | -      | -     |
| LeNet Margin 2  | **98.93%**        | 2      | 0.1   |
| LeNet Margin 4  | **98.96%**        | 4      | 0.1   |
| LeNet Margin 6  | <u>**99.01%**</u> | 6      | 0.05  |
| LeNet Margin 8  | **98.93%**        | 8      | 0.05  |
| LeNet Margin 10 | **98.98%**        | 10     | 0.025 |
