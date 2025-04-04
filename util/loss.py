import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveKernelLoss(nn.Module):
    def __init__(self, margin=1.0, eps=1e-8):
        """
        margin (float): Margin for the hinge loss.
        eps (float): Small constant to avoid division by zero when normalizing.
        """
        super().__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, kernels_list):
        """
        Computes the contrastive kernel loss for a list of convolutional kernels.
        
        Args:
            kernels_list (list of Tensors): Each tensor has shape (n, d, d) representing 
                the kernels of one conv layer (after squeezing out any singleton channel dimension).
                
        Returns:
            A scalar loss value computed as the average of the losses over all conv layers.
        """
        losses = []
        for kernels in kernels_list:
            n, d, d2 = kernels.shape
            assert d == d2, "Each kernel must be a square matrix."
            
            # Normalize each kernel by its Frobenius norm.
            norms = kernels.norm(dim=(1, 2), p='fro', keepdim=True) + self.eps
            kernels_normed = kernels / norms  # shape: (n, d, d)
            
            # Compute the inverse of each normalized kernel.
            inv_kernels = torch.linalg.inv(kernels_normed)  # shape: (n, d, d)
            
            # Expand dims to perform pairwise multiplication:
            # For each pair (i, j): compute inv(kernel_i) @ kernel_j.
            inv_expanded = inv_kernels.unsqueeze(1)   # shape: (n, 1, d, d)
            kernels_expanded = kernels_normed.unsqueeze(0)  # shape: (1, n, d, d)
            
            # Compute inv(kernel_i) @ kernel_j for all pairs
            pairwise_inv_AB = torch.matmul(inv_expanded, kernels_expanded)  # shape: (n, n, d, d)
            # Compute inv(kernel_j) @ kernel_i for all pairs
            pairwise_inv_BA = torch.matmul(inv_kernels.unsqueeze(0), kernels_normed.unsqueeze(1))  # shape: (n, n, d, d)
            
            # Create the pairwise difference matrix by subtracting from identity
            I = torch.eye(d, device=kernels.device, dtype=kernels.dtype).view(1, 1, d, d)
            diff_AB = I - pairwise_inv_AB  # shape: (n, n, d, d)
            diff_BA = I - pairwise_inv_BA  # shape: (n, n, d, d)
            
            # Compute the Frobenius norm of the differences for each pair.
            diff_norm_AB = torch.linalg.norm(diff_AB, dim=(2, 3))  # shape: (n, n)
            diff_norm_BA = torch.linalg.norm(diff_BA, dim=(2, 3))  # shape: (n, n)
            
            # Combine the two differences (for lower triangle and upper triangle).
            combined_diff_norm = torch.tril(diff_norm_AB) + torch.triu(diff_norm_BA)  # Combine lower and upper triangles

            # Apply the hinge loss: only penalize pairs for which diff_norm is below margin.
            loss_matrix = F.relu(self.margin - combined_diff_norm)
            
            # Zero out the diagonal (self-comparisons are ignored).
            mask = torch.ones(n, n, dtype=torch.bool, device=kernels.device)
            mask.fill_diagonal_(False)
            loss_matrix = loss_matrix * mask
            
            num_pairs = n * (n - 1)
            losses.append(loss_matrix.sum() / num_pairs)
        
        # Average the losses from all conv layers.
        total_loss = sum(losses) / len(losses) if losses else torch.tensor(0.0, device=kernels.device)
        return total_loss


# class ContrastiveKernelLoss(nn.Module):
#     def __init__(self, margin=1.0, eps=1e-8):
#         """
#         margin (float): Margin for the hinge loss.
#         eps (float): Small constant to avoid division by zero when normalizing.
#         """
#         super().__init__()
#         self.margin = margin
#         self.eps = eps

#     def forward(self, kernels_list):
#         """
#         Computes the contrastive kernel loss for a list of convolutional kernels.
        
#         Args:
#             kernels_list (list of Tensors): Each tensor has shape (n, d, d) representing 
#                 the kernels of one conv layer (after squeezing out any singleton channel dimension).
                
#         Returns:
#             A scalar loss value computed as the average of the losses over all conv layers.
#         """
#         losses = []
#         for kernels in kernels_list:
#             n, d, d2 = kernels.shape
#             assert d == d2, "Each kernel must be a square matrix."
            
#             # Normalize each kernel by its Frobenius norm.
#             norms = kernels.norm(dim=(1, 2), p='fro', keepdim=True) + self.eps
#             kernels_normed = kernels / norms  # shape: (n, d, d)
            
#             # Compute the inverse of each normalized kernel.
#             inv_kernels = torch.linalg.inv(kernels_normed)  # shape: (n, d, d)
            
#             # Expand dims to perform pairwise multiplication:
#             # For each pair (i, j): compute inv(kernel_i) @ kernel_j.
#             inv_expanded = inv_kernels.unsqueeze(1)   # shape: (n, 1, d, d)
#             kernels_expanded = kernels_normed.unsqueeze(0)  # shape: (1, n, d, d)
#             pairwise_product = torch.matmul(inv_expanded, kernels_expanded)  # shape: (n, n, d, d)
            
#             # Compute the difference from the identity: I - inv(kernel_i) @ kernel_j.
#             I = torch.eye(d, device=kernels.device, dtype=kernels.dtype).view(1, 1, d, d)
#             diff = I - pairwise_product  # shape: (n, n, d, d)
            
#             # Compute the Frobenius norm of the difference for each pair.
#             diff_norm = torch.linalg.norm(diff, dim=(2, 3))  # shape: (n, n)
            
#             # Apply the hinge loss: only penalize pairs for which diff_norm is below margin.
#             loss_matrix = F.relu(self.margin - diff_norm)
            
#             # Zero out the diagonal (self-comparisons are ignored).
#             mask = torch.ones(n, n, dtype=torch.bool, device=kernels.device)
#             mask.fill_diagonal_(False)
#             loss_matrix = loss_matrix * mask
            
#             # Average the loss for this conv layer.
#             losses.append(loss_matrix.mean())
        
#         # Average the losses from all conv layers.
#         total_loss = sum(losses) / len(losses) if losses else torch.tensor(0.0, device=kernels.device)
#         return total_loss

# A simple model with 3 convolutional layers.
class SimpleConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        # For simplicity, each conv has one input channel so that weight shape is (out_channels, 1, k, k)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # kernels: (16, 1, 3, 3)
        self.conv2 = nn.Conv2d(1, 32, kernel_size=7, padding=3)  # kernels: (32, 1, 7, 7)
        self.conv3 = nn.Conv2d(1, 8, kernel_size=5, padding=2)   # kernels: (8, 1, 5, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

if __name__ == "__main__":
    # Instantiate the model.
    model = SimpleConvModel()
    
    # Extract kernels from each convolution.
    # Since the conv layers have weight shape (out_channels, 1, k, k), we squeeze the channel dimension.
    kernels1 = model.conv1.weight.squeeze(1)  # shape: (16, 3, 3)
    kernels2 = model.conv2.weight.squeeze(1)  # shape: (32, 7, 7)
    kernels3 = model.conv3.weight.squeeze(1)  # shape: (8, 5, 5)
    
    # Put all conv kernels into a list.
    kernels_list = [kernels1, kernels2, kernels3]
    
    # Instantiate the loss function with a desired margin.
    loss_fn = ContrastiveKernelLoss(margin=10)
    
    # Compute the loss.
    loss_value = loss_fn(kernels_list)
    print("Contrastive Kernel Loss:", loss_value.item())
