import torch
import torch.nn as nn
import torch.nn.functional as F

class DiversifiedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, k, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 similarity_mode="cosine", div_factor = 2):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.k = max(k, out_channels//div_factor)
        self.push_weights = nn.Parameter(torch.zeros(self.k, self.k))
        
        if similarity_mode not in ["cosine", "identity"]:
            raise ValueError(f"similarity_mode must be 'cosine' or 'identity', got {similarity_mode}")
        self.similarity_mode = similarity_mode
        
    def forward(self, x):
        orig     = self.weight                             # [out_c, in_c, kH, kW]
        selected = orig[:self.k]        
        S        = selected.view(self.k, -1)               # flatten to [k, d]
        # 1) Cosine similarity + learnable push-weights
        normed     = F.normalize(S, dim=1, eps=1e-8)       # [k, d] :contentReference[oaicite:0]{index=0}
        sim        = normed @ normed.t()  
        P          = F.relu(self.push_weights) + F.relu(sim)       # [k, k] :contentReference[oaicite:2]{index=2}
        mask = torch.ones_like(P) - torch.eye(self.k, device=P.device, dtype=P.dtype)
        P = P * mask
        # 2) Raw pairwise diffs and sum across j
        diffs      = S.unsqueeze(1) - S.unsqueeze(0)       # [k, k, d]
        summed     = diffs.sum(dim=1)                      # [k, d] :contentReference[oaicite:3]{index=3}

        # 3) Normalize the combined direction
        delta_flat = F.normalize(summed, dim=1, eps=1e-8)  # [k, d] :contentReference[oaicite:4]{index=4}

        # 4) Weight by total push magnitude
        row_sums   = P.sum(dim=1, keepdim=True)            # [k, 1]
        delta_flat = row_sums * delta_flat                # [k, d]

        # 5) Reshape back & apply
        modified   = delta_flat.view_as(selected) + selected  # [k, in_c, kH, kW]

        # 6) Reassemble full weight & conv
        new_weight = torch.cat([modified, orig[self.k:]], dim=0)
        return F.conv2d(x, new_weight, self.bias,
                        self.stride, self.padding,
                        self.dilation, self.groups)

        

if __name__ == "__main__":
    model_cosine = DiversifiedConv2d(3, 64, kernel_size=3, k=2, similarity_mode="cosine")
    
    x = torch.randn(1, 3, 32, 32)
    output_cosine = model_cosine(x)
    loss  = output_cosine.mean()
    loss.backward()
    print("Output shape (cosine):", output_cosine.shape)
    print(model_cosine.push_weights.grad ) 
