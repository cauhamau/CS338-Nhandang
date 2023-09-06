import torch
from torch import nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=6, padding_idx=0, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.padding_idx = padding_idx

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            print(true_dist)
            true_dist.fill_(self.smoothing / (self.cls - 2))
            print(true_dist)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            print(true_dist)
            true_dist[:, self.padding_idx] = 0
            print(true_dist)
            mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
            
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

pred = torch.tensor([[-0.1808, -0.7827, -0.4109, -0.0139,  0.1728,  0.1933],
        [-0.2600, -0.5728,  0.7359,  0.4318, -0.0337,  0.0297]])

target = torch.tensor([197, 193, 214, 191, 192,   2,   0])

print(target.data.unsqueeze(1))
# mask = torch.tensor([[False, False, False, False, False, False,  True]])

# LS = LabelSmoothingLoss()

# res = LS(pred, target)

# print(res)