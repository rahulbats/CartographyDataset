import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        """
        Focal Loss for addressing class imbalance.
        gamma: Focusing parameter for hard examples.
        alpha: Balancing parameter for positive examples.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, labels):
        """
        Compute Focal Loss.
        logits: Model output logits of shape (batch_size, num_classes).
        labels: Ground truth labels of shape (batch_size).
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        probs_true = probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)

        # Compute Focal Loss
        focal_weight = self.alpha * (1 - probs_true) ** self.gamma
        loss = -focal_weight * torch.log(probs_true + 1e-12)  # Add epsilon for numerical stability
        return loss.mean()
