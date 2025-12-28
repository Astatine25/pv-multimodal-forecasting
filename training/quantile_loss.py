import torch

def quantile_loss(preds, target, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, :, i]
        loss = torch.max(q * errors, (q - 1) * errors)
        losses.append(loss.unsqueeze(-1))
    return torch.mean(torch.sum(torch.cat(losses, dim=-1), dim=-1))
