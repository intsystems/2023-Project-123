import typing

import numpy as np
import torch
import torch.nn as nn


def get_negative_mask(batch_size):
    """
    For both images, takes the positions corresponding to itself and
    to its augmentation. Resulting mask is a matrix (2bs, 2bs) with 
    the main diagonal and 2 positive parallel sub-diagonals zeroed-out
    """
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def get_target_mask(target: torch.Tensor) -> torch.Tensor:
    """
    Generate bool matrix for equal targets
    """
    target_ = torch.cat([target, target], dim=0).view(-1, 1)  # (2bs,)
    mask = (target_ == target_.t().contiguous()) & (target_ != -1)  # (2bs, 2bs)
    return mask


def replace_pos_diagonals(full_matrix, pos_m):
    """
    Replace positive sub-diagonals in full_matrix shaped (2bs, 2bs) by pos_m vector
    """
    batch_size = full_matrix.shape[0] // 2

    new_matrix = full_matrix.clone()

    new_matrix[range(batch_size), range(batch_size, 2*batch_size)] = pos_m[:batch_size]
    new_matrix[range(batch_size, 2*batch_size), range(batch_size)] = pos_m[batch_size:]
    return new_matrix


class ContrastiveLossBase(nn.Module):
    def __init__(self, temperature, cuda, drop_fn):
        super().__init__()
        self.temperature: float = temperature
        self.cuda: bool = cuda
        self.drop_fn: bool = drop_fn

    def forward(self, out_1, out_2, target):
        batch_size = out_1.shape[0]

        ### neg score ###
        out = torch.cat([out_1, out_2], dim=0)  # (2bs, d)
        # scalar product of all pairs
        full = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)  # (2bs, 2bs)
        # drop false-negaive, if needed
        if self.drop_fn:
            mask = get_target_mask(target)
            if self.cuda:
                mask = mask.cuda()
            full = full.masked_fill(mask, 0.0)

        # keep negative pairs
        mask = get_negative_mask(batch_size)  # (2bs, 2bs)
        if self.cuda:
            mask = mask.cuda()
        neg = full.masked_select(mask).view(2 * batch_size, -1)  # (2bs, 2bs - 2)

        ### pos score ####
        # scalar product of 2 augmentations of 1 image
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)  # (bs,)
        pos = torch.cat([pos, pos], dim=0)  # (2bs,)

        return pos, neg

    def extra_repr(self):
        return "Temperature: {}\nCuda: {}\nDrop FN".format(self.temperature, self.cuda, self.drop_fn)


class ContrastiveLoss(ContrastiveLossBase):
    def forward(self, out_1, out_2, out_m, target):
        pos, neg = super().forward(out_1, out_2, target)
        Ng = neg.sum(dim=-1)  # (2bs)
        loss = (-torch.log(pos / (pos + Ng))).mean() # contrastive loss
        return loss


class DebiasedNegLoss(ContrastiveLossBase):
    def __init__(self, temperature, cuda, drop_fn, tau_plus):
        super().__init__(temperature, cuda, drop_fn)
        self.tau_plus = tau_plus

    def forward(self, out_1, out_2, out_m, target):
        pos, neg = super().forward(out_1, out_2, target)
        batch_size = out_1.shape[0]

        pos_m = [pos]
        for vec in out_m:
            pos_1 = torch.exp(torch.sum(out_1 * vec, dim=-1) / self.temperature)  # (bs,)
            pos_2 = torch.exp(torch.sum(out_2 * vec, dim=-1) / self.temperature) # (bs,)
            pos_new = torch.cat([pos_1, pos_2], dim=0)  # (2bs,)
            pos_m.append(pos_new)
        pos_m = torch.stack(pos_m, dim=0).mean(dim=0)

        N = batch_size * 2 - 2
        Ng = (-self.tau_plus * N * pos_m + neg.sum(dim=-1)) / (1 - self.tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.temperature))
        loss = (-torch.log(pos / (pos + Ng))).mean() # contrastive loss
        return loss

    def extra_repr(self):
        return "Tau plus: {}".format(self.tau_plus)


class DebiasedPosLoss(nn.Module):
    def __init__(self, temperature, cuda, drop_fn, tau_plus):
        super().__init__()
        self.temperature: float = temperature
        self.cuda: bool = cuda
        self.drop_fn: bool = drop_fn
        self.tau_plus: float = tau_plus

    def forward(self, out_1, out_2, out_m, target):
        batch_size = out_1.shape[0]
        N = batch_size * 2 - 2

        ### full score ###
        out = torch.cat([out_1, out_2], dim=0)  # (2bs, d)
        full = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)  # (2bs, 2bs)
        if self.drop_fn:
            mask = get_target_mask(target)
            if self.cuda:
                mask = mask.cuda()
            full = full.masked_fill(mask, 0.0)

        ### pos score ###
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)  # (bs,)
        pos = torch.cat([pos, pos], dim=0)  # (2bs,)
        pos_m = [pos] # for pos_grouping
        for vec in out_m: 
            pos_1 = torch.exp(torch.sum(out_1 * vec, dim=-1) / self.temperature)  # (bs,)
            pos_2 = torch.exp(torch.sum(out_2 * vec, dim=-1) / self.temperature) # (bs,)
            pos_new = torch.cat([pos_1, pos_2], dim=0)  # (2bs,)
            pos_m.append(pos_new)
        pos_m = torch.stack(pos_m, dim=0).mean(dim=0)

        full = replace_pos_diagonals(full, pos_m)
        p_estimate = full.mean(dim=-1) # (2bs,)

        ### neg score ###
        neg_mask = get_negative_mask(batch_size)  # (2bs, 2bs)
        if self.cuda:
            neg_mask = neg_mask.cuda()
        neg = full.masked_select(neg_mask).view(2 * batch_size, -1)  # (2bs, N)

        ### loss calculation ###
        tau_minus = 1 - self.tau_plus
        g = neg.mean(dim=-1)  # (2bs,)
        numerator = p_estimate - tau_minus * g  # (2bs,)
        denominator = p_estimate + (N * self.tau_plus - tau_minus) * g  # (2bs,)
        loss = (-torch.log(numerator / denominator)).mean()

        return loss

    def extra_repr(self):
        return "Temperature: {}\nCuda: {}\nDrop FN{}\nTau plus: {}".format(
            self.temperature, self.cuda, self.drop_fn, self.tau_plus)


class CombinationLoss(nn.Module):
    def __init__(self, temperature, cuda, drop_fn, tau_plus, alpha):
        super().__init__()
        self.temperature: float = temperature
        self.cuda: bool = cuda
        self.drop_fn: bool = drop_fn
        self.tau_plus: float = tau_plus
        self.alpha = alpha
        assert 0. <= alpha <= 1.0, "Alpha should be between 0 and 1"
        self.pos = DebiasedPosLoss(temperature, cuda, drop_fn, tau_plus)
        self.neg = DebiasedNegLoss(temperature, cuda, drop_fn, tau_plus)

    def forward(self, out_1, out_2, out_m, target):
        loss_neg = self.neg(out_1, out_2, out_m, target)
        loss_pos = self.pos(out_1, out_2, out_m, target)
        return self.alpha * loss_neg + (1. - self.alpha) * loss_pos

    def extra_repr(self):
        return "Temperature: {}\nCuda: {}\nDrop FN{}\nTau plus: {}\nAlpha: {}".format(
            self.temperature, self.cuda, self.drop_fn, self.tau_plus, self.alpha)


def get_loss(
    name: str,
    temperature: float,
    cuda: bool,
    tau_plus: float,
    drop_fn: bool,
    alpha: typing.Optional[float],
) -> nn.Module:

    if name == "Contrastive":
        return ContrastiveLoss(temperature, cuda, drop_fn)
    if name == "DebiasedNeg":
        return DebiasedNegLoss(temperature, cuda, drop_fn, tau_plus)
    if name == "DebiasedPos":
        return DebiasedPosLoss(temperature, cuda, drop_fn, tau_plus)
    if name == "Combination":
        return CombinationLoss(temperature, cuda, drop_fn, tau_plus, alpha)
    raise Exception("Unknown loss {}".format(name))
