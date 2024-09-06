import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_flow_loss(pred, scale):
    gt = torch.zeros_like(pred)
    loss_map = (pred - gt) ** 2
    loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
    return loss_map.mean() * scale


class mLoss(nn.Module):
    def __init__(self):
        super(mLoss, self).__init__()

    def forward(self, I_pred, I_gt, L_pred, L_gt, smvs, smv_residuals):
        """
        L_main = L1(I - I_gt) + L1(L - L_gt)
        L_sep = 0
        L_lmv_residual = L1reg(lmv_residual) + L1reg(Sobel(lmv_residual))
        """
        L_main = abs(I_pred - I_gt).mean().sum()
        L_main += abs(L_pred - L_gt).mean().sum()


        L_sep = 0

        L_lmv_residual = 0
        for residuals in smv_residuals:
            for idx, resi in enumerate(residuals):
                L_lmv_residual += zero_flow_loss(resi, 8 / (2 ** max(1, idx)))
        for smv_list in smvs:
            for idx, smv in enumerate(smv_list):
                L_lmv_residual += zero_flow_loss(smv, 8 / (2 ** max(1, idx)))

        return L_main + L_sep + L_lmv_residual
