import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import random
from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)

class L1LossSonar(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', binary = 0.1):
        super(L1LossSonar, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.binary = binary

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # 将预测值和目标值二值化
        pred_binary = torch.where(pred > self.binary, torch.ones_like(pred), torch.zeros_like(pred))
        target_binary = torch.where(target > self.binary, torch.ones_like(target), torch.zeros_like(target))
        
        # 在GPU上执行操作
        if torch.cuda.is_available():
            pred_binary = pred_binary.cuda()
            target_binary = target_binary.cuda()

        # 计算 L1 损失
        loss_shaddle = l1_loss(pred_binary, target_binary, reduction=self.reduction)
        loss_l1 = l1_loss(pred, target, weight, reduction=self.reduction)
        
        return self.loss_weight * (loss_shaddle+loss_l1)
    
class L1LossChannel(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        channel (int, optional): The channel to compute loss on. If None, all channels are used. Default: None.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', channel=None):
        super(L1LossChannel, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.channel = channel

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # 如果指定了通道，则只取该通道进行计算
        if self.channel is not None:
            pred = pred[:, self.channel:self.channel+1]
            target = target[:, self.channel:self.channel+1]
            if weight is not None:
                weight = weight[:, self.channel:self.channel+1]
                
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)
    
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class L1LossSr(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1LossSr, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # print(pred['hq'].shape, target['hq'].shape)
        # print(pred['sr'].shape, target['sr'].shape)
                # 将预测值和目标值二值化

        hlshadow = self.shadow(pred['hq'], target['hq'], weight)
        hlloss = self.loss_weight * l1_loss(pred['hq'], target['hq'], weight, reduction=self.reduction)

        if pred['sr'] is not None:
            srshadow = self.shadow(pred['sr'], target['sr'], weight)
            srloss = self.loss_weight * l1_loss(pred['sr'], target['sr'], weight, reduction=self.reduction)
        else:
            srshadow = 0
            srloss = 0

        return 0.5 * hlloss + 0.25 * (srloss) + 0.25 * (hlshadow + srshadow)

    def shadow(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # 将预测值和目标值二值化
        pred_binary = torch.where(pred > 0.1, torch.ones_like(pred), torch.zeros_like(pred))
        target_binary = torch.where(target > 0.1, torch.ones_like(target), torch.zeros_like(target))
        
        # 在GPU上执行操作
        if torch.cuda.is_available():
            pred_binary = pred_binary.cuda()
            target_binary = target_binary.cuda()

        return self.loss_weight * l1_loss(pred_binary, target_binary, weight, reduction=self.reduction)

class L1Lossweight(nn.Module):
    """L1 (mean absolute error, MAE) loss with smooth weighting across channels.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sigma (float): Standard deviation for the Gaussian weighting function.
            A smaller sigma makes the weighting more peaked. Default: 1.0.
        invert (bool): If True, inverts the Gaussian to have higher weights at the ends. Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sigma=2.0, weight= [1.5, 1], invert=False):
        super(L1Lossweight, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: ["none", "mean", "sum"]')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sigma = sigma
        self.invert = invert
        self.max_weight = weight[0]  # Adjust as needed
        self.min_weight = weight[1]  # Adjust as needed

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # Compute element-wise L1 loss
        loss = torch.abs(pred - target)  # Shape (N, C, H, W)

        # Get the number of channels
        C = pred.size(1)

        # Create channel weights using a Gaussian function
        channel_positions = torch.arange(C, dtype=pred.dtype, device=pred.device)
        mid_channel = (C - 1) / 2.0

        # Gaussian function parameters
        sigma = self.sigma  # Standard deviation

        # Compute Gaussian weights
        if not self.invert:
            # Higher weights in the middle
            channel_weights = torch.exp(-0.5 * ((channel_positions - mid_channel) / sigma) ** 2)
        else:
            # Higher weights at the ends
            channel_weights = 1 - torch.exp(-0.5 * ((channel_positions - mid_channel) / sigma) ** 2)

        channel_weights = self.min_weight + (self.max_weight - self.min_weight) * (
            (channel_weights - channel_weights.min()) / (channel_weights.max() - channel_weights.min())
        )

        # Reshape channel weights for broadcasting
        channel_weights = channel_weights.view(1, C, 1, 1)

        # Apply channel weights to the loss
        loss = loss * channel_weights

        # Apply element-wise weight if provided
        if weight is not None:
            loss = loss * weight

        # Apply reduction
        if self.reduction == 'mean':
            return self.loss_weight * loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * loss.sum()
        elif self.reduction == 'max':
            # First, average the loss across the spatial dimensions (H, W)
            loss_per_channel = loss.mean(dim=[2, 3])  # Shape (N, C)
            # Then, find the maximum loss across the channels for each sample
            max_loss_per_sample = loss_per_channel.max(dim=1).values  # Shape (N)
            # Finally, return the mean of the maximum loss across all samples
            return self.loss_weight * max_loss_per_sample.mean()
        else:  # 'none'
            return self.loss_weight * loss

# class L1LossForVideoFramesweight(nn.Module):
#     """用于连续视频帧的L1损失函数，包含帧间的时间一致性损失。

#     Args:
#         loss_weight (float): L1损失的权重。默认值：1.0。
#         reduction (str): 指定对输出应用的归约方式。支持选项为'none' | 'mean' | 'sum'。默认值：'mean'。
#         sigma (float): 高斯权重函数的标准差。较小的sigma会使权重更集中。默认值：2.0。
#         weight (list of float): [max_weight, min_weight]，用于调整权重范围。默认值：[1.5, 1.0]。
#         invert (bool): 如果为True，则反转高斯函数，使得权重在两端较高。默认值：False。
#         temporal_weight (float): 时间一致性损失的权重。默认值：1.0。
#     """

#     def __init__(self, loss_weight=1.0, reduction='mean', sigma=2.0, weight=[1.5, 1.0], invert=False, temporal_weight=1.0):
#         super(L1LossForVideoFramesweight, self).__init__()
#         if reduction not in ['none', 'mean', 'sum']:
#             raise ValueError(f'Unsupported reduction mode: {reduction}. '
#                              f'Supported ones are: ["none", "mean", "sum"]')

#         self.loss_weight = loss_weight
#         self.reduction = reduction
#         self.sigma = sigma
#         self.invert = invert
#         self.max_weight = weight[0]
#         self.min_weight = weight[1]
#         self.temporal_weight = temporal_weight

#     def forward(self, pred, target, weight=None, **kwargs):
#         """
#         Args:
#             pred (Tensor): 形状为 (N, C, H, W)。预测张量，其中C表示连续帧的数量。
#             target (Tensor): 形状为 (N, C, H, W)。真实值张量。
#             weight (Tensor, optional): 形状为 (N, C, H, W)。元素级权重。默认值：None。
#         """
#         # 计算每帧的L1损失
#         per_frame_loss = torch.abs(pred - target)  # 形状 (N, C, H, W)

#         # 获取帧的数量（通道数）
#         C = pred.size(1)

#         # 使用高斯函数创建帧权重
#         frame_positions = torch.arange(C, dtype=pred.dtype, device=pred.device)
#         mid_frame = (C - 1) / 2.0

#         # 计算高斯权重
#         if not self.invert:
#             # 中间帧权重较高
#             frame_weights = torch.exp(-0.5 * ((frame_positions - mid_frame) / self.sigma) ** 2)
#         else:
#             # 两端帧权重较高
#             frame_weights = 1 - torch.exp(-0.5 * ((frame_positions - mid_frame) / self.sigma) ** 2)

#         # 将权重归一化到指定范围
#         frame_weights = self.min_weight + (self.max_weight - self.min_weight) * (
#             (frame_weights - frame_weights.min()) / (frame_weights.max() - frame_weights.min())
#         )

#         # 调整权重形状以进行广播
#         frame_weights = frame_weights.view(1, C, 1, 1)

#         # 应用帧权重到每帧的损失
#         per_frame_loss = per_frame_loss * frame_weights

#         # 如果提供了元素级权重，应用它
#         if weight is not None:
#             per_frame_loss = per_frame_loss * weight

#         # 计算时间一致性损失
#         if C > 1:
#             # 计算预测和真实值的时间差分
#             delta_pred = pred[:, 1:, :, :] - pred[:, :-1, :, :]  # 形状 (N, C-1, H, W)
#             delta_target = target[:, 1:, :, :] - target[:, :-1, :, :]  # 形状 (N, C-1, H, W)

#             # 计算时间一致性损失
#             temporal_loss = torch.abs(delta_pred - delta_target)  # 形状 (N, C-1, H, W)

#             # 创建时间差分的权重
#             temporal_frame_positions = torch.arange(C - 1, dtype=pred.dtype, device=pred.device)
#             mid_frame_temporal = (C - 2) / 2.0

#             # 计算时间差分的高斯权重
#             if not self.invert:
#                 temporal_frame_weights = torch.exp(-0.5 * ((temporal_frame_positions - mid_frame_temporal) / self.sigma) ** 2)
#             else:
#                 temporal_frame_weights = 1 - torch.exp(-0.5 * ((temporal_frame_positions - mid_frame_temporal) / self.sigma) ** 2)

#             # 归一化权重到指定范围
#             temporal_frame_weights = self.min_weight + (self.max_weight - self.min_weight) * (
#                 (temporal_frame_weights - temporal_frame_weights.min()) / (temporal_frame_weights.max() - temporal_frame_weights.min())
#             )

#             # 调整权重形状以进行广播
#             temporal_frame_weights = temporal_frame_weights.view(1, C - 1, 1, 1)

#             # 应用权重到时间一致性损失
#             temporal_loss = temporal_loss * temporal_frame_weights

#             # 对时间一致性损失进行归约
#             if self.reduction == 'mean':
#                 temporal_loss = temporal_loss.mean()
#             elif self.reduction == 'sum':
#                 temporal_loss = temporal_loss.sum()
#             elif self.reduction == 'max':
#                 temporal_loss = temporal_loss.max()
#             else:  # 'none'
#                 pass  # 保持原样

#             # 对每帧损失进行归约
#             if self.reduction == 'mean':
#                 per_frame_loss = per_frame_loss.mean()
#             elif self.reduction == 'sum':
#                 per_frame_loss = per_frame_loss.sum()
#             else:  # 'none'
#                 pass  # 保持原样

#             # 合并每帧损失和时间一致性损失
#             total_loss = self.loss_weight * per_frame_loss + self.temporal_weight * temporal_loss

#             return total_loss
#         else:
#             # 如果只有一帧，只计算每帧损失
#             if self.reduction == 'mean':
#                 per_frame_loss = per_frame_loss.mean()
#             elif self.reduction == 'sum':
#                 per_frame_loss = per_frame_loss.sum()
#             else:  # 'none'
#                 pass  # 保持原样

#             return self.loss_weight * per_frame_loss


class L1LossForVideoFrames(nn.Module):
    """用于连续视频帧的L1损失函数，包含帧间的时间一致性损失。

    Args:
        loss_weight (float): L1损失的权重。默认值：1.0。
        reduction (str): 指定对输出应用的归约方式。支持选项为'none' | 'mean' | 'sum'。默认值：'mean'。
        sigma (float): 高斯权重函数的标准差。较小的sigma会使权重更集中。默认值：2.0。
        weight (list of float): [max_weight, min_weight]，用于调整权重范围。默认值：[1.5, 1.0]。
        invert (bool): 如果为True，则反转高斯函数，使得权重在两端较高。默认值：False。
        temporal_weight (float): 时间一致性损失的权重。默认值：1.0。
    """

    def __init__(self, l1loss_weight=0.64, reduction='mean', sigma=2.0, weight=[1.5, 1.0], invert=False, temporal_weight=0.36, binary=0.1):
        super(L1LossForVideoFrames, self).__init__()
        if reduction not in ['none', 'mean', 'sum', 'max', 'mix']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: ["none", "mean", "sum", "max", "mix"]')

        self.l1loss_weight = l1loss_weight
        self.reduction = reduction
        self.sigma = sigma
        self.invert = invert
        self.max_weight = weight[0]
        self.min_weight = weight[1]
        self.temporal_weight = temporal_weight
        self.binary = binary

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): 形状为 (N, C, H, W)。预测张量，其中C表示连续帧的数量。
            target (Tensor): 形状为 (N, C, H, W)。真实值张量。
            weight (Tensor, optional): 形状为 (N, C, H, W)。元素级权重。默认值：None。
        """
        if self.reduction == "mix":
            if random.random() < 0.64:
                self.reduction = "mean"
            else:
                self.reduction = "max"
                
        pred_binary = torch.where(pred > self.binary, torch.ones_like(pred), torch.zeros_like(pred))
        target_binary = torch.where(target > self.binary, torch.ones_like(target), torch.zeros_like(target))
        
        # 在GPU上执行操作
        if torch.cuda.is_available():
            pred_binary = pred_binary.cuda()
            target_binary = target_binary.cuda()

        # 计算每帧的L1损失
        per_frame_loss = torch.abs(pred - target) + torch.abs(pred_binary - target_binary) # 形状 (N, C, H, W)

        # 获取帧的数量（通道数）
        C = pred.size(1)

        # 如果提供了元素级权重，应用它
        if weight is not None:
            per_frame_loss = per_frame_loss * weight

        # 计算时间一致性损失
        if C > 1:
            # 计算预测和真实值的时间差分
            delta_pred = pred[:, 1:, :, :] - pred[:, :-1, :, :]  # 形状 (N, C-1, H, W)
            delta_target = target[:, 1:, :, :] - target[:, :-1, :, :]  # 形状 (N, C-1, H, W)

            # 计算时间一致性损失
            temporal_loss = torch.abs(delta_pred - delta_target)  # 形状 (N, C-1, H, W)

            # 对时间一致性损失进行归约
            if self.reduction == 'mean':
                temporal_loss = temporal_loss.mean()
            elif self.reduction == 'sum':
                temporal_loss = temporal_loss.sum()
            elif self.reduction == 'max':
                # First, average the loss across the spatial dimensions (H, W)
                loss_per_channel = temporal_loss.mean(dim=[2, 3])  # Shape (N, C)
                # Then, find the maximum loss across the channels for each sample
                max_loss_per_sample = loss_per_channel.max(dim=1).values  # Shape (N)
                # Finally, return the mean of the maximum loss across all samples
                temporal_loss = max_loss_per_sample.mean()
            else:  # 'none'
                pass  # 保持原样

            # 对每帧损失进行归约
            if self.reduction == 'mean':
                per_frame_loss = per_frame_loss.mean()
            elif self.reduction == 'sum':
                per_frame_loss = per_frame_loss.sum()
            elif self.reduction == 'max':
                # First, average the loss across the spatial dimensions (H, W)
                loss_per_channel = per_frame_loss.mean(dim=[2, 3])  # Shape (N, C)
                # Then, find the maximum loss across the channels for each sample
                max_loss_per_sample = loss_per_channel.max(dim=1).values  # Shape (N)
                # Finally, return the mean of the maximum loss across all samples
                per_frame_loss = max_loss_per_sample.mean()
            else:  # 'none'
                pass  # 保持原样

            # 合并每帧损失和时间一致性损失
            total_loss = self.l1loss_weight * per_frame_loss + self.temporal_weight * temporal_loss

            return total_loss
        else:
            # 如果只有一帧，只计算每帧损失
            if self.reduction == 'mean':
                per_frame_loss = per_frame_loss.mean()
            elif self.reduction == 'sum':
                per_frame_loss = per_frame_loss.sum()
            elif self.reduction == 'max':
                # First, average the loss across the spatial dimensions (H, W)
                loss_per_channel = per_frame_loss.mean(dim=[2, 3])  # Shape (N, C)
                # Then, find the maximum loss across the channels for each sample
                max_loss_per_sample = loss_per_channel.max(dim=1).values  # Shape (N)
                # Finally, return the mean of the maximum loss across all samples
                per_frame_loss = max_loss_per_sample.mean()
            else:  # 'none'
                pass  # 保持原样

            return self.l1loss_weight * per_frame_loss
        
        
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
