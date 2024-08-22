import torch
import torch.nn as nn
import torch.nn.functional as F


def F1_IoU_BCELoss(pred_masks, first_gt_mask):
    """
    binary cross entropy loss (iou loss) of the first frame for single sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
    """
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks)  # [bs*5, 1, 224, 224]

    indices = torch.tensor(list(range(0, len(pred_masks), 5)))
    indices = indices.cuda()
    first_pred = torch.index_select(
        pred_masks, dim=0, index=indices)  # [bs, 1, 224, 224]
    assert first_pred.requires_grad == True, "Error when indexing predited masks"
    if len(first_gt_mask.shape) == 5:
        first_gt_mask = first_gt_mask.squeeze(1)  # [bs, 1, 224, 224]
    # print('first_pred', first_pred.shape)
    # print('first_gt_mask', first_gt_mask.shape)
    first_bce_loss = nn.BCELoss()(first_pred, first_gt_mask)

    return first_bce_loss

def F5_Dice_loss(pred_mask, five_gt_masks, smooth=1.):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, h, w)
    """
    five_gt_masks = torch.sigmoid(five_gt_masks)
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)
    pred = pred_mask.contiguous()
    target = five_gt_masks.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()




class F5_IoU_BCELoss(nn.Module):
    def __init__(self, pic_num=5):
        super(F5_IoU_BCELoss, self).__init__()
        self.pic_num = pic_num
        self.mse = nn.MSELoss()
    def forward(self, x, five_gt_masks, gt_mask):
        # if len(gt_mask.shape) == 5:
        #     gt_mask = gt_mask.squeeze(1)  # [bs, 1, 224, 224]
        # fivemasks = five_gt_masks.detach()
        # bs = five_gt_masks.size(0) // self.pic_num
        # c = five_gt_masks.size(1)
        # h = five_gt_masks.size(2)
        # w = five_gt_masks.size(3)
        # tensor1 = fivemasks.view(bs, self.pic_num, c, h, w)
        # tensor1[:, 0, :, :, :] = gt_mask
        # # print('gt', gt_mask.tolist())
        # new_five_gt_masks = tensor1.view(bs * self.pic_num, c, h, w)
        # # print('t')
        # # print('new_', new_five_gt_masks.tolist())
        loss = self.mse(x, five_gt_masks)
        return loss

def F1_Dice_loss(pred_masks, first_gt_mask):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs*5, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, 1, h, w)
    """
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks)

    indices = torch.tensor(list(range(0, len(pred_masks), 5)))
    indices = indices.cuda()
    first_pred = torch.index_select(
        pred_masks, dim=0, index=indices)  # [bs, 1, 224, 224]
    assert first_pred.requires_grad == True, "Error when indexing predited masks"
    if len(first_gt_mask.shape) == 5:
        first_gt_mask = first_gt_mask.squeeze(1)  # [bs, 1, 224, 224]
        # print('first_gt_mask', first_gt_mask.shape)
    pred_mask = first_pred.flatten(1)
    # print('pred_mask', pred_mask.shape)
    gt_mask = first_gt_mask.flatten(1)
    # print('gt_mask', gt_mask.shape)
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    return loss.mean()



def IouSemanticAwareLoss(pred_masks, gt_mask, weight_dict, loss_type='bce', **kwargs):
    total_loss = 0
    loss_dict = {}
    bce = F5_IoU_BCELoss()
    if loss_type == 'bce':
        loss_func = F1_IoU_BCELoss
    elif loss_type == 'dice':
        loss_func = F1_Dice_loss
    else:
        raise ValueError

    iou_loss = loss_func(pred_masks, gt_mask)
    total_loss += weight_dict['iou_loss'] * iou_loss
    loss_dict['iou_loss'] = weight_dict['iou_loss'] * iou_loss.item()

    return total_loss, loss_dict
