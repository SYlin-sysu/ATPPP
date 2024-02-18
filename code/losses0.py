import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DICELossMultiClass(nn.Module):

    def __init__(self,batch_size = 16):
        super(DICELossMultiClass, self).__init__()
        self.batch_size = batch_size
    def forward(self, output, mask):
        #classes_weight = [0.25,0.25,0.25,0.05,0.2]
        num_classes = output.size(1)
        size1 = output.size(2)
        size2 = output.size(3)
        dice_eso = 0.0
        #dice_eso = torch.Tensor([0.0 for a in range(self.batch_size)]).float().to(device = output.device)
        for i in range(0,num_classes):
            probs = torch.squeeze(output[:, i, :, :], 1)
            mask2 = torch.squeeze(mask[:, i, :, :], 1)

            num = probs * mask2
            num = torch.sum(num, 2)
            num = torch.sum(num, 1)

            # print( num )

            den1 = probs * probs
            # print(den1.size())
            den1 = torch.sum(den1, 2)
            den1 = torch.sum(den1, 1)

            # print(den1.size())

            den2 = mask2 * mask2
            # print(den2.size())
            den2 = torch.sum(den2, 2)
            den2 = torch.sum(den2, 1)

            # print(den2.size())
            eps = 1e-5

            dice = (2.* (num )+eps) / (den1 + den2 + eps)
            # dice_eso = dice[:, 1:]
            dice_eso += dice
            #print(dice)
        loss = 1 - torch.sum(dice_eso) / (dice_eso.size(0)*num_classes)
        return loss


class DICELoss(nn.Module):

    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, output, mask):

        probs = torch.squeeze(output, 1)
        mask = torch.squeeze(mask, 1)

        intersection = probs * mask
        intersection = torch.sum(intersection, 2)
        intersection = torch.sum(intersection, 1)

        den1 = probs * probs
        den1 = torch.sum(den1, 2)
        den1 = torch.sum(den1, 1)

        den2 = mask * mask
        den2 = torch.sum(den2, 2)
        den2 = torch.sum(den2, 1)

        eps = 1e-5
        dice = (2.* intersection + eps) / (den1 + den2 + eps)
        # dice_eso = dice[:, 1:]
        dice_eso = dice

        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss
def CE_Loss(inputs, target, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.NLLLoss(ignore_index=num_classes)(F.log_softmax(temp_inputs, dim = -1), temp_target)
    return CE_loss
def Dice_Loss2(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    tp = torch.sum(temp_target * temp_inputs, dim=[0, 1])
    fp = torch.sum(temp_inputs, dim=[0, 1]) - tp
    fn = torch.sum(temp_target, dim=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss
