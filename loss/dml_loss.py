import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.autograd import Variable
import loss


class DMLLoss(nn.Module):
    def __init__(self):
        super(DMLLoss, self).__init__()
    
    def forward(self, feat1, feat2):
        """

        Args:
            feat1 (List[torch.tensor]): _description_
            feat2 (List[torch.tensor]): _description_

        Returns:
            reduced kl loss
        """
        feat1_counts = len(feat1)
        feat2_counts = len(feat2)
            
        kl_loss = []
        for idx, (i, j) in enumerate(zip(feat1, feat2)):
            temp1 = loss.utils.kd_loss_function(F.log_softmax(i, dim = 1), F.softmax(Variable(j), dim=1)) # mid --> final 
            temp2 = loss.utils.kd_loss_function(F.log_softmax(j, dim = 1), F.softmax(Variable(i), dim=1)) # final --> mid
            kl_loss.append(temp1+temp2)
            
        kl_loss = torch.tensor(kl_loss)
        kl_loss = torch.mean(kl_loss)
        # kl_loss *= -1

        return kl_loss