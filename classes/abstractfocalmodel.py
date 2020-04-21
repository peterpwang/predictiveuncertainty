from .abstractmodel import AbstractImageClassificationModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Abstract focal model
class AbstractFocalModel(AbstractImageClassificationModel):
    
    # compile model
    def compile_model(self, net):
        print("Learning rate is set to ", self.learning_rate)
        optimizer = optim.SGD(net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        criterion = FocalLoss(self.focal_gamma).cuda(0)
        return optimizer, criterion


class FocalLoss(nn.Module):

    def __init__(self, focal_gamma_arg):
        self.focal_gamma = focal_gamma_arg
        super(FocalLoss, self).__init__()

    def forward(self, y_pred, y):
        alpha = 0.25
        epsilon = 1e-7

        y_pred = F.softmax(y_pred, dim=1)
        y_pred = y_pred[range(y_pred.shape[0]), y] + epsilon
        focal_loss =  -alpha * torch.pow(1.0 - y_pred, self.focal_gamma) * torch.log(epsilon + y_pred)
        #print("fl:", focal_loss)
        return focal_loss.mean()

