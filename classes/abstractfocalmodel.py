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
        criterion = focal_loss
        return optimizer, criterion


def focal_loss(y_pred, y):
    alpha = 0.25
    gamma = 2.0
    epsilon = 1e-7

    y_pred = F.softmax(y_pred, dim=1)
    y_hot = F.one_hot(y, num_classes=y_pred.shape[1])
    pt_plus = y_pred * y_hot
    pt_minus = (1.0 - y_hot) * (1.0 - y_pred)
    pt = pt_plus + pt_minus

    focal_loss =  -1.0 * alpha * torch.pow(1.0 - pt, gamma) * torch.log(epsilon + pt)
    #print("fl:", focal_loss.sum() / y.shape[0])
    return focal_loss.sum() / y.shape[0]
