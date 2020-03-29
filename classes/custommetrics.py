import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


# NLL(loss)
class NLL(Metric):

    def __init__(self, output_transform=lambda x: x, device=None):
        self._sum_nll = 0.0
        super(NLL, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_nll = 0.0
        super(NLL, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        nll = F.nll_loss(F.log_softmax(y_pred, dim=0), y)
        self._sum_nll += nll/y.shape[0]

    def compute(self):
        return self._sum_nll


# Correct NLL(loss)
class CorrectNLL(Metric):

    def __init__(self, output_transform=lambda x: x, device=None):
        self._sum_nll = 0.0
        super(CorrectNLL, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_nll = 0.0
        super(CorrectNLL, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        indices = torch.argmax(y_pred, dim=1)
        mask = torch.eq(indices, y).view(-1)
        y_pred = y_pred[mask]
        y = y[mask]

        if y.shape[0]>0:
            nll = F.nll_loss(F.log_softmax(y_pred, dim=0), y)
            self._sum_nll += nll/y.shape[0]

# Entropy
#>>> input = torch.randn(3, 5, requires_grad=True)
#>>> target = torch.randint(5, (3,), dtype=torch.int64)
#>>> loss = F.cross_entropy(input, target)
#>>> loss.backward()

    def compute(self):
        return self._sum_nll


# Incorrect NLL(loss)
class IncorrectNLL(Metric):

    def __init__(self, output_transform=lambda x: x, device=None):
        self._sum_nll = 0.0
        super(IncorrectNLL, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_nll = 0.0
        super(IncorrectNLL, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        indices = torch.argmax(y_pred, dim=1)
        mask = torch.ne(indices, y).view(-1)
        y_pred = y_pred[mask]
        y = y[mask]

        if y.shape[0]>0:
            nll = F.nll_loss(F.log_softmax(y_pred, dim=0), y)
            self._sum_nll += nll/y.shape[0]

    def compute(self):
        return self._sum_nll



# Correct Crossentropy
class CorrectCrossEntropy(Metric):

    def __init__(self, output_transform=lambda x: x, device=None):
        self._sum_crossentropy = 0.0
        super(CorrectCrossEntropy, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_cross_entropy = 0.0
        super(CorrectCrossEntropy, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        indices = torch.argmax(y_pred, dim=1)
        mask = torch.eq(indices, y).view(-1)
        y_pred = y_pred[mask]
        y = y[mask]

        if y.shape[0]>0:
            entropy = F.cross_entropy(y_pred, y)
            self._sum_cross_entropy += entropy/y.shape[0]

    def compute(self):
        return self._sum_cross_entropy


# Incorrect Entropy
class IncorrectCrossEntropy(Metric):

    def __init__(self, output_transform=lambda x: x, device=None):
        self._sum_cross_entropy = 0.0
        super(IncorrectCrossEntropy, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_cross_entropy = 0.0
        super(IncorrectCrossEntropy, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        indices = torch.argmax(y_pred, dim=1)
        mask = torch.ne(indices, y).view(-1)
        y_pred = y_pred[mask]
        y = y[mask]

        if y.shape[0]>0:
            entropy = F.cross_entropy(y_pred, y)
            self._sum_cross_entropy += entropy/y.shape[0]

    def compute(self):
        return self._sum_cross_entropy

