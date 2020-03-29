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
        self._count_nll = 0
        super(NLL, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_nll = 0.0
        self._count_nll = 0
        super(NLL, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        nll = F.nll_loss(F.log_softmax(y_pred, dim=0), y)
        self._sum_nll += nll
        self._count_nll += y.shape[0]

    def compute(self):
        return self._sum_nll / self._count_nll


# Correct NLL(loss)
class CorrectNLL(Metric):

    def __init__(self, output_transform=lambda x: x, device=None):
        self._sum_nll = 0.0
        self._count_nll = 0
        super(CorrectNLL, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_nll = 0.0
        self._count_nll = 0
        super(CorrectNLL, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        indices = torch.argmax(y_pred, dim=1)
        mask = torch.eq(indices, y).view(-1)
        y_pred = y_pred[mask]
        y = y[mask]

        nll = F.nll_loss(F.log_softmax(y_pred, dim=0), y)
        self._sum_nll += nll
        self._count_nll += y.shape[0]

    def compute(self):
        return self._sum_nll / self._count_nll


# Incorrect NLL(loss)
class IncorrectNLL(Metric):

    def __init__(self, output_transform=lambda x: x, device=None):
        self._sum_nll = 0.0
        self._count_nll = 0
        super(IncorrectNLL, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_nll = 0.0
        self._count_nll = 0
        super(IncorrectNLL, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        indices = torch.argmax(y_pred, dim=1)
        mask = torch.ne(indices, y).view(-1)
        y_pred = y_pred[mask]
        y = y[mask]

        nll = F.nll_loss(F.log_softmax(y_pred, dim=0), y)
        self._sum_nll += nll
        self._count_nll += y.shape[0]

    def compute(self):
        return self._sum_nll / self._count_nll



# Correct Crossentropy
class CorrectCrossEntropy(Metric):

    def __init__(self, output_transform=lambda x: x, device=None):
        self._sum_cross_entropy = 0.0
        self._count_cross_entropy = 0
        super(CorrectCrossEntropy, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_cross_entropy = 0.0
        self._count_cross_entropy = 0
        super(CorrectCrossEntropy, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        indices = torch.argmax(y_pred, dim=1)
        mask = torch.eq(indices, y).view(-1)
        y_pred = y_pred[mask]
        y = y[mask]

        entropy = F.cross_entropy(F.log_softmax(y_pred, dim=0), y)
        self._sum_cross_entropy += entropy
        self._count_cross_entropy += y.shape[0]

    def compute(self):
        return self._sum_cross_entropy / self._count_cross_entropy


# Incorrect Entropy
class IncorrectCrossEntropy(Metric):

    def __init__(self, output_transform=lambda x: x, device=None):
        self._sum_cross_entropy = 0.0
        self._count_cross_entropy = 0
        super(IncorrectCrossEntropy, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_cross_entropy = 0.0
        self._count_cross_entropy = 0
        super(IncorrectCrossEntropy, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        indices = torch.argmax(y_pred, dim=1)
        mask = torch.ne(indices, y).view(-1)
        y_pred = y_pred[mask]
        y = y[mask]

        entropy = F.cross_entropy(F.log_softmax(y_pred, dim=0), y)
        self._sum_cross_entropy += entropy
        self._count_cross_entropy += y.shape[0]

    def compute(self):
        return self._sum_cross_entropy / self._count_cross_entropy


# ECE & histogram
class ECE(Metric):

    def __init__(self, output_transform=lambda x: x, device=None):
        self.n_bins = 25
        self._sum_ece = 0.0
        self._accuracy_sum_bins = torch.zeros([self.n_bins], dtype=torch.float32)
        self._accuracy_num_bins = torch.zeros([self.n_bins], dtype=torch.int32)
        super(ECE, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_ece = 0.0
        self._accuracy_sum_bins = torch.zeros([self.n_bins], dtype=torch.float32)
        self._accuracy_num_bins = torch.zeros([self.n_bins], dtype=torch.int32)
        super(ECE, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        softmaxes = F.softmax(y_pred, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(y)

        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = torch.zeros(1, device=y_pred.device)
        idx = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                # Count accuracy in each bin
                self._accuracy_sum_bins[idx] += accuracies[in_bin].float().sum()
                self._accuracy_num_bins[idx] += in_bin.int().sum()

            idx += 1

        self._sum_ece += ece.item()

    def compute(self):
        for i in range(self.n_bins):
            if (self._accuracy_num_bins[i] == 0):
                self._accuracy_sum_bins[i] = 0
            else:
                self._accuracy_sum_bins[i] = self._accuracy_sum_bins[i]/self._accuracy_num_bins[i]

        return self._sum_ece, self._accuracy_sum_bins, self._accuracy_num_bins

