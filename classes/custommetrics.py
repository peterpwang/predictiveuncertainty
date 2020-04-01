import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

# All inputs are
# y_pred: (64*classes) [-0.4012, -1.4351,  0.1430,  1.4880,  0.0253,  1.8506, -1.3660,  0.7904, -0.6024, -1.4305],...
# y: (64) [7, 8, 8, 8, 0, ...]

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
        y_pred = y_pred.to("cuda")
        y = y.to("cuda")
        batch_size = y_pred.shape[0]

        y_pred = F.softmax(y_pred, dim=1).to("cuda")
        y = F.one_hot(y, num_classes=y_pred.shape[1]).to("cuda")
        nll = -torch.log(torch.sum(y * y_pred, dim=1)).to("cuda").sum().item()

        self._sum_nll += nll
        self._count_nll += batch_size
        #print("all:", self._sum_nll, ":", self._count_nll)

    def compute(self):
        if self._count_nll == 0:
            return 0.0
        else:
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
        batch_size = y_pred.shape[0]
        y_pred = y_pred.to("cuda")
        y = y.to("cuda")

        indices = torch.argmax(y_pred, dim=1).to("cuda")
        mask = torch.eq(indices, y).view(-1).to("cuda")
        y_pred = y_pred[mask].to("cuda")
        y = y[mask].to("cuda")

        if len(y_pred) > 0:
            y_pred = F.softmax(y_pred, dim=1).to("cuda")
            y = F.one_hot(y, num_classes=y_pred.shape[1]).to("cuda")
            nll = -torch.log(torch.sum(y * y_pred, dim=1)).sum().item()
            self._sum_nll += nll
            self._count_nll += batch_size
        #print("correct:", self._sum_nll, ":", self._count_nll)

    def compute(self):
        if self._count_nll == 0:
            return 0.0
        else:
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
        batch_size = y_pred.shape[0]
        y_pred = y_pred.to("cuda")
        y = y.to("cuda")

        indices = torch.argmax(y_pred, dim=1).to("cuda")
        mask = torch.ne(indices, y).view(-1).to("cuda")
        y_pred = y_pred[mask].to("cuda")
        y = y[mask].to("cuda")

        if len(y_pred) > 0:
            y_pred = F.softmax(y_pred, dim=1).to("cuda")
            y = F.one_hot(y, num_classes=y_pred.shape[1]).to("cuda")
            nll = -torch.log(torch.sum(y * y_pred, dim=1)).sum().item()
            self._sum_nll += nll
            self._count_nll += batch_size
        #print("incorrect:", self._sum_nll, ":", self._count_nll)

    def compute(self):
        if self._count_nll == 0:
            return 0.0
        else:
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
        y_pred = y_pred.to("cuda")
        y = y.to("cuda")

        y_pred = F.softmax(y_pred, dim=1).to("cuda")
        indices = torch.argmax(y_pred, dim=1).to("cuda")
        mask = torch.eq(indices, y).view(-1).to("cuda")
        y_pred = y_pred[mask].to("cuda")
        y = y[mask].to("cuda")

        if len(y_pred) > 0:
            entropy = F.cross_entropy(torch.log(y_pred), y).item()
            self._sum_cross_entropy += entropy
            self._count_cross_entropy += y_pred.shape[0]

    def compute(self):
        if self._count_cross_entropy == 0:
            return 0.0
        else:
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
        y_pred = y_pred.to("cuda")
        y = y.to("cuda")

        y_pred = F.softmax(y_pred, dim=1).to("cuda")
        indices = torch.argmax(y_pred, dim=1).to("cuda")
        mask = torch.ne(indices, y).view(-1).to("cuda")
        y_pred = y_pred[mask].to("cuda")
        y = y[mask].to("cuda")

        if len(y_pred) > 0:
            entropy = F.cross_entropy(torch.log(y_pred), y).item()
            self._sum_cross_entropy += entropy
            self._count_cross_entropy += y_pred.shape[0]

    def compute(self):
        if self._count_cross_entropy == 0:
            return 0.0
        else:
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
        y_pred = y_pred.to("cuda")
        y = y.to("cuda")

        softmaxes = F.softmax(y_pred, dim=1).to("cuda")
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(y).to("cuda")

        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1).to("cuda")
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = torch.zeros(1, device=y_pred.device).to("cuda")
        idx = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item()).to("cuda")
            prop_in_bin = in_bin.float().mean().to("cuda")
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean().to("cuda")
                avg_confidence_in_bin = confidences[in_bin].mean().to("cuda")
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                # Count accuracy in each bin
                self._accuracy_sum_bins[idx] += accuracies[in_bin].float().sum()
                self._accuracy_num_bins[idx] += in_bin.int().sum()

            idx += 1

        # ece is for each epoch
        self._sum_ece = ece.item()

    def compute(self):
        for i in range(self.n_bins):
            if (self._accuracy_num_bins[i] == 0):
                self._accuracy_sum_bins[i] = 0
            else:
                self._accuracy_sum_bins[i] = self._accuracy_sum_bins[i]/self._accuracy_num_bins[i]

        return self._sum_ece, self._accuracy_sum_bins, self._accuracy_num_bins

