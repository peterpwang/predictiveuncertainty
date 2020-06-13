import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from copy import deepcopy

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

epsilon = 1e-7

# All inputs are
# y_pred: (records*classes) [-0.4012, -1.4351,  0.1430,  1.4880,  0.0253,  1.8506, -1.3660,  0.7904, -0.6024, -1.4305],...
# y: (records) [0, 1, 2, 2, 0, ...]
def calculate_accuracy(predictions, labels):
    val, predictions = torch.max(predictions, 1)

    correct = (predictions==labels).sum()
    total = labels.size(0)
    acc = float(correct)/total
    return acc

def calculate_nll(y_pred, y):
    batch_size = y_pred.shape[0]

    y_pred = F.softmax(y_pred, dim=1)
    y_pred = y_pred[range(y_pred.shape[0]), y] + epsilon
    nll = -torch.log(y_pred).sum().item()

    return nll / batch_size

def calculate_correct_nll(y_pred, y):
    batch_size = y_pred.shape[0]

    indices = torch.argmax(y_pred, dim=1)
    mask = torch.eq(indices, y).view(-1)
    y_pred = y_pred[mask]
    y = y[mask]

    null = 0.0
    if len(y_pred) > 0:
        y_pred = F.softmax(y_pred, dim=1)
        y_pred = y_pred[range(y_pred.shape[0]), y] + epsilon
        nll = -torch.log(y_pred).sum().item()

    return nll / batch_size

def calculate_incorrect_nll(y_pred, y):
    batch_size = y_pred.shape[0]

    indices = torch.argmax(y_pred, dim=1)
    mask = torch.ne(indices, y).view(-1)
    y_pred = y_pred[mask]
    y = y[mask]

    nll = 0.0
    if len(y_pred) > 0:
        y_pred = F.softmax(y_pred, dim=1)
        y_pred = y_pred[range(y_pred.shape[0]), y] + epsilon
        nll = -torch.log(y_pred).sum().item()

    return nll / batch_size

def calculate_correct_entropy(y_pred, y):
    batch_size = y_pred.shape[0]

    y_pred = F.softmax(y_pred, dim=1)
    mask = torch.eq(torch.argmax(y_pred, dim=1), y).view(-1)
    y_pred = y_pred[mask]

    entropy = 0.0
    if len(y_pred) > 0:
        entropy = -y_pred * torch.log(y_pred + epsilon)

    return entropy / batch_size

def calculate_incorrect_entropy(y_pred, y):
    batch_size = y_pred.shape[0]

    y_pred = F.softmax(y_pred, dim=1)
    mask = torch.ne(torch.argmax(y_pred, dim=1), y).view(-1)
    y_pred = y_pred[mask]

    entropy = 0.0
    if len(y_pred) > 0:
        entropy = -y_pred * torch.log(y_pred + epsilon)

    return entropy / batch_size

# ECE & histogram
def calculate_ece(y_pred, y):
    n_bins = 25
    _sum_ece = 0.0
    _accuracy_sum_bins = torch.zeros([n_bins], dtype=torch.float32)
    _accuracy_num_bins = torch.zeros([n_bins], dtype=torch.int32)

    softmaxes = F.softmax(y_pred, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(y)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
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
            _accuracy_sum_bins[idx] += accuracies[in_bin].float().sum()
            _accuracy_num_bins[idx] += in_bin.int().sum()

        idx += 1

    # ece is for each epoch
    _sum_ece = ece.item()

    for i in range(n_bins):
        if (_accuracy_num_bins[i] == 0):
            _accuracy_sum_bins[i] = 0
        else:
            _accuracy_sum_bins[i] = _accuracy_sum_bins[i]/_accuracy_num_bins[i]

    return _sum_ece, _accuracy_sum_bins.cpu(), _accuracy_num_bins.cpu()

