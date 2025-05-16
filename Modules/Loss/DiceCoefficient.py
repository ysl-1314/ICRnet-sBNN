import torch
import numpy as np
from numpy import newaxis


class DiceCoefficient(torch.nn.Module):
    def __init__(self):
        super(DiceCoefficient, self).__init__()

    def forward(self, segment1, segment2):
        batch = segment1.size()[0]
        segment1 = segment1.view(batch, -1).float()
        segment2 = segment2.view(batch, -1).float()
        intersection = torch.sum(segment1 * segment2, 1)
        return 2 * intersection / (torch.sum(segment1, 1) + torch.sum(
            segment2, 1) + np.finfo(float).eps)

class DiceNumpy():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def dicecal(self):
        a_f = self.a.flatten()
        b_f = self.b.flatten()
        intersection = np.sum(a_f * b_f)
        c = np.sum(a_f)
        return (2. * intersection) / (np.sum(a_f) + np.sum(b_f))


class DiceCoefficientAll(torch.nn.Module):
    def __init__(self) -> None:
        super(DiceCoefficientAll, self).__init__()
        self.dice = DiceCoefficient()

    def forward(self, segment1, segment2):
        labels = torch.unique(segment1)
        res = []
        for l in labels:
            if l != 0:
                res.append(self.dice(segment1 == l, segment2 == l))
        if len(labels) > 2:
            res.append(self.dice((segment1 == labels[-1]) | (segment1 == labels[-2]), (segment2 == labels[-1]) | (segment2 == labels[-2])))
        res = torch.cat(res)
        res[res != res] = 0
        non_zero = res > 0
        mean = res.sum() / non_zero.sum()
        return mean if mean == mean else torch.tensor(0.0).cuda()
