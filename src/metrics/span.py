import torch
from torchmetrics import Metric


class SpanF1(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('span_correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('span_all_pred', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('span_all_true', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('head_correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('head_all_pred', default=torch.tensor(0.0), dist_reduce_fx='sum')
        






class SpanEvaluator(object):
    def __init__(self):
        super().__init__()
    
    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)
    
    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum()+1)
    
    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred>0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true>0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall