import torch
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score


class Metric:

    @staticmethod
    def nmi(y, y_pred):
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
        return float(normalized_mutual_info_score(y, y_pred))

    @staticmethod
    def ar(y, y_pred):
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
        return float(adjusted_rand_score(y, y_pred))

    @staticmethod
    def cluster_accuracy(y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        d = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((d, d), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size

    @staticmethod
    def calculate_purity(labels, preds):
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
        num_inst = len(preds)
        num_labels = np.max(labels) + 1
        conf_matrix = np.zeros((num_labels, num_labels))
        for i in range(0, num_inst):
            gt_i = labels[i]
            pr_i = preds[i]
            conf_matrix[gt_i, pr_i] = conf_matrix[gt_i, pr_i] + 1
        num_inst = np.sum(conf_matrix).astype(float)
        best_unions = np.amax(conf_matrix, axis=0)
        return np.sum(best_unions) / num_inst
