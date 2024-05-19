from sklearn import metrics
import numpy as np


def get_metrics(targets, predictions, metric_config):
    if metric_config.report_type == 'classification':

        classes = np.argmax(predictions, axis=-1)

        return {**top1_accuracy(targets, classes),
                **top5_accuracy(targets, predictions),
                **recall(targets, classes),
                **f1(targets, classes),
                **auc(targets, predictions)}


def top5_accuracy(targets, predictions):
    return {'accuracy, top5': metrics.top_k_accuracy_score(targets, predictions, k=5)}


def top1_accuracy(targets, predictions):
    return {'accuracy': metrics.accuracy_score(targets, predictions)}


def recall(targets, predictions):
    return {'recall, macro': metrics.recall_score(targets, predictions, average="macro"),
            'recall, micro': metrics.recall_score(targets, predictions, average="micro")}


def f1(targets, predictions):
    return {'f1_score, macro': metrics.f1_score(targets, predictions, average="macro"),
            'f1_score, micro': metrics.f1_score(targets, predictions, average="micro")}


def auc(targets, predictions):
    return {'auc, ovo': metrics.roc_auc_score(targets, predictions, average='macro', multi_class='ovo'),
            'auc, ovr': metrics.roc_auc_score(targets, predictions, average='macro', multi_class='ovr')}
