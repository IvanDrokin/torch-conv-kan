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

    if metric_config.report_type == 'classification_minimum':

        classes = np.argmax(predictions, axis=-1)

        return {**top1_accuracy(targets, classes),
                **top5_accuracy(targets, predictions)}

    if metric_config.report_type == 'segmentation':
        return dice(targets, predictions)


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


def dice(targets, predictions, smooth: float = 1.):
    num_classes = predictions.shape[1]

    dice_val = 0.
    for i in range(num_classes):
        # flatten label and prediction tensors

        c_inp = predictions[:, i].reshape(predictions.shape[0], -1)
        c_tgt = targets[:, i].reshape(predictions.shape[0], -1)

        intersection = np.sum(c_inp * c_tgt, axis=1)
        dice_val += (2. * intersection + smooth) / (c_inp.sum(axis=1) + c_tgt.sum(axis=1) + smooth)
    return {"dice": np.mean(dice_val) / float(predictions.shape[1])}
