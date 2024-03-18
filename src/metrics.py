from typing import Tuple, List

import numpy as np
import torch
from torchtext.data.metrics import bleu_score
from torchmetrics.classification import BinaryAccuracy, Accuracy


def bleu_scorer(predicted: np.ndarray, actual: np.ndarray, target_tokenizer):
    """Convert predictions to sentences and calculate
    BLEU score.

    Args:
        predicted (np.ndarray): batch of indices of predicted words
        actual (np.ndarray): batch of indices of ground truth words

    Returns:
        Tuple[float, List[str], List[str]]: tuple of
            (
                bleu score,
                ground truth sentences,
                predicted sentences
            )
    """
    print(predicted, actual)
    batch_bleu = []
    predicted_sentences = []
    actual_sentences = []
    for a, b in zip(predicted, actual):
        words_predicted = target_tokenizer.decode(a)
        to_metric_pred = words_predicted.replace(" ", "") if words_predicted else ""
        words_actual = target_tokenizer.decode(b)
        to_metric_trg = words_actual.replace(" ", "") if words_actual else ""
        batch_bleu.append(bleu_score([to_metric_pred], [[to_metric_trg]], max_n=4, weights=[0.25, 0.25, 0.25, 0.25]))
        predicted_sentences.append(words_predicted)
        actual_sentences.append(words_actual)
    batch_bleu = np.mean(batch_bleu)
    return batch_bleu, actual_sentences, predicted_sentences

def clas_scorer(predicted: np.ndarray, actual: np.ndarray):
    metric = BinaryAccuracy()
    batch_metric = float(metric(predicted, actual))
    return batch_metric

def str_scorer(predicted: np.ndarray, actual: np.ndarray, target_tokenizer):
    metric = Accuracy(task="multiclass", num_classes=len(target_tokenizer), ignore_index=-1)
    batch_metric = []
    predicted_sentences = []
    actual_sentences = []

    for a, b in zip(predicted, actual):
        predict = np.where(b != target_tokenizer.tokenizer.pad_token_id, a, -1)
        labels = np.where(b != target_tokenizer.tokenizer.pad_token_id, b, -1)
        batch_metric.append(float(metric(torch.tensor(predict), torch.tensor(labels))))

        words_predicted = target_tokenizer.decode(a)
        words_actual = target_tokenizer.decode(b)
        predicted_sentences.append(words_predicted)
        actual_sentences.append(words_actual)

    batch_metric = np.mean(batch_metric)
    return batch_metric, actual_sentences, predicted_sentences

def mask_scorer(input_tensor: np.ndarray, predicted: np.ndarray, actual: np.ndarray, tokenizer):
    metric = Accuracy(task="multiclass", num_classes=len(tokenizer), ignore_index=-1)
    batch_metric = []
    predicted_sentences = []
    actual_sentences = []

    for a, b, c in zip(input_tensor, predicted, actual):
        predict = np.where(a == tokenizer.tokenizer.mask_token_id, b, -1)
        labels = np.where(a == tokenizer.tokenizer.mask_token_id, c, -1)
        batch_metric.append(float(metric(torch.tensor(predict), torch.tensor(labels))))

        predict = predict[(predict != -1)]
        labels = labels[(labels != -1)]

        words_predicted = tokenizer.decode(predict)
        words_actual = tokenizer.decode(labels)
        predicted_sentences.append(words_predicted)
        actual_sentences.append(words_actual)

    batch_metric = np.mean(batch_metric)
    return batch_metric, actual_sentences, predicted_sentences