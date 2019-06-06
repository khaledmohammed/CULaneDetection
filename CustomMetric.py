
import keras.backend as KB

import numpy as np

def f1(y_true, y_pred):
    y_true_f = KB.flatten(y_true)
    y_pred_f = KB.flatten(y_pred)
    true_positives = KB.sum(KB.round(KB.clip(y_true_f * y_pred_f, 0, 1)), axis=-1)
    possible_positives = KB.sum(KB.round(KB.clip(y_true_f, 0, 1)), axis=-1)
    recall = true_positives / (possible_positives + KB.epsilon())
    predicted_positives = KB.sum(KB.round(KB.clip(y_pred_f, 0, 1)), axis=-1)
    precision = true_positives / (predicted_positives + KB.epsilon())

    return 2*((precision*recall)/(precision+recall+KB.epsilon()))


def compute_f1(y_true, y_pred, threshold):
    epsilon = 1e-7
    if threshold == 0.5:
        offset = 0
    else:
        offset = 0.5 - threshold
    true_positives = np.sum(np.clip(np.round(offset + np.clip(y_true * y_pred, 0, 1)), 0,1))
    possible_positives = np.sum(y_true)
    recall = true_positives / (possible_positives + epsilon)
    predicted_positives = np.sum(np.clip(np.round(offset + np.clip(y_pred, 0, 1)),0,1))
    precision = true_positives / (predicted_positives + epsilon)
    return (2*((precision*recall)/(precision+recall+epsilon)))


def true_positive(y_true, y_pred):
    true_positives = KB.sum(KB.round(KB.clip(y_true * y_pred, 0, 1)), -1)
    return true_positives

def possible_positive(y_true, y_pred):
    possible_positives = KB.sum(y_true, axis=-1)
    return possible_positives

def recall(y_true, y_pred):
    y_true_f = KB.flatten(y_true)
    y_pred_f = KB.flatten(y_pred)
    true_positives = true_positive(y_true_f, y_pred_f)
    possible_positives = possible_positive(y_true_f, y_pred_f)
    recall = true_positives / (possible_positives + KB.epsilon())
    return recall
