
import keras.backend as KB

def f1(y_true, y_pred):
    true_positives = KB.sum(KB.round(KB.clip(y_true * y_pred, 0, 1)))
    possible_positives = KB.sum(y_true)
    recall = true_positives / (possible_positives + KB.epsilon())
    predicted_positives = KB.sum(KB.round(y_pred))
    precision = true_positives / (predicted_positives + KB.epsilon())

    return 2*((precision*recall)/(precision+recall+KB.epsilon()))


def recall(y_true, y_pred):
    true_positives = KB.sum(KB.round(KB.clip(y_true * y_pred, 0, 1)))
    possible_positives = KB.sum(KB.round(KB.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + KB.epsilon())
    return recall
