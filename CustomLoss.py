

import keras.backend as KB

def dice_coef(y_true, y_pred, smooth, thresh):
    #y_pred = (y_pred > thresh).astype(float)
    y_true_f = KB.flatten(y_true)
    y_pred_f = KB.flatten(y_pred)
    intersection = KB.sum(y_true_f * y_pred_f, axis=-1)

    return (2. * intersection + smooth) / (KB.sum(KB.square(y_true_f), axis=-1) + KB.sum(KB.square(y_pred_f), axis=-1) + smooth)

def dice_loss(y_true, y_pred):
    return 1- dice_coef(y_true, y_pred, 1e-5, 0.5)
