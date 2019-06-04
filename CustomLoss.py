

import keras.backend as KB

def dice_coef(y_true, y_pred, smooth, thresh):
    #y_pred = y_pred > thresh
    y_true_f = KB.flatten(y_true)
    y_pred_f = KB.flatten(y_pred)
    intersection = KB.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (KB.sum(y_true_f) + KB.sum(y_pred_f) + smooth)

def dice_loss(smooth, thresh):
  def dice(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred, smooth, thresh)
  return dice