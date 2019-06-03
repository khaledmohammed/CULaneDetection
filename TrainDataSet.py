import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Import necessary items from Keras
from keras.preprocessing.image import ImageDataGenerator
import keras as K
import keras.backend as KB
import DataSet
from Model import CreateModel

# TODO
# 1. increase the # of filters - 8 is too small.
# 2. try bigger dataset


matplotlib.use("Agg")

# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
image_resizing_factor = 0.2
batch_size = 128
epochs = 150
input_shape = (int(590 * image_resizing_factor), int(1640 * image_resizing_factor), 3)

model = CreateModel(input_shape)

model.summary()

def recall(y_true, y_pred):
    true_positives = KB.sum(KB.round(KB.clip(y_true * y_pred, 0, 1)))
    possible_positives = KB.sum(KB.round(KB.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + KB.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = KB.sum(KB.round(KB.clip(y_true * y_pred, 0, 1)))
        possible_positives = KB.sum(KB.round(KB.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + KB.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = KB.sum(KB.round(KB.clip(y_true * y_pred, 0, 1)))
        predicted_positives = KB.sum(KB.round(KB.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + KB.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+KB.epsilon()))


# Compiling and training the model
optimizer = K.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[f1, recall, 'acc'])

trainGen = DataSet.TrainDataGenerator(batch_size=batch_size, mode='train', image_resizing_factor=image_resizing_factor)
devGen = DataSet.TrainDataGenerator(batch_size=batch_size, mode='dev', image_resizing_factor=image_resizing_factor)
validation_steps = DataSet.dev_set_count // batch_size
steps_per_epoch = DataSet.train_set_count // batch_size
#print('validation_steps=', str(validation_steps), 'steps_per_epoch=', steps_per_epoch)
history = model.fit_generator(generator=trainGen, steps_per_epoch=steps_per_epoch, epochs=epochs,
                              verbose=1, validation_data=devGen, validation_steps=validation_steps) 

# plot a graph 
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

# Freeze layers since training is done
model.trainable = False
model.compile(optimizer=optimizer, loss='binary_crossentropy')

# Save model architecture and weights
model.save('Model.h5')

# Show summary of model
model.summary()

