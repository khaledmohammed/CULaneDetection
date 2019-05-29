import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Import necessary items from Keras
from keras.preprocessing.image import ImageDataGenerator
import keras as K
import DataSet
from Model import CreateModel

# TODO
# 1. increase the # of filters - 8 is too small.
# 2. try bigger dataset


matplotlib.use("Agg")

# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 128
epochs = 130
input_shape = (590 * 0.2, 1640 * 0.2, 3)

model = CreateModel(input_shape)


# Compiling and training the model
optimizer = K.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=["mae", "acc"])

trainGen = DataSet.TrainDataGenerator(batch_size=batch_size, mode='train')
devGen = DataSet.TrainDataGenerator(batch_size=batch_size, mode='dev')
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
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Save model architecture and weights
model.save('Model.h5')

# Show summary of model
model.summary()

