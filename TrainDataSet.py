import numpy as np


import matplotlib
import matplotlib.pyplot as plt

# Import necessary items from Keras
from keras.preprocessing.image import ImageDataGenerator
import keras as K
import DataSet
from Model import CreateModel


matplotlib.use("Agg")

#(X_train, y_train, X_val, y_val) = DataSet.GetPickleDataSet()

# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 16
epochs = 30
input_shape = (590, 1640, 3)

model = CreateModel(input_shape)


# Compiling and training the model
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=["mae", "acc"])
#history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, y_val))

trainGen = DataSet.TrainDataGenerator(batch_size=batch_size, mode='train')
devGen = DataSet.TrainDataGenerator(batch_size=batch_size*10, mode='train')
val_data = next(devGen)
history = model.fit_generator(generator=trainGen, steps_per_epoch=30, epochs=epochs,
                              verbose=1, validation_data=val_data) 

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
model.compile(optimizer='Adam', loss='mean_squared_error')

# Save model architecture and weights
model.save('Model.h5')

# Show summary of model
model.summary()

