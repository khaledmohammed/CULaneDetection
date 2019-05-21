import numpy as np


import matplotlib
import matplotlib.pyplot as plt

# Import necessary items from Keras
from keras.preprocessing.image import ImageDataGenerator
import keras as K
from DataSet import GetPickleDataSet
from Model import CreateModel


matplotlib.use("Agg")


(X_train, y_train, X_val, y_val) = GetPickleDataSet()

# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 16
epochs = 10
input_shape = X_train.shape[1:]

model = CreateModel(input_shape)
model.summary()

# Using a generator to help the model use less data
# Channel shifts help with shadows slightly
#datagen = ImageDataGenerator(channel_shift_range=0.2)
#datagen.fit(X_train)

# Compiling and training the model
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=["mae", "acc"])
history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, y_val))

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

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')

# Freeze layers since training is done
model.trainable = False
model.compile(optimizer='Adam', loss='mean_squared_error')

# Save model architecture and weights
model.save('Model.h5')

# Show summary of model


