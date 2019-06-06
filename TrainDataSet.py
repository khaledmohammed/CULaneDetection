import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Import necessary items from Keras
import keras as K


import DataSet
from Model import CreateModel
import CustomLoss
import CustomMetric

matplotlib.use("Agg")

# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
image_resizing_factor = 0.2
batch_size = 128 
epochs = 300
input_shape = (int(590 * image_resizing_factor), int(1640 * image_resizing_factor), 3)

# Create Network Model
model = CreateModel(input_shape)
model.summary()

# Compiling and training the model
optimizer = K.optimizers.Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss=CustomLoss.dice_loss, metrics=[CustomMetric.f1, 'acc', CustomMetric.recall])

# DataSet Generators
trainGen = DataSet.TrainDataGenerator(batch_size=batch_size, mode='train', image_resizing_factor=image_resizing_factor)
devGen = DataSet.TrainDataGenerator(batch_size=batch_size, mode='dev', image_resizing_factor=image_resizing_factor)
validation_steps = DataSet.dev_set_count // batch_size
steps_per_epoch = DataSet.train_set_count // batch_size

# Start training
history = model.fit_generator(generator=trainGen, steps_per_epoch=steps_per_epoch, epochs=epochs,
                              verbose=1, validation_data=devGen, validation_steps=validation_steps) 


# Freeze layers since training is done
model.trainable = False
model.compile(optimizer=optimizer, loss=CustomLoss.dice_loss)

# Save model architecture and weights
model.save('Model.h5')

# plot a graph 
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), history.history["recall"], label="recall")
plt.plot(np.arange(0, epochs), history.history["f1"], label="f1")
plt.plot(np.arange(0, epochs), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), history.history["val_acc"], label="val_acc")
plt.plot(np.arange(0, epochs), history.history["val_f1"], label="val_f1")
plt.plot(np.arange(0, epochs), history.history["val_recall"], label="val_recall")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")


