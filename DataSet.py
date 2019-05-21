
import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def GetPickleDataSet():
    # Load training images
    train_images = pickle.load(open("X.p", "rb" ))

    # Load image labels
    labels = pickle.load(open("Y.p", "rb" ))

    # Make into arrays as the neural network wants these
    train_images = np.array(train_images)
    labels = np.array(labels)

    # Normalize labels - training images get normalized to start in the network
    labels = labels/4

    # Shuffle images along with their labels, then split into training/validation sets
    train_images, labels = shuffle(train_images, labels)
    # Test size may be 10% or 20%
    X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)
    
    return (X_train, y_train, X_val, y_val)