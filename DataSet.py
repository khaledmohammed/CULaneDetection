import sys
import pickle
import cv2
import os, fnmatch
import glob
import sys
import gc
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


def TrainDataGenerator(batch_size, mode):
    X_top_folder = 'D:/cs230-project/CULaneOriginalImage'
    Y_top_folder = 'D:/cs230-project/CULaneLabels/'
    
    count = 1
    skipCount = 0
    X_train = []
    Y_train = []
    for subDir in os.listdir(X_top_folder):
        X_path = os.path.join(X_top_folder, subDir)
        Y_path = os.path.join(Y_top_folder, subDir)
        for subDir2 in os.listdir(X_path):
            X_path2 = os.path.join(X_path, subDir2)
            Y_path2 = os.path.join(Y_path, subDir2)
            for file in os.listdir(X_path2):
                if fnmatch.fnmatch(file, "*.jpg"):
                    X_img_file = os.path.join(X_path2, file)
                    Y_img_file = os.path.join(Y_path2, file)
                    Y_img_file = os.path.splitext(Y_img_file)[0] + '.png'
                    if (os.path.isfile(Y_img_file)):
                        skipCount = skipCount + 1
                        if mode=="dev" and skipCount % 100 == 0:
                            continue
                        imgArrY = cv2.imread(Y_img_file, cv2.IMREAD_COLOR)
                        imgArrX = cv2.imread(X_img_file, cv2.IMREAD_COLOR)
                        X_train.append(imgArrX)
                        Y_train.append(imgArrY)
                        if count % batch_size == 0:
#                            print("New batch: ", X_path2, 'Img Cnt: ', count)
                            yield((np.array(X_train), np.array(Y_train)))
                            X_train = []
                            Y_train = []                            
                        count = count + 1
                    
    

    
d = TrainDataGenerator(16, 'dev')
print(next(d)[0].shape)
print(next(d)[0].shape)