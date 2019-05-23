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


X_train_files = []
Y_train_files = []
X_dev_files = []
Y_dev_files = []
train_set_count = 0
dev_set_count = 0

def ScanFolders(dev_set_ratio):
    global X_train_files
    global Y_train_files
    global X_dev_files
    global Y_dev_files
    global train_set_count
    global dev_set_count

    X_top_folder = 'D:/cs230-project/CULaneOriginalImage'
    Y_top_folder = 'D:/cs230-project/CULaneLabels'
    
    
    for subDir in os.listdir(X_top_folder):
        X_path = os.path.join(X_top_folder, subDir)
        Y_path = os.path.join(Y_top_folder, subDir)
        if not os.path.isdir(X_path):
                continue
        for subDir2 in os.listdir(X_path):
            X_path2 = os.path.join(X_path, subDir2)
            Y_path2 = os.path.join(Y_path, subDir2)
            if not os.path.isdir(X_path2):
                continue
            X_files = []
            Y_files = []
            for file in os.listdir(X_path2):
                if fnmatch.fnmatch(file, "*.jpg"):
                    X_img_file = os.path.join(X_path2, file)
                    Y_img_file = os.path.join(Y_path2, file)
                    Y_img_file = os.path.splitext(Y_img_file)[0] + '.png'
                    if (os.path.isfile(Y_img_file)):
                        X_files.append(X_img_file)
                        Y_files.append(Y_img_file)

            X, Y = shuffle(np.array(X_files), np.array(Y_files))
            X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=dev_set_ratio)
            X_train_files = X_train_files + X_train.tolist()            
            Y_train_files = Y_train_files + Y_train.tolist()
            X_dev_files = X_dev_files + X_dev.tolist()
            Y_dev_files = Y_dev_files + Y_dev.tolist()
        
    train_set_count = len(X_train_files)
    dev_set_count = len(X_dev_files)
    print('Train Set:', str(train_set_count), 'Dev Set:', str(dev_set_count))



def TrainDataGenerator(batch_size, mode):
        
    if mode == 'dev':
        X_files = X_dev_files
        Y_files = Y_dev_files
    else:
        X_files = X_train_files
        Y_files = Y_train_files
    
    #print('Available images in', mode, 'dataset:', str(len(X_files)))
    
    ii = 0
    X_train = []
    Y_train = []
    batch_count = 0
    while True:                
        imgArrY = cv2.imread(Y_files[ii], cv2.IMREAD_COLOR)
        imgArrX = cv2.imread(X_files[ii], cv2.IMREAD_COLOR)
        X_train.append(imgArrX)
        Y_train.append(imgArrY)
        if len(X_train) == batch_size:            
            yield((np.array(X_train), np.array(Y_train)))
            X_train = []
            Y_train = []
        
        ii = ii + 1

        if ii == len(X_files):
            ii = 0
        
ScanFolders(0.1)
