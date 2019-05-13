
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 20:26:50 2019

@author: khaledm
"""

import cv2
import numpy as np
import os
import pickle
import sys
import gc

currentDir = os.path.dirname(os.path.realpath(__file__))
X_train_img = 'D:\\cs230-project\\driver_23_30frame\\' 
Y_train_img = 'D:\\cs230-project\\repro\\seg_label_generate\\laneseg_label\\driver_23_30frame\\' 

X_train = []
Y_train = []
count = 0
inputset = 0
for dirname, dirnames, filenames in os.walk(X_train_img):
    for dirn in dirnames:
        X_subdirname = os.path.join(X_train_img, dirn)
        Y_subdirname = os.path.join(Y_train_img, dirn)
        if (os.path.isdir(X_subdirname) and os.path.isdir(Y_subdirname)):
            for dirname, dirnames, filenames in os.walk(X_train_img):
                for imgfile in filenames:
                    if os.path.splitext(imgfile)[1] == '.jpg':
                        img_fileX = os.path.join(X_subdirname, imgfile)
                        img_fileY = os.path.join(Y_subdirname, imgfile)
                        img_fileY = os.path.splitext(img_fileY)[0] + '.png'
                     
                        if (os.path.isfile(img_fileX) and os.path.isfile(img_fileY)):
                            imgArrY = cv2.imread(img_fileY, cv2.IMREAD_COLOR)
                            imgArrX = cv2.imread(img_fileX, cv2.IMREAD_COLOR)
                            X_train.append(imgArrX)
                            Y_train.append(imgArrY)
                            del imgArrY
                            del imgArrX
                            count = count + 1
                            if count == 600:
                                pickle.dump(X_train, open(os.path.join(currentDir, "X.p"), "wb"))
                                pickle.dump(Y_train, open(os.path.join(currentDir, "Y.p"), "wb"))
                                print("Completed", str(inputset+1), "set. Count=", len(X_train))
                                sys.exit(0)
                                del X_train
                                del Y_train
                                gc.collect()
                                X_train = []
                                Y_train = []                                
                            elif count == 100:
                                gc.collect()
