import numpy as np
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import os

currentDir = os.path.dirname(os.path.realpath(__file__))
model = load_model(os.path.join(currentDir, 'Model.h5'))

XtestImgFile = os.path.join(currentDir, "00000.jpg")
XtestImg = cv2.imread(XtestImgFile, cv2.IMREAD_COLOR)

XList = np.array([ XtestImg ])

print(XList.shape)
Y_hat = model.predict(XList)

Y_hat = Y_hat[0]

for i in range(Y_hat.shape[0]):
    for j in range(Y_hat.shape[1]):
        if (Y_hat[i][j][0]>=1):            
            Y_hat[i][j] = (255, 0, 0)
            

plt.figure('original image')
plt.imshow(testImg)

plt.figure('detected lanes')
plt.imshow(Y_hat)

plt.show()