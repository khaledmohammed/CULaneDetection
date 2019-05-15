import numpy as np
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import os

currentDir = os.path.dirname(os.path.realpath(__file__))
model = load_model(os.path.join(currentDir, 'Model.h5'))

XtestImgFile = os.path.join(currentDir, "Images/00000.jpg")
XtestImg = cv2.imread(XtestImgFile, cv2.IMREAD_COLOR)

XList = np.array([ XtestImg ])

print(XList.shape)
Y_hat = model.predict(XList)

Y_hat = Y_hat[0]

print(Y_hat.shape[0], Y_hat.shape[1])

for i in range(Y_hat.shape[0]):
    for j in range(Y_hat.shape[1]):
        rvalue = int(Y_hat[i][j][0])
        if (rvalue>=1):
            XtestImg[i][j][rvalue%3] = 255
            

#plt.figure('image')
#plt.imshow(XtestImg)

#plt.show()

cv2.imshow("image", XtestImg)
cv2.waitKey(0)
cv2.destroyAllWindows()