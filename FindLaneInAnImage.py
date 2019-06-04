import numpy as np
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import os, sys, datetime, platform
import DataSet
from PIL import Image



def ApplyLaneDetection(Y_hat, XtestImg):
    for i in range(Y_hat.shape[0]):
        for j in range(Y_hat.shape[1]):
            rvalue = Y_hat[i][j]
            if (rvalue>0.5):
                XtestImg[i][j][1] = 255
    
def SaveInFolder(id, XtestImg, Y, out_dir):
    im = Image.fromarray(XtestImg)
    imgFile = os.path.join(out_dir, str(id)+'.jpg')
    im.save(imgFile)

    im = Image.fromarray(Y)
    imgFile = os.path.join(out_dir, str(id)+'_Y.jpg')
    im.save(imgFile)
    #cv2.imshow("image", XtestImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

currentDir = os.path.dirname(os.path.realpath(__file__))
model = load_model(os.path.join(currentDir, 'Model.h5'))

resultdir = os.getcwd()
if platform.system() == 'Linux':
    resultdir = '/var/www/html/results'

mydir = os.path.join(resultdir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(mydir)


ds = DataSet.TrainDataGenerator(DataSet.dev_set_count, 'dev', 0.2)
#ds = DataSet.TrainDataGenerator(10, 'dev')

devSet = next(ds)
XList = devSet[0]
YList = devSet[1]
Y_hat = model.predict(XList)

h = XList[0].shape[0] // 2
w = XList[0].shape[1] // 2
hw = '' #' height=' + str(h) + " width=" + str(w)

htmlfile = os.path.join(mydir, 'index.html')
out = open(htmlfile, 'w')
out.write('<html><body><table>')
for ii in range(XList.shape[0]):
    tempX = np.copy(XList[ii])
    print(ii, np.count_nonzero(Y_hat[ii]))
    ApplyLaneDetection(Y_hat[ii], XList[ii])    
    ApplyLaneDetection(YList[ii], tempX)
    SaveInFolder(ii, XList[ii], tempX, mydir)
    out.write('<tr><td><img src=\'' + str(ii) + '.jpg\'' + hw + '></td><td><img src=\'' + str(ii)+'_Y.jpg\'' + hw + '></td></tr>')

out.write('</table></body></html>')
out.close()

