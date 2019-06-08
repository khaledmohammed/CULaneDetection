import numpy as np
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import os
import sys
import datetime
import platform
import DataSet
from PIL import Image
import CustomMetric
import CustomLoss


def ApplyLaneDetection(Y_hat, XtestImg, generateHeatMap = False):
    for i in range(Y_hat.shape[0]):
        for j in range(Y_hat.shape[1]):
            rvalue = Y_hat[i][j]
            if rvalue > 1:
                out.write('rvalue = ' + str(rvalue))
                XtestImg[i][j] = ( 255, 0, 0)
            elif (rvalue >= 0.5):
                XtestImg[i][j] = (0, 255, 0)
            elif generateHeatMap and (rvalue > 0.25):
                XtestImg[i][j] = (np.clip(100 * 2 * rvalue, 0, 100) + 150, 100, 100)
            elif generateHeatMap and (rvalue > 0):
                XtestImg[i][j] = (100, 100,  150 + np.clip(100 * 4 * rvalue, 0, 100))
                
    
def SaveInFolder(id, XtestImg, appendText, out_dir):
    im = Image.fromarray(XtestImg)
    imgFile = os.path.join(out_dir, str(id) + appendText + '.jpg')
    im.save(imgFile)

currentDir = os.path.dirname(os.path.realpath(__file__))
modelDL = load_model(os.path.join(currentDir, 'Model.h5'), custom_objects={'dice_loss': CustomLoss.dice_loss})
modelMSE = load_model(os.path.join(currentDir, 'Model_MSE.h5'))

resultdir = os.getcwd()
if platform.system() == 'Linux':
    resultdir = '/var/www/html/results'

mydir = os.path.join(resultdir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(mydir)

num_images = 100
#ds = DataSet.TrainDataGenerator(DataSet.dev_set_count, 'dev', 0.2)
ds = DataSet.TrainDataGenerator(num_images, 'test', 0.2)

devSet = next(ds)
XList = devSet[0]
YList = devSet[1]
Y_hat_DL = modelDL.predict(XList)
Y_hat_MSE = modelMSE.predict(XList)

h = XList[0].shape[0] // 2
w = XList[0].shape[1] // 2
hw = '' #' height=' + str(h) + " width=" + str(w)
htmlfile = os.path.join(mydir, 'index.html')
out = open(htmlfile, 'w')
out.write('<html><body><table border=1><tr><td>Ground Truth</td><td>Predicted Using DiceLoss</td><td>Predicted Lane Using MSE probability > 0.5</td><td>MSE HeatMap</td></tr>')

f1_scores = [0] * 9
for ii in range(XList.shape[0]):
    for jj in range(9):
        f1_scores[jj] = f1_scores[jj] + CustomMetric.compute_f1(YList[ii], Y_hat_DL[ii], (jj + 1) / 10)
    tempDL = np.copy(XList[ii])
    tempMSE = np.copy(XList[ii])
    tempMSE_HM = np.copy(XList[ii])
    
    ApplyLaneDetection(YList[ii], XList[ii])
    ApplyLaneDetection(Y_hat_DL[ii], tempDL)    
    ApplyLaneDetection(Y_hat_MSE[ii], tempMSE)
    ApplyLaneDetection(Y_hat_MSE[ii], tempMSE_HM, True)

    SaveInFolder(ii, XList[ii], '', mydir)
    SaveInFolder(ii, tempDL, '_DL', mydir)
    SaveInFolder(ii, tempMSE, '_MSE', mydir)
    SaveInFolder(ii, tempMSE_HM, '_MSE_HM', mydir)
    
    out.write('<tr>')
    out.write('<td><img src=\'' + str(ii) + '.jpg\'' + hw + '></td>')
    out.write('<td><img src=\'' + str(ii) + '_DL.jpg\'' + hw + '></td>')
    out.write('<td><img src=\'' + str(ii) + '_MSE.jpg\'' + hw + '></td>')
    out.write('<td><img src=\'' + str(ii) + '_MSE_HM.jpg\'' + hw + '></td>')
    out.write('<td>' + str(CustomMetric.compute_f1(YList[ii], Y_hat_DL[ii], 0.5)) + '</td>')
    out.write('</tr>')

avg_f1_scores = [x / num_images for x in f1_scores]
plt.xlabel('probability threshold')
plt.ylabel('f1 score across dev set')

plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], avg_f1_scores)
pltfile = os.path.join(mydir, 'plt_f1_score.png')
plt.savefig(pltfile)
plt.clf()
out.write('<tr><td colspan=4><img src=plt_f1_score.png></td></tr>')

out.write('</table></body></html>')
out.close()

