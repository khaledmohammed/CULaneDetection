import numpy as np
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import os, sys, datetime, platform
import DataSet
from PIL import Image
import CustomMetric
import CustomLoss


def ApplyLaneDetection(Y_hat, XtestImg):
    for i in range(Y_hat.shape[0]):
        for j in range(Y_hat.shape[1]):
            rvalue = Y_hat[i][j]
            if (rvalue>0.2):
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
model = load_model(os.path.join(currentDir, 'Model.h5'), custom_objects={'dice_loss': CustomLoss.dice_loss})

resultdir = os.getcwd()
if platform.system() == 'Linux':
    resultdir = '/var/www/html/results'

mydir = os.path.join(resultdir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(mydir)


#ds = DataSet.TrainDataGenerator(DataSet.dev_set_count, 'dev', 0.2)
ds = DataSet.TrainDataGenerator(10, 'test', 0.2)

devSet = next(ds)
XList = devSet[0]
YList = devSet[1]
Y_hat = model.predict(XList)

h = XList[0].shape[0] // 2
w = XList[0].shape[1] // 2
hw = '' #' height=' + str(h) + " width=" + str(w)

htmlfile = os.path.join(mydir, 'index.html')
out = open(htmlfile, 'w')
out.write('<html><body><table border=1><tr><td>Predicted Lane</td><td>Ground Truth</td></tr>')

f1_scores = [0] * 9
for ii in range(XList.shape[0]):
    for jj in range(9):
        f1_scores[jj] = f1_scores[jj] + CustomMetric.compute_f1(YList[ii], Y_hat[ii], (jj+1)/10)
    tempX = np.copy(XList[ii])
    print(ii, np.count_nonzero(Y_hat[ii]))
    ApplyLaneDetection(Y_hat[ii], XList[ii])    
    ApplyLaneDetection(YList[ii], tempX)
    SaveInFolder(ii, XList[ii], tempX, mydir)
    out.write('<tr><td><img src=\'' + str(ii) + '.jpg\'' + hw + '></td><td><img src=\'' + str(ii)+'_Y.jpg\'' + hw + '></td>')        
    out.write('</tr>')

avg_f1_scores = [x / 10 for x in f1_scores]
plt.xlabel('probability threshold')
plt.ylabel('f1 score across dev set')
#ax.set_title('Choosing ')
plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], avg_f1_scores)
pltfile = os.path.join(mydir, 'plt_f1_score.png')
plt.savefig(pltfile)
plt.clf()
out.write('<tr><td colspan=2><img src=plt_f1_score.png></td></tr>')

out.write('</table></body></html>')
out.close()

