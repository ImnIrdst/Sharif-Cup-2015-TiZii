from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet

import os
import cv2
import time
import theano
import numpy as np
import pandas as pd

from PIL import Image
from PIL import ImageOps
from nolearn.lasagne import BatchIterator
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
#%matplotlib inline
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer
PIXELS = 64
CHANNELS = 1
BATCH  = 5000

def float32(k):
    return np.cast['float32'](k)


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weighets)
            raise StopIteration()

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

import pickle 
net = pickle.load( open( 'pickles/Deep5000SharifCup.pickle', "rb" ) ) # pickle-path

PIXELS_CROP = 40
PIXELS_TRAIN = 32
STEP = 25
UPPER = 150
LOWER = 400

def slidingWindow(img):
    for scale in range(6, 5, -1):
        w = (PIXELS_CROP*scale)/6
        for y in range(UPPER, LOWER - w, STEP):
            for x in range(LEFT, RIGHT - w, STEP):
                image = img.crop((x,y, x + w, y + w))
                image = image.resize((PIXELS_TRAIN, PIXELS_TRAIN), Image.ANTIALIAS)
                yield (x,y,w,image)

def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        #print i
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in xrange(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the suppression list
        idxs = np.delete(idxs, suppress)
    return boxes[pick]
    # return only the bou


def predictonImage(ImageName):
    Coords = []
    Widths = []
    Crops = []
    ImageCrop = Image.open(ImageName)
    #ImageCrop = ImageCrop.convert('RGB')
    ImageCrop = ImageCrop.resize((600, 350), Image.ANTIALIAS)

    startTime = time.clock()
    for x,y,w,img  in slidingWindow(ImageCrop):
        Crops.append(img)
        Coords.append((x,y))
        Widths.append(w)

    CROPSCNT = len(Crops)
    print 'Subrects', CROPSCNT

    testCrops = np.zeros((CROPSCNT,CHANNELS, PIXELS_TRAIN, PIXELS_TRAIN), dtype='float32')

    for i in range(0,CROPSCNT):
        img = Crops[i]
        img = img.resize((PIXELS_TRAIN, PIXELS_TRAIN), Image.ANTIALIAS)
        img = np.asarray(img, dtype = 'float32') / 255.
        testCrops[i] = img.reshape(CHANNELS, PIXELS_TRAIN, PIXELS_TRAIN)


    print 'preprocess time:', time.clock() - startTime
    #print testCrops[0]
    ypred = net.predict(testCrops)
    #print ypred

    print 'predict time: ', time.clock() - startTime
    ones = 0
    for i in range(0, len(ypred)):
        if ypred[i] == 1 :
            ones += 1
    print 'Ones: ', ones

    #ploot = plt.imshow(Crops[23])
    #ploot = plt.imshow(ImageCrop)

    boundingBoxes = np.zeros((ones, 4))

    ii = 0
    for i in range(0,CROPSCNT):
        if ypred[i] == 1:
            boundingBoxes[ii] = [Coords[i][0], Coords[i][1], Coords[i][0] + Widths[i], Coords[i][1] + Widths[i]]
            ii += 1
    retImage = np.asarray(ImageCrop)
    boxes = non_max_suppression_slow(boundingBoxes, 0.005)
    cntt = len(boxes)
    print 'true Ones: ', len(boxes)
    #print type(retImage)
    for i in range(0,len(boxes)):
        w = int(boxes[i][2] - boxes[i][0])
        x = int(boxes[i][0])
        y = int(boxes[i][1])
        if x == 64. and y == 15.:
        	continue
        #print x,y,x+w,y+w
        cv2.rectangle(retImage,(x,y),(x+w,y+w),(0,255,0),3)
        print boxes[i]
    return retImage, cntt
    
PIXELS_CROP = 140
PIXELS_TRAIN = 32
STEP = 32
UPPER = 15
LOWER = 335
LEFT = 32
RIGHT = 600 - 16
pathh = "/media/iman/4E903A9B903A8A0B/Work/GitHub/Vision-Python-Snippits-2015-TiZii/data/TEST/2/" # video-path
imgs = os.listdir(pathh)
for i in range(0, len(imgs)):
    while len(imgs[i]) < 8 :
        imgs[i] = '0' + imgs[i]
imgs.sort()
for i in range(0, len(imgs)):
    while imgs[i][0] == '0' and imgs[i][1] != '.':
        imgs[i] = imgs[i][1:] 

f = 0
s = 0
p = 0
cnts = 0
fIle = open('1.txt', 'w')
for i in range(0, len(imgs)):
    #print imgs[i]
    img, cntf = predictonImage(pathh + imgs[i]) #write to file
    fIle.write("F" + str(f) + "," + str(cntf)+"\n");
    if p%8 == 0 :
        cnts += cntf
    if (f+1)%150 == 0 and f != 0:
        "S"  + str(s) + "," + str(cnts)
        fIle.write("S"  + str(s) + "," + str(cnts) + '\n');
        cnts = 0
        s    = s + 1
    p += 1
    f += 1

    #print img.shape
    #print img
    print ' \n'

    cv2.imshow("asd",img)
    cv2.waitKey(100)
    
cv2.waitKey(10)
cv2.destroyAllWindows()
cv2.waitKey(10)