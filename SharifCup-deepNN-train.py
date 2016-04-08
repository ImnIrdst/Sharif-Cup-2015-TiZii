from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet

import os
import cv2
import theano
import numpy as np
import pandas as pd


from PIL import Image
from PIL import ImageOps

from nolearn.lasagne import BatchIterator
from sklearn.utils import shuffle

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer
HPIXELS = 32
VPIXELS = 32
CHANNELS = 1
BATCH  = 7000

path = 'data/Car/'
imgs = os.listdir(path)
print imgs[:5]

# Creating images (fake only 3 real images) just for demonstration

X = np.zeros((BATCH,CHANNELS* HPIXELS* VPIXELS), dtype='float32')
y = np.zeros(BATCH)

for i in range(0,len(imgs)):
    name = imgs[i]
    img = Image.open(path + name)
    img = img.resize((HPIXELS, VPIXELS), Image.ANTIALIAS)
    img = img.convert('L')
    #prinbt img.size
    img = np.asarray(img, dtype = 'float32') / 255.
    
    img = img.reshape(CHANNELS* HPIXELS* VPIXELS)
    X[i] = img

    y[i] = int(name[0])
    if i%400==0: 
        print name[0], ' ', name
        
X, y = shuffle(X, y, random_state=42)  # shuffle train

tmp = np.zeros((BATCH,CHANNELS, HPIXELS, VPIXELS), dtype='float32')
for i in range(0,BATCH):
  tmp[i] = X[i].reshape(CHANNELS,HPIXELS,VPIXELS)

X = tmp

def float32(k):
    return np.cast['float32'](k)


class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        Xb = Xb.astype(np.float32)
        yb = yb.astype(np.int32)
        return Xb, yb

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
            nn.load_params_from(self.best_weights)
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

net = NeuralNet(
      layers=[
          ('input', layers.InputLayer),
          ('hidden1', layers.DenseLayer),
          ('dropout1', layers.DropoutLayer),
          ('hidden2', layers.DenseLayer),
          ('dropout2', layers.DropoutLayer),
          ('hidden3', layers.DenseLayer),
          ('dropout3', layers.DropoutLayer),
          ('hidden4', layers.DenseLayer),
          ('output', layers.DenseLayer),
          ],
      input_shape=(None, CHANNELS, HPIXELS, VPIXELS),
      hidden1_num_units=500,
      dropout1_p=0.5,
      hidden2_num_units=400,
      dropout2_p=0.5,
      hidden3_num_units=100,
      dropout3_p=0.5,
      hidden4_num_units=500,
      output_num_units=2, output_nonlinearity=nonlinearities.softmax,

      update_learning_rate=theano.shared(float32(0.03)),
      update_momentum=theano.shared(float32(0.9)),

      regression=False,
   
      on_epoch_finished=[
          AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
          AdjustVariable('update_momentum', start=0.9, stop=0.999),
          EarlyStopping(patience=200),
          ],
      max_epochs=200,
      verbose=1,
      )

X = X.astype(np.float32)
y = y.astype(np.int32)
print(X.shape)
print(y.shape)
net.fit(X, y)

import cPickle as pickle
with open('pickles/deep5000SharifCup-final.pickle', 'wb') as f:
    pickle.dump(net, f, -1)


'''
/media/iman/4E903A9B903A8A0B/Mix/Work/GitHub/Vision-Python-Snippits-2015-TiZii/src/lasagne/lasagne/init.py:86: UserWarning: The uniform initializer no longer uses Glorot et al.'s approach to determine the bounds, but defaults to the range (-0.01, 0.01) instead. Please use the new GlorotUniform initializer to get the old behavior. GlorotUniform is now the default for all layers.
  warnings.warn("The uniform initializer no longer uses Glorot et al.'s "
  input               (None, 3, 64, 64)     produces   12288 outputs
  conv1               (None, 16, 62, 62)    produces   61504 outputs
  pool1               (None, 16, 31, 31)    produces   15376 outputs
  dropout1            (None, 16, 31, 31)    produces   15376 outputs
  conv2               (None, 32, 30, 30)    produces   28800 outputs
  pool2               (None, 32, 15, 15)    produces    7200 outputs
  dropout2            (None, 32, 15, 15)    produces    7200 outputs
  conv3               (None, 64, 14, 14)    produces   12544 outputs
  pool3               (None, 64, 7, 7)      produces    3136 outputs
  dropout3            (None, 64, 7, 7)      produces    3136 outputs
  hidden4             (None, 500)           produces     500 outputs
  dropout4            (None, 500)           produces     500 outputs
  hidden5             (None, 500)           produces     500 outputs
  output              (None, 2)             produces       2 outputs
  epoch    train loss    valid loss    train/val    valid acc  dur
-------  ------------  ------------  -----------  -----------  ------
      1       0.55908       0.46141      1.21169      0.78741  80.95s
      2       0.39363       0.44992      0.87488      0.79890  80.96s
      3       0.34718       0.32131      1.08053      0.89145  80.98s
      4       0.27288       0.41427      0.65871      0.85246  81.80s
      5       0.21820       0.29857      0.73080      0.90144  81.22s
      6       0.18713       0.14923      1.25396      0.94524  80.99s
      7       0.14641       0.18848      0.77676      0.93870  81.09s
      8       0.12841       0.28169      0.45587      0.89213  81.15s
      9       0.20185       0.16927      1.19247      0.92713  81.07s
     10       0.15876       0.13758      1.15391      0.95200  80.91s
     11       0.13415       0.12615      1.06349      0.95080  80.98s
     12       0.11984       0.19565      0.61254      0.92668  81.00s
     13       0.11344       0.10395      1.09125      0.95906  81.01s
     14       0.10644       0.12738      0.83556      0.95200  81.14s
     15       0.09640       0.10909      0.88370      0.95613  81.11s
     16       0.09871       0.08949      1.10313      0.96439  81.05s
     17       0.09304       0.09876      0.94207      0.96342  81.19s
     18       0.08445       0.11245      0.75098      0.95080  81.07s
     19       0.08223       0.09032      0.91042      0.96830  81.06s
     20       0.07698       0.08138      0.94593      0.96950  81.13s
     21       0.06852       0.06244      1.09727      0.97611  81.18s
     22       0.06204       0.06838      0.90732      0.97514  81.35s
     23       0.05444       0.07202      0.75589      0.97025  81.35s
     24       0.05446       0.07482      0.72789      0.97318  81.50s
     25       0.05559       0.09182      0.60540      0.96514  81.46s
     26       0.05019       0.05783      0.86797      0.98024  81.43s
     27       0.05256       0.06550      0.80238      0.97709  81.44s
     28       0.04391       0.05766      0.76149      0.98122  81.76s
     29       0.04227       0.10623      0.39796      0.96199  81.14s
     30       0.04379       0.06177      0.70883      0.98220  80.95s
     31       0.03568       0.05491      0.64986      0.98145  81.49s
     32       0.03399       0.07494      0.45353      0.97318  81.39s
     33       0.03379       0.06174      0.54733      0.98002  81.12s
     34       0.03130       0.07131      0.43892      0.97806  81.25s
     35       0.03668       0.06146      0.59678      0.97806  81.00s
     36       0.03036       0.04634      0.65510      0.98881  81.04s
     37       0.03005       0.04939      0.60850      0.98535  81.09s
     38       0.02660       0.05645      0.47121      0.98220  81.95s
     39       0.03936       0.05820      0.67629      0.98295  81.34s
     40       0.02739       0.04888      0.56039      0.98588  81.23s
     41       0.03231       0.06892      0.46877      0.97927  81.28s
     42       0.02613       0.04891      0.53437      0.98513  81.26s
     43       0.03175       0.06593      0.48155      0.97754  81.31s
     44       0.02985       0.05213      0.57259      0.98490  81.46s
     45       0.02224       0.05697      0.39036      0.98317  81.37s
     46       0.02837       0.04292      0.66096      0.98610  81.40s
     47       0.02123       0.05074      0.41847      0.98415  81.25s
     48       0.02362       0.04914      0.48063      0.98828  81.40s
     49       0.02668       0.04430      0.60220      0.98730  81.47s
     50       0.02364       0.05151      0.45894      0.98730  81.58s
iman@IMAN-E530:/media/iman/4E903A9B903A8A0B/Mix/Work/GitHub/Vision-Python-Snippits-2015-TiZii
'''


'''
net = NeuralNet(
      layers=[
          ('input', layers.InputLayer),
          ('conv1', Conv2DLayer),
          ('pool1', MaxPool2DLayer),
          ('dropout1', layers.DropoutLayer),
          ('conv2', Conv2DLayer),
          ('pool2', MaxPool2DLayer),
          ('dropout2', layers.DropoutLayer),
          ('conv3', Conv2DLayer),
          ('pool3', MaxPool2DLayer),
          ('dropout3', layers.DropoutLayer),
          ('hidden4', layers.DenseLayer),
          ('dropout4', layers.DropoutLayer),
          ('hidden5', layers.DenseLayer),
          ('output', layers.DenseLayer),
          ],
      input_shape=(None, 3, PIXELS, PIXELS),
      conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
      dropout1_p=0.1,
      conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
      dropout2_p=0.2,
      conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
      dropout3_p=0.3,
      hidden4_num_units=1000,
      dropout4_p=0.5,
      hidden5_num_units=1000,
      output_num_units=2, output_nonlinearity=nonlinearities.softmax,

      update_learning_rate=theano.shared(float32(0.03)),
      update_momentum=theano.shared(float32(0.9)),

      regression=False,
      batch_iterator_train=FlipBatchIterator(batch_size=128),
      on_epoch_finished=[
          AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
          AdjustVariable('update_momentum', start=0.9, stop=0.999),
          EarlyStopping(patience=200),
          ],
      max_epochs=50,
      verbose=1,
      )
'''
