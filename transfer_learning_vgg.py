
#import libraries

from numpy.random import seed
seed(101)
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense,Flatten,BatchNormalization,Conv2D,MaxPool2D
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
warnings.simplefilter(action='ignore',category=FutureWarning)

#organize data into train,valid,test
import os

os.chdir('C:\\Users\\itsios\\Desktop\\dissertation\\data\\trainn')

if os.path.isdir('train\\Covid') is False:
    os.makedirs('train\\Covid')
    os.makedirs('train\\Non-Covid')
    os.makedirs('valid\\Covid')
    os.makedirs('valid\\Non-Covid')
    os.makedirs('test\\Covid')
    os.makedirs('test\\Non-Covid')
    
    for c in random.sample(glob.glob('Covid*'),500):
        shutil.move(c,'train\\Covid')
    for c in random.sample(glob.glob('Non-Covid*'),500):
        shutil.move(c,'train\\Non-Covid')
    for c in random.sample(glob.glob('Covid*'),100):
        shutil.move(c,'valid\\Covid')
    for c in random.sample(glob.glob('Non-Covid*'),100):
        shutil.move(c,'valid\\Non-Covid')
    for c in random.sample(glob.glob('Covid*'),50):
        shutil.move(c,'test\\Covid')
    for c in random.sample(glob.glob('Non-Covid*'),50):
        shutil.move(c,'test\\Non-Covid')

train_path='C:\\Users\\itsios\\Desktop\\dissertation\\data\\trainn\\train'
valid_path='C:\\Users\itsios\\Desktop\\dissertation\\data\\trainn\\valid'
test_path='C:\\Users\itsios\\Desktop\\dissertation\\data\\trainn\\test'

train_batches=ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path,target_size=(224,224),classes=['Covid','Non-Covid'],batch_size=10)
valid_batches=ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path,target_size=(224,224),classes=['Covid','Non-Covid'],batch_size=10)
test_batches=ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path,target_size=(224,224),classes=['Covid','Non-Covid'],batch_size=10,shuffle=False)

# import the vgg16 architecture
vgg16_model=keras.applications.vgg16.VGG16()
vgg16_model.summary()

# put out the last layer with the predictions
model= tf.keras.Sequential()

#or
model= keras.Sequential()

#freeze the last layer of vgg16 which has 1000 labels
for layer in vgg16_model.layers[:-1]:
    model.add(layer)


#transfer learning
for layer in model.layers:
    layer.trainable=False
    
#covid or non-covid
model.add(Dense(units=2,activation='softmax'))

#run the next command if you want to load the greyscale weights
checkpoint_path = "C:\\Users\\itsios\\Desktop\\dissertation\\checkpoints\\model.ckpt-1495066"
model.load_weights(checkpoint_path)

#now our model has only 2 classes instead of 1000 of
#vgg16 and the trainable parameters reduced and 
#are only for the last fcl

model.summary()

#train the fine -tuned vgg16 model

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#fit our model
model.fit(x=train_batches,validation_data=valid_batches,epochs=3,verbose=2)

#predict
predictions=model.predict(x=test_batches,verbose=0)
model.summary()
#check the classes 
test_batches.classes
test_batches.class_indices

#confusion matrix
cm=confusion_matrix(y_true=test_batches.classes,y_pred=np.argmax(predictions,axis=-1))
print(cm)


#to improve the results we can do DATA AUGMENTATION
#MORE AGGRESIVE DROPOUT
#L1 AND L2 REGULARIZATION
#FINE TUNING ONE MORE CONVOLUTIONAL BLOCK

#get the layers output (tensors)
flatten=model.layers[18].output
fc_layer=model.layers[20].output


#extract the weights of the fc2 layer and save it
maybe=model.layers[21].get_weights()
maybe2=maybe[0]
