
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


#freazing the layers
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
history=model.fit(x=train_batches,validation_data=valid_batches,epochs=3,verbose=2)

#predict
predictions=model.predict(x=test_batches,verbose=0)
model.summary()
#check the classes 
test_batches.classes
test_batches.class_indices

#confusion matrix
cm=confusion_matrix(y_true=test_batches.classes,y_pred=np.argmax(predictions,axis=-1))
print(cm)


#fine-tuning#
# Unfreeze the base model
model.trainable = True

# It's important to recompile your model after you make any changes
# to the `trainable` attribute of any inner layer, so that your changes
# are take into account

model.compile(optimizer=keras.optimizers.Adam(1e-5),loss='categorical_crossentropy',metrics=['accuracy'])

# Train end-to-end. Be careful to stop before you overfit!
history1=model.fit(x=train_batches,validation_data=valid_batches,epochs=30,verbose=2)

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


#
import tensorflow as tf

vgg16 = tf.keras.applications.VGG16()
#flatten layer
flatten_output = tf.keras.backend.function(model.input, model.get_layer('flatten').output)
#fully-connected2 layer    
fc2_output = tf.keras.backend.function(model.input, model.get_layer('fc2').output)
                                          
                                           
image=train_batches[1]
result = flatten_output(image)
                                           
print('Flatten layer outputs:', result)
print('Shape:', result.shape)


######################################################## I did not use this method ########################################
#import image
rot=tf.keras.preprocessing.image.load_img('/content/drive/My Drive/trainn/train/Covid/Covid (1001).png',target_size=(224,224))

#import pil
from PIL import Image

#transform image to numpy array

img = tf.keras.preprocessing.image.array_to_img(rot)
array = tf.keras.preprocessing.image.img_to_array(img)
data_np = np.asarray(array, np.float32)

#tranfrorm numpy to tensor
data_tf = tf.convert_to_tensor(data_np, np.float32)
#prepare image to fit the output
new_image = tf.expand_dims(data_tf,0)

#fully connected layer output 
result_fc2=fc2_output(new_image)
print('FC2 layer outputs:', result_fc2)
print('Shape:', result_fc2.shape)
                                                                              
################################################ FINISH I did not use this method ######################################################


################################################# initial trials ,better go to line 215 to see the 2 methods ################
import tensorflow as tf

vgg16 = tf.keras.applications.VGG16()
flatten_output = tf.keras.backend.function(model.input, model.get_layer('flatten').output)
fc2_output = tf.keras.backend.function(model.input, model.get_layer('fc2').output)


import cv2
images = [cv2.imread(file) for file in glob.glob("/content/drive/My Drive/trainn/train/Covid/*.png")]

y=[]
for i in range(1,11):
    gray=cv2.resize(images[i],(224,224))
    y.append(gray)


list=tf.convert_to_tensor(y)

list2=fc2_output(list)

import pandas as pd

test1=pd.DataFrame(list2)

##############################                FINISH             ################### initial trials ,better go to line 215 to see the 2 methods ################


###########################################################15/7/2020 ( unfreeze the 5 final layer and train them with our images)
############################ FIRST APPROACH #######################################
# let's freeze the feature extractor layers and train with the train_batches the flatten,fc and prediction layer
vgg16_model=keras.applications.vgg16.VGG16()
model= Sequential()
for layer in vgg16_model.layers[:-5]:
    model.add(layer)
for layer in model.layers:
    layer.trainable=False
    
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(units=4096,activation='relu'))
model.add(Dense(units=4096,activation='relu'))
model.add(Dense(units=2,activation='softmax'))    

#train the fine -tuned vgg16 model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#fit our model
model.fit(x=train_batches,validation_data=valid_batches,epochs=20,verbose=2)

##### extract the fc2 layer ("dense6) 
fc2_output = tf.keras.backend.function(model.input, model.get_layer('dense_6').output)

# put the images from train set to extract the features ( pleon ta flatten, fc2, and predict) exoun ekpaideutei me tis fwto moy 
import cv2
images = [cv2.imread(file) for file in glob.glob("/content/drive/My Drive/trainn/train/Covid/*.png")]
y=[]
for i in range(1,200):
    gray=cv2.resize(images[i],(224,224))
    y.append(gray)
list_cov=tf.convert_to_tensor(y)
list11=fc2_output(list_cov)
pd_list_cov=pd.DataFrame(list11)

import cv2
images = [cv2.imread(file) for file in glob.glob("/content/drive/My Drive/trainn/train/Non-Covid/*.png")]
y1=[]
for i in range(1,200):
    gray1=cv2.resize(images[i],(224,224))
    y1.append(gray1)
list_ncov=tf.convert_to_tensor(y1)
list1=fc2_output(list_ncov)
pd_list_ncov=pd.DataFrame(list1)

##two ways  either
freezer=pd.concat([pd_list_cov, pd_list_cov])
#or
greez = pd_list_cov.append(pd_list_cov)

#how to save on colab
from google.colab import files

greez.to_csv('greez.csv')
files.download('greez.csv')

############################################ FINISH 1ST APPROACH 3#########################################




#######################################    SECOND APPROACH              ###########################
########################################## train only the last layer with our dataset ################
#from tensorflow import keras
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Activation,Dense,Flatten,BatchNormalization,Conv2D,MaxPool2D
#from tensorflow.keras.metrics import categorical_crossentropy
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
vgg16_model=keras.applications.vgg16.VGG16()

model= keras.Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable=False

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.add(Dense(units=2,activation='softmax'))


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x=train_batches,validation_data=valid_batches,epochs=20,verbose=2)

fc2_output = tf.keras.backend.function(model.input, model.get_layer('fc2').output)


import cv2
images = [cv2.imread(file) for file in glob.glob("/content/drive/My Drive/trainn/train/Non-Covid/*.png")]
y555=[]
for i in range(1,200):
    gray=cv2.resize(images[i],(224,224))  
    y555.append(gray)
list_555=tf.convert_to_tensor(y555)
list556=fc2_output(list_555)
list_557=pd.DataFrame(list556)




import cv2
images = [cv2.imread(file) for file in glob.glob("/content/drive/My Drive/trainn/train/Non-Covid/*.png")]
y777=[]
for i in range(1,200):
    gray=cv2.resize(images[i],(224,224))  
    y777.append(gray)

list777=tf.convert_to_tensor(y777)
list778=fc2_output(list777)
list779=pd.DataFrame(list778)


first = list779.append(list_557)

###  save "first"  ###
from google.colab import files

first.to_csv('first.csv')
files.download('first.csv')

#plot the performance metrics
history=model.fit(x=train_batches,validation_data=valid_batches,epochs=15,verbose=2)
test_loss, test_acc = model.evaluate(test_batches)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val_accuracy'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

##################################### new exp ##############################   it is saved as 'dokimi' at colab #########

# implement dropout to the pretrained with imagenet model which has learnt with covid dataset the five last layers

vgg16_model=keras.applications.vgg16.VGG16()
model= Sequential()
for layer in vgg16_model.layers[:-5]:
    model.add(layer)
for layer in model.layers:
    layer.trainable=False
    
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
dropout1 = tf.keras.layers.Dropout(0.5)
dropout2 = tf.keras.layers.Dropout(0.5)
model.add(MaxPool2D())
model.add(Flatten())

model.add(Dense(units=4096,activation='relu'))
model.add(dropout1)
model.add(Dense(units=4096,activation='relu'))
model.add(dropout2)
model.add(Dense(units=2,activation='softmax')) 

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(x=train_batches,validation_data=valid_batches,epochs=15,verbose=2)
test_loss, test_acc = model.evaluate(test_batches)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val_accuracy'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
################################# finish new exp ############################################

#################################### calculate metrics ###########

#extract predict classes y of test set ######
yhat_classes = model.predict_classes(test_batches, verbose=0)

#extract actual classes y of test set ######
y_actual=test_batches.classes

# accuracy: (tp + tn) / (p + n)
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report,f1_score
from sklearn.metrics import roc_auc_score , classification_report

accuracy = accuracy_score(y_actual, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_actual, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_actual, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_actual, yhat_classes)
print('F1 score: %f' % f1)

#plot the roc curve

from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report,f1_score
from sklearn.metrics import roc_auc_score , classification_report
from sklearn.metrics import roc_curve,roc_auc_score

auc=roc_auc_score(y_actual, yhat_classes)
print('AUC: %.2f' % auc)
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig("roc curve mushroom.png")
    plt.show()
fpr,tpr,thresholds=roc_curve(y_actual,yhat_classes)
plot_roc_curve(fpr,tpr)

## PLOT THE CONFUSION MATRIX ##
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_actual,yhat_classes)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_lr")
plt.ylabel("y_true_lr")
plt.savefig("confusion matrix mushrooms")
plt.show()


# new dataset compare metrics# 
test_path_2='/content/drive/My Drive/test_2'

test_batches_2=ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path_2,target_size=(224,224),classes=['Covid','Non-Covid'],batch_size=10,shuffle=False)
