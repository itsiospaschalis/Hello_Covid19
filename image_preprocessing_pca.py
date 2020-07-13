import cv2
import glob
import numpy as np

images = [cv2.imread(file) for file in glob.glob("C:\\Users\\itsios\\Desktop\\dissertation\\COVID\\*.png")]

X=[]

for i in range(1,1252):
    gray=cv2.resize(images[i],(224,224))
    gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    gray=gray.reshape((-1,1))
    X.append(gray)
      

y=[]

for i in range(1,1252):
    gray=cv2.resize(images[i],(224,224))
    gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    gray = gray.flatten()
    y.append(gray)


#transform it to a pandas dataframe

import pandas as pd
df = pd.DataFrame(y)  

#PCA 

from sklearn . decomposition import PCA

pca = PCA(n_components =5)
pca.fit ( df )
Coeff = pca. components_
print(Coeff)

%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(list(pca.explained_variance_ratio_),'-o')
plt.title('Explained variance ratio as function of PCA components')
plt.ylabel('Explained variance ratio')
plt.xlabel('Component')
plt.savefig("Figure2.png")
plt.show()



#extract FC2 LAYER



from keras.applications.vgg16 import VGG16
# example of using the vgg16 model as a feature extraction model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model
model1 = VGG16()
model1.layers.pop()
model1 = Model(inputs=model1.inputs, outputs=model1.layers[-1].output)
features = model1.predict(train_batches)

a=(features[1:500])
test2=pd.DataFrame(a)
#now test2 and df has 500 samples
#PCA for extracted fc2 layer

from sklearn . decomposition import PCA

pca = PCA(n_components =5)
pca.fit ( test2 )
Coeff = pca. components_
print(Coeff)

%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(list(pca.explained_variance_ratio_),'-o')
plt.title('Explained variance ratio as function of PCA components')
plt.ylabel('Explained variance ratio')
plt.xlabel('Component')
plt.savefig("Figure2.png")
plt.show()


#see how many components we need in order to achive 70 % variance explainability
https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Image_Reconstruction_and_such.ipynb
# Indices corresponding to the first occurrence are returned with the np.argmax function
# Adding 1 to the end of value in list as principal components start from 1 and indexes start from 0 (np.argmax)
componentsVariance = [499, np.argmax(cum_var_exp > 99) + 1, np.argmax(cum_var_exp > 95) + 1, np.argmax(cum_var_exp > 90) + 1, np.argmax(cum_var_exp >= 70) + 1]


######################################################################## project the original and compressed image #################################################################
import cv2
import glob
train_x_covid = [cv2.imread(file) for file in glob.glob("/content/drive/My Drive/trainn/train/Covid/*.png")]
train_x_noncovid=[cv2.imread(file) for file in glob.glob("/content/drive/My Drive/trainn/train/Non-Covid/*.png")]
train_x_covid=np.array(train_x_covid)
train_x_noncovid=np.array(train_x_noncovid)

train_y_covid=[1] * 500
train_y_noncovid=[0]*500

train_x=np.concatenate((train_x_covid,train_x_noncovid))
train_y=np.concatenate((train_y_covid,train_y_noncovid))


y=[]

for i in range(1,1000):
    gray=cv2.resize(train_x[i],(224,224))
    gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    gray = gray.flatten()
    y.append(gray)



from sklearn.decomposition import PCA
pca_dims = PCA()
pca_dims.fit(y)
cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.70) + 1

pca = PCA(n_components=d)
X_reduced = pca.fit_transform(y)
X_recovered = pca.inverse_transform(X_reduced)

f = plt.figure()
f.add_subplot(1,2, 1)
plt.title("original")
plt.imshow(y[0].reshape((224,224)))
f.add_subplot(1,2, 2)

plt.title("PCA compressed")
plt.imshow(X_recovered[0].reshape((224,224)))
plt.show(block=True)
