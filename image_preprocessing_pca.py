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
