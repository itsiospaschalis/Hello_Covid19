import numpy as np
import cv2

img=cv2.imread("C://Users//itsios//Desktop//dissertation//train//train_covid//Covid (1).png")

#this makes the image from (202,256,3) to (51712 )
img2= img.reshape((-1,3))

from sklearn.mixture import GaussianMixture as GMM

gmm_model=GMM(n_components=2,covariance_type='diag').fit(img2)

gmm_labels=gmm_model.predict(img2)

original_shape=img.shape
segmented=gmm_labels.reshape(original_shape[0],original_shape[1])
cv2.imwrite("segmented.plant.jpg",segmented)


