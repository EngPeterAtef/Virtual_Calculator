
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import cluster
import pickle

path = "D:/Engineering/CUFE/3rd Year (Computer) (2022)/First Semester/Image Processing/Projects/Training set/2/"
img = cv.imread(path + '1.jpg')
sift = cv.SIFT_create()
kp, descriptor = sift.detectAndCompute(img, None)
feature_set = np.copy(descriptor)
descriptors = []
descriptors.append(descriptor)
bagOfWords = []
y = []
for g in range(2):
    if g == 0:
        path = "D:/Engineering/CUFE/3rd Year (Computer) (2022)/First Semester/Image Processing/Projects/Training set/2/"
    else:
        path = "D:/Engineering/CUFE/3rd Year (Computer) (2022)/First Semester/Image Processing/Projects/Training set/13/"
    for i in range(2, 101):
        # Read image
        img = cv.imread(path + f'{i}.jpg')
        # Grayscale
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Binary
        # et, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        # Initialize sift
        sift = cv.SIFT_create()
        # Keypoints, descriptors
        kp, descriptor = sift.detectAndCompute(img, None)
        descriptors.append(np.array(descriptor))
        # Each keypoint has a descriptor with length 128
        feature_set = np.concatenate((feature_set, descriptor), axis=0)
# Kmeans clustering on all training set
n_clusters = 100
np.random.seed(0)
k_means = cluster.KMeans(n_clusters=n_clusters, n_init=4)
k_means.fit(feature_set)
# Produce "bag of words" histogram for each image
k = 0
for descriptor in descriptors:
    vq = [0] * 100
    descriptor = k_means.predict(descriptor)
    for feature in descriptor:
        vq[feature] = vq[feature] + 1
    bagOfWords.append(vq)
    if k < 100:
        y.append(2)
    else:
        y.append(13)
    k = k + 1

# Train the SVM multiclass classification model
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(bagOfWords, y)
# dec = clf.decision_function([[1]])
# dec.shape[1] # 4 classes: 4*3/2 = 6
# clf.decision_function_shape = "ovr"
# dec = clf.decision_function([[1]])
# dec.shape[1] # 4 classes

# save the kmeans model to disk
filename1 = 'kmeans_model.sav'
pickle.dump(k_means, open(filename1, 'wb'))
# save the SVM model to disk
filename2 = 'gestures_model.sav'
pickle.dump(clf, open(filename2, 'wb'))
