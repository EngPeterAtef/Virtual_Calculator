
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import cluster
import pickle

path = "E:/Koleya/3rd/image project last/captured/"
img = cv.imread(path + '1/1.jpg')
sift = cv.SIFT_create()
kp, descriptor = sift.detectAndCompute(img, None)
feature_set = np.copy(descriptor)
descriptors = []
descriptors.append(descriptor)
bagOfWords = []
y = []
y.append(1)

for g in range(9):
    # if g == 0:
    #     path = "D:/Engineering/CUFE/3rd Year (Computer) (2022)/First Semester/Image Processing/Projects/Training set/2/"
    # else:
    #     path = "D:/Engineering/CUFE/3rd Year (Computer) (2022)/First Semester/Image Processing/Projects/Training set/13/"
    for i in range(2, 501):
        # Read image
        img = cv.imread(path + f'{g}/{i}.jpg')
        # Grayscale
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Binary
        # et, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        # Initialize sift
        sift = cv.SIFT_create()
        # Keypoints, descriptors
        kp, descriptor = sift.detectAndCompute(img, None)
        # Each keypoint has a descriptor with length 128
        print(f"SIFT {g}/{i}")
        if descriptor is None:
            continue
        else:
            descriptors.append(np.array(descriptor))
            feature_set = np.concatenate((feature_set, descriptor), axis=0)
            y.append(g)
# Kmeans clustering on all training set
print(f"Success")
print(f"Running kmeans...")
n_clusters = 1600
np.random.seed(0)
k_means = cluster.KMeans(n_clusters=n_clusters, n_init=4)
k_means.fit(feature_set)
# Produce "bag of words" histogram for each image
# k = 0
print(f"Success")
print(f"Generating bag of words...")
for descriptor in descriptors:
    vq = [0] * n_clusters
    descriptor = k_means.predict(descriptor)
    for feature in descriptor:
        vq[feature] = vq[feature] + 1
    bagOfWords.append(vq)
    # if k < 100:
    #     y.append(2)
    # else:
    #     y.append(13)
    # k = k + 1

print(f"Success")
print(f"Training SVM model...")
# Train the SVM multiclass classification model
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(bagOfWords, y)
# dec = clf.decision_function([[1]])
# dec.shape[1] # 4 classes: 4*3/2 = 6
# clf.decision_function_shape = "ovr"
# dec = clf.decision_function([[1]])
# dec.shape[1] # 4 classes
print(f"Success")
print(f"Saving models")

# save the kmeans model to disk
filename1 = 'kmeans_model.sav'
pickle.dump(k_means, open(filename1, 'wb'))
# save the SVM model to disk
filename2 = 'gestures_model.sav'
pickle.dump(clf, open(filename2, 'wb'))
print(f"Success")
