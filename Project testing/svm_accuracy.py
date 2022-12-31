import cv2 as cv
import pickle
from sklearn import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np
path = "D:/Engineering/CUFE/3rd Year (Computer) (2022)/First Semester/Image Processing/Projects/data set/data2/"

# Load kmeans model
print("Loading Kmeans model...")
filename1 = 'kmeans_model.sav'
k_means = pickle.load(open(filename1, 'rb'))
n_clusters = 1600
# Load SVM model
print("Success")
print("Loading SVM model...")
filename2 = 'gestures_model.sav'
clf = pickle.load(open(filename2, 'rb'))
print("Success")
img = cv.imread(path + '0/1.jpg')
sift = cv.SIFT_create()
kp, descriptor = sift.detectAndCompute(img, None)
feature_set = np.copy(descriptor)
descriptors = []
descriptors.append(descriptor)
bagOfWords = []
y = []
y.append(1)
for g in range(0, 2):
    if g == 0:
        c = 2
    else:
        c = 1
    for i in range(c, 1251):
        # Read image
        img = cv.imread(path + f'{g}/{i}.jpg')
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
fig = plt.figure()
ax = fig.gca()
X = k_means.predict(descriptors)

disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="predict",
    cmap=plt.cm.coolwarm,
    alpha=0.8,
    ax=ax,
    xlabel="x1",
    ylabel="x2",
)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
plt.show()
