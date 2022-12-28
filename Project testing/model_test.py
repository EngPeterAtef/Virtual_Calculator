import cv2 as cv
import pickle
path = "D:/Engineering/CUFE/3rd Year (Computer) (2022)/First Semester/Image Processing/Projects/Training set/2/"
# Load kmeans model
filename1 = 'kmeans_model.sav'
k_means = pickle.load(open(filename1, 'rb'))
# Load SVM model
filename2 = 'gestures_model.sav'
clf = pickle.load(open(filename2, 'rb'))
# Read image
img = cv.imread(path + '106.jpg')
# Gray
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Binary
et, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# Feature extraction
sift = cv.SIFT_create()
kp, descriptor = sift.detectAndCompute(img, None)
# Produce "bag of words" vector
descriptor = k_means.predict(descriptor)
vq = [0] * 100
for feature in descriptor:
    vq[feature] = vq[feature] + 1  # load the model from disk
# Predict the result
result = clf.predict([vq])
print(result)
