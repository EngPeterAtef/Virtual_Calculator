import cv2
import pickle
from sklearn.metrics import accuracy_score

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
y_true = []
y_predict = []
path = "D:/Engineering/CUFE/3rd Year (Computer) (2022)/First Semester/Image Processing/Projects/data set/data set/"
for g in range(0, 2):
    for i in range(1001, 1251):
        # Read image
        img = cv2.imread(path + f'{g}/{i}.jpg')
        # Feature extraction
        sift = cv2.SIFT_create()
        kp, descriptor = sift.detectAndCompute(img, None)
        # Produce "bag of words" vector
        descriptor = k_means.predict(descriptor)
        print(f"SIFT {g}/{i}")
        vq = [0] * n_clusters
        for feature in descriptor:
            vq[feature] = vq[feature] + 1  # load the model from disk
        y_true.append(g)
        # Predict the result
        y_predict.append(clf.predict([vq]))
accuracy = accuracy_score(y_true, y_predict)
print(accuracy)
