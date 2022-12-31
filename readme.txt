Libraries:
1. time
2.numpy
3.cv2
4.pickle
5.os
6.threading
7.cv2
8.sklearn.metrics
9.sklearn

How to run the project:
1. Place hand_detection.py, gestures_model.sav, kmeans_model.sav in the same folder
2. run hand_detection.py and calculate!

How to train the models (kmeans and SVM multiclass):
1. open multiple_gestures_training.py
2. Change the path to the path containing the training set folders (each gestures inside a folder, the folder name indicates the label of this gesture)
3. run the script
4. models are saved to disk after running successfully

How to calculate model accuracy:
1. open model_accuracy.py
2. Change the path to the path containing the test set folders (each gestures inside a folder, the folder name indicates the label of this gesture)
3. run the script, output is the accuracy

Test the model on separate images:
1. open model_test.py
2. Change the path to the path containing the test image
3. run the script, output is the predicted class