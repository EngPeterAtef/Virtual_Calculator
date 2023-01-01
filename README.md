# Virtual_Keyboard
An augmented reality project using image processing and machine learning

### Demo Link <a href="https://youtu.be/EqCD2lAiloM">Link</a> <br/>

### Libraries:
1. time <br/>
2.numpy <br/>
3.cv2 <br/>
4.pickle<br/>
5.os<br/>
6.threading<br/>
7.sklearn.metrics<br/>
8.sklearn<br/>

### How to run the project:
1. Place hand_detection.py, gestures_model.sav, kmeans_model.sav in the same folder<br/>
2. run hand_detection.py and calculate!<br/><br/>

### How to train the models (kmeans and SVM multiclass):
1. open multiple_gestures_training.py<br/>
2. Change the path to the path containing the training set folders (each gestures inside a folder, the folder name indicates the label of this gesture)<br/>
3. run the script<br/>
4. models are saved to disk after running successfully<br/><br/>

### How to calculate model accuracy:
1. open model_accuracy.py<br/>
2. Change the path to the path containing the test set folders (each gestures inside a folder, the folder name indicates the label of this gesture)<br/>
3. run the script, output is the accuracy<br/><br/>

### Test the model on separate images:
1. open model_test.py<br/>
2. Change the path to the path containing the test image<br/>
3. run the script, output is the predicted class<br/><br/>

### Used Algorithms :
1- Gaussian filter on the taken picture to eliminate the noise.<br/>
2- Adaptive (local) thresholding on skin colour :we put lower and upper values for the skin colour hsv then we threshold on this range of values.<br/>
3- Sift algorithm to extract the features (key points) of the thresholded hand and create a descriptor of length 128 for each feature<br/>
4- kmeans with 1600 clusters on the features of all the training set .<br/>
5-the we generate a vector(bag of words)for each picture containing number of the features belonging to each cluster.<br/>
6- then we generate the model using SVM multiclass classification model, the input of the model is the bag of words.<br/>
7- while running the application, we get the picture of the hand gesture ,perform adaptive thresholding, perform sift algorithm , predict kmeans clusters of the picture ,generate bag of words and we get the class form the
svm model to which the gesture belongs.<br/><br/>

### Experiment Results and Analysis :
At first, we tried to train the models (Kmeans and SVM) with 100 images
for each gesture (having 11 gestures) and 100 clusters. We noticed poor
prediction results due to small dataset and cluster size, so we captured
more training dataset till we reached 1250 for each gesture, and increased the cluster number to 1600, where we noticed a high
improvement in accuracy <br/>
Our data is separated into 80% training set and 20% test set to calculate
the SVM model accuracy<br/><br/>

### Level of variety for test cases used in experimental results:
We captured the test set ourselves to ensure that the sizes and colours
of our hands are different, We can change the background for the 10
gestures with the limitation that the background does not contain a
brown colour that can be mistaken to the skin colour,and the light is not
very weak nor strong, the size of the hand can change (no limitation on
the skin colour nor the hand size)<br/><br/>

### Accuracy calculated from test set= 0.8956363636363637
### Complete analysis for the system elaborating points of strengths and weakness:
-The application is able to detect the hand gestures existing in the
dataset, as we extend the dataset and we introduce hand size variety ;
the application becomes more accurate in detecting the gesture and less
sensitive to errors in the test picture. <br/>
-We specify a window in a fixed location for the user to put his hand in it
in the beginning of the run to reduce the amount of objects that are
needed to be eliminated from the image , then we apply adaptive
thresholding based on a range of skin colour values to get the hand out
of the small window without other objects … but as we use a certain
range of skin colour values our algorithm can sometimes get confused
between brown objects and the hand itself.<br/>
-The user has to fix his hand in the box with the intended gesture for 10
seconds…..as the user may not be able to fix his hand from the start of
the timer , the first 5 seconds is a period for the user to put his hand in the window and make the gesture , and the next 5 seconds is used to
detect the gesture.
But to overcome the instability of the detection due to the light and the
movements of the user, we count the number of switches between the
gestures in the second 5 seconds and get the gesture with the max
number of occurrences .<br/><br/>

### Results:
<b>We generated an application that detects the number from 0 to 10 and perform
the following operations : + , - , / , * , ^ .</b>
![image](https://user-images.githubusercontent.com/75852529/210168169-7d44c7a6-5244-4b24-af21-9377461a7ddc.png)
<br/><br/>
### Test Cases:
![image](https://user-images.githubusercontent.com/75852529/210168239-e156db56-27fc-4ac9-8ea1-f02cd99ab97c.png)
![image](https://user-images.githubusercontent.com/75852529/210168242-15717997-f0d0-4bad-864a-ba8ccce89a6b.png)
![image](https://user-images.githubusercontent.com/75852529/210168245-22bf2e96-57f3-44cb-8f42-a67b26c21f64.png)
![image](https://user-images.githubusercontent.com/75852529/210168253-1fa3326c-7f52-4c5d-b2fe-3733cc586543.png)
<br/><br/>
### References:
<a href="https://link.springer.com/content/pdf/10.1007/978-3-642-35473-1.pdf?pdf=button"> reference link </a>
