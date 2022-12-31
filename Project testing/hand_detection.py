import time
import numpy as np
import cv2
import pickle
import os
import threading

operand1 = operation = operand2 = timer = operationStr = "?"
inputsCount = 0
finalResult = "?"
maxTime = 10


def fun():
    # print("callback func")
    global operand1, operation, operand2, timer, inputsCount, result, finalResult, start_time
    # print(f"{inputsCount}")
    inputsCount = inputsCount + 1
    # print(
    #     f"operand1= {operand1}, operation = {operation}, operand2 = {operand2}")
    if(inputsCount < 4):
        timer = threading.Timer(maxTime, fun)
        start_time = time.time()
        timer.start()
        # print ("Running for : %s seconds"%(time.time()-start_time))
        print("after calling start_time.start()")
    if(inputsCount == 4):
        inputsCount = 0
        timer = threading.Timer(maxTime, fun)
        start_time = time.time()
        timer.start()
        # print ("Running for : %s seconds"%(time.time()-start_time))
        print("after calling start_time.start()")


# Position of ROI of hand thresholding
top, right, bottom, left = 350, 490, 565, 730
cap = cv2.VideoCapture(0)
# --------------------Capture dataset---------------------
index = 751
capture = False
path = "E:/Koleya/3rd/image project last/captured/giza"
# operationsArr = np.array([])


def getOperationString(operation):
    if operation == 6:
        operationStr = "+"
    elif operation == 7:
        operationStr = "-"
    elif operation == 8:
        operationStr = "*"
    elif operation == 9:
        operationStr = "/"
    elif operation == 10:
        operationStr = "^"
    else:  # default
        operationStr = "?"
    return operationStr


def getThresholdedHand(frame, roi):
    global top, right, bottom, left, index, capture
    # Draw rectangle to indicate the area in which we initialize hand positon for the first time
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    # Convert image to HSV
    hsvim = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Lower boundary of skin color in HSV
    lower = np.array([0, 48, 80], dtype="uint8")
    # Upper boundary of skin color in HSV
    upper = np.array([20, 255, 255], dtype="uint8")
    skinMask = cv2.inRange(hsvim, lower, upper)

    # Gaussian filter (blur) to remove noise
    skinMask = cv2.GaussianBlur(skinMask, (17, 17), 0)

    # get thresholded image
    # ret, thresh1 = cv2.threshold(
    # skinMask, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh1 = cv2.adaptiveThreshold(
        skinMask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 355, 5)
    # Show thresholded hand
    cv2.imshow('Hand threshold', thresh1)
    # Capture dataset
    if capture:
        cv2.imwrite(os.path.join(path, f'{index}.jpg'), thresh1)
        print(index)
        index = index + 1
    return thresh1


# -----------------------------LOAD MODELS---------------------------
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

timer = threading.Timer(maxTime, fun)
start_time = time.time()
timer.start()

operand1Arr = np.array([])
# ----------------------------------------
# ----------------MAIN LOOP---------------
# ----------------------------------------
print("Running for : %s seconds" % (time.time()-start_time))
while True:
    # READ FRAME
    success, img = cap.read()
    img = cv2.resize(img, (1000, 600))
    img = cv2.flip(img, 1)
    # ---------------------------DRAW TIMER--------------------------------
    cv2.rectangle(img, (770, 5), (995, 100), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, f'{int(time.time() - start_time)}/{maxTime}', (800, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
    # ----------------------------------------------------
    # ---------------HANDS THRSHOLDING--------------------
    # ----------------------------------------------------
    # Region of interest to be used for hand thresholding
    roiForHandThresholding = img[top:bottom, right:left]
    thres = getThresholdedHand(img, roiForHandThresholding)
    # ----------------------------------------------------
    # ----------------GESTURE PREDICTION------------------
    # ----------------------------------------------------
    # Feature extraction
    sift = cv2.SIFT_create()
    kp, descriptor = sift.detectAndCompute(thres, None)
    if descriptor is None:
        cv2.rectangle(img, (5, 5), (500, 100), (175, 0, 175), cv2.FILLED)
        cv2.putText(img, f'{operand1} {operationStr} {operand2} = {finalResult}', (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
        cv2.imshow('Hand Tracker', img)
        if cv2.waitKey(1) & 0xff == 27:
            break
        continue
    else:
        # Produce "bag of words" vector
        descriptor = k_means.predict(descriptor)
        vq = [0] * n_clusters
        for feature in descriptor:
            vq[feature] = vq[feature] + 1  # load the model from disk
        # Predict the result
        result = clf.predict([vq])

    # ---------------------DRAW GESTURE PREDICTION-------------------------
    cv2.rectangle(img, (5, 5), (500, 100), (175, 0, 175), cv2.FILLED)
    if(inputsCount == 0):
        # Reset operation
        operationsArr = np.array([])
        # Set operand 1
        operand1 = int(result[0])
        if int(time.time() - start_time) >= maxTime // 2:
            operand1Arr = np.append(operand1Arr, operand1)
        cv2.putText(img, f'{operand1}', (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
    elif(inputsCount == 1):
        # Calculate average operand 1
        if operand1Arr.size > 0:
            operand1Count = np.bincount(operand1Arr.astype(int))
            maxOperand1Count = np.argmax(operand1Count)
            operand1 = maxOperand1Count
        else:
            operand1 = 0
        # Reset operand 2
        operand2Arr = np.array([])
        # Set operation
        operation = int(result[0])
        if int(time.time() - start_time) >= maxTime // 2:
            operationsArr = np.append(operationsArr, operation)
        operationStr = getOperationString(operation)
        cv2.putText(img, f'{operand1} {operationStr}', (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
    elif(inputsCount == 2):
        # Calculate average operation
        if operationsArr.size > 0:
            operationsCount = np.bincount(operationsArr.astype(int))
            maxOperationsCount = np.argmax(operationsCount)
            operation = maxOperationsCount
            operationStr = getOperationString(operation)
        else:
            operation = 6
            operationStr = getOperationString(operation)
        # Reset operand 1
        operand1Arr = np.array([])
        # Set operand 2
        operand2 = int(result[0])
        if int(time.time() - start_time) >= maxTime // 2:
            operand2Arr = np.append(operand2Arr, operand2)
        cv2.putText(img, f'{operand1} {operationStr} {operand2}', (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
    elif(inputsCount == 3):
        # Calculate average operand 2
        if operand2Arr.size > 0:
            operand2Count = np.bincount(operand2Arr.astype(int))
            maxOperand2Count = np.argmax(operand2Count)
            operand2 = maxOperand2Count
        else:
            operand2 = 0
        if operation == 6:
            finalResult = operand1 + operand2
        elif operation == 7:
            finalResult = operand1 - operand2
        elif operation == 8:
            finalResult = operand1 * operand2
        elif operation == 9:
            finalResult = operand1 / operand2
        elif operation == 10:
            finalResult = operand1 ** operand2
        else:  # default
            finalResult = operand1 + operand2

        # print("finalResult = ", finalResult)
        cv2.putText(img, f'{operand1} {operationStr} {operand2} = {finalResult}', (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
    elif(inputsCount == 4):
        cv2.putText(img, f'{operand1} {operationStr} {operand2} = {finalResult}', (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
    # cv2.putText(img, f'{result[0]}', (40, 80),
    #             cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
    # cv2.putText(img, f'finalResult =  {finalResult}', (60, 100), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0, 255), 6)

    # ----------Draw rectangle that contains the output word---------
    # cv2.rectangle(img, (0, 0), (350, 100), (175, 0, 175), cv2.FILLED)
    # cv2.putText(img, finalText, (60, 500),
    #             cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    # --------------------Draw output frame--------------------------
    cv2.imshow('Hand Tracker', img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xff == ord('s'):
        capture = True
    if cv2.waitKey(1) & 0xff == 27:
        break
