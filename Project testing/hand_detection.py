import time
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, skeletonize, thin
import cv2
import mediapipe as mp
import cvzone
from pynput.keyboard import Controller
from cvzone.HandTrackingModule import HandDetector
import math
import pandas as pd
import pickle
import os
import threading
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ----------DONT DELETE ANY COMMENT FROM THIS FILE!!!!!!!!!!-----------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Structure element path
# path1 = "D:/Engineering/CUFE/3rd Year (Computer) (2022)/First Semester/Image Processing/Projects/Virtual_Keyboard/Project testing"
# structure_element = pd.read_excel(path1 + '/stel20x20.xlsx')
# structure_element = structure_element.to_numpy()
# print(structure_element.max())
operand1 = operation = operand2 = timer = None
inputsCount = 0
finalResult = 0


def fun():
    print("callback func")
    global operand1, operation, operand2, timer, inputsCount, result, finalResult, start_time
    inputsCount = inputsCount + 1
    print(
        f"operand1= {operand1}, operation = {operation}, operand2 = {operand2}")
    if(inputsCount < 4):
        timer = threading.Timer(10, fun)
        start_time = time.time()
        timer.start()
        # print ("Running for : %s seconds"%(time.time()-start_time))
        print("after calling start_time.start()")
    if(inputsCount == 4):
        inputsCount = 0
        timer = threading.Timer(10, fun)
        start_time = time.time()
        timer.start()
        # print ("Running for : %s seconds"%(time.time()-start_time))
        print("after calling start_time.start()")


def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


# Position of ROI of hand thresholding
top, right, bottom, left = 350, 490, 565, 730
# Change the resolution of video
# cap = cv2.VideoCapture(0,  apiPreference=cv2.CAP_ANY, params=[
#     cv2.CAP_PROP_FRAME_WIDTH, 1024,
#     cv2.CAP_PROP_FRAME_HEIGHT, 768])
cap = cv2.VideoCapture(0)
# --------------------Capture dataset---------------------
index = 751
capture = False
path = "D:/CMP/third_Year/first_Semester/imageProcessing and computerVision/Project/data set/2"


def getThresholdedHand(frame, roi):
    global top, right, bottom, left, index, capture
    # Draw rectangle to indicate the area in which we initialize hand positon for the first time
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    # Convert to gray scale
    # roi2 = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # if capture:
    #     cv2.imwrite(os.path.join(path, f'{index}.jpg'), roi2)
    #     print(index)
    #     index = index + 1
    # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Gaussiam filter
    # roi = cv2.GaussianBlur(roi, (17, 17), 0)
    # Threshold
    # et, thresh1 = cv2.threshold(
    #     roi, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    

    # define the upper and lower boundaries of the HSV pixel intensities
    # to be considered 'skin'
    hsvim = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 180, 230], dtype="uint8")
    skinMask = cv2.inRange(hsvim, lower, upper)

    # blur the mask to help remove noise
    skinMask = cv2.GaussianBlur(skinMask, (17, 17), 0)
    # skinMask = cv2.blur(skinMask, (2, 2))

    # get threshold image
    # ret, thresh1 = cv2.threshold(
        # skinMask, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh1 = cv2.adaptiveThreshold(
        skinMask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 355, 5)
    # cv2.imshow("thresh", thresh1)
    # Show hand
    cv2.imshow('Hand threshold', thresh1)
    # if capture:
    #     cv2.imwrite(os.path.join(path, f'{index}.jpg'), thresh1)
    #     print(index)
    #     index = index + 1
    return thresh1


mpHands = mp.solutions.hands
hands = mpHands.Hands()

tipIds = [4, 8, 12, 16, 20]
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
finalText = ""


class Button():
    def __init__(self, pos, text, size=[60, 60]):
        self.pos = pos
        self.size = size
        self.text = text


buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 15, 100 * i + 100], key))


def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
                          20, rt=0)
        cv2.rectangle(img, button.pos, (x + w, y + h),
                      (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 10, y + 45),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img


def checkIfClickedOnKeyboard(x_finger, y_finger):
    for button in buttonList:
        x, y = button.pos
        thres = 60
        if(abs(x - x_finger) < thres and abs(y - y_finger) < thres-30):
            return button.text
    return ''


# -----------------------------LOAD MODELS---------------------------
# Load kmeans model
filename1 = 'kmeans_model.sav'
k_means = pickle.load(open(filename1, 'rb'))
n_clusters = 1600
# Load SVM model
filename2 = 'gestures_model.sav'
clf = pickle.load(open(filename2, 'rb'))
# ----------------------MEAN SHIFT INITIALIZATION--------------------
# take first frame of the video
ret, frame = cap.read()
# setup initial location of window
x, y, w, h = 0, 0, 300, 400  # simply hardcoded the values
track_window = (x, y, w, h)
# set up the ROI for tracking in MeanShift
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                   np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


timer = threading.Timer(10, fun)
start_time = time.time()
timer.start()
# ----------------------------------------
# ----------------MAIN LOOP---------------
# ----------------------------------------
print("Running for : %s seconds" % (time.time()-start_time))
while True:

    # READ FRAME
    success, img = cap.read()
    img = cv2.resize(img, (1000, 600))
    img = cv2.flip(img, 1)
    # TRANSFORM TO RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # GET HANDS USING MEDIAPEP
    # results = hands.process(imgRGB)
    # ----------------------------------------
    # ---------------MEAN SHIFT---------------
    # ----------------------------------------
    # if success == True:
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    #     # -----------apply meanshift to get the new location-----------
    #     ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    #     # -----------apply CamShift to get the new location-----------
    #     # ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    #     # Draw it on image
    #     # pts = cv2.boxPoints(ret)
    #     # pts = np.int0(pts)
    #     # img = cv2.polylines(img, [pts], True, 255, 2)
    #     x, y, w, h = track_window
    #     img = cv2.rectangle(img, (x, y), (x+w, y+h), 255, 2)
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         break
    # else:
    #     break
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
        continue
    else:
        # Produce "bag of words" vector
        descriptor = k_means.predict(descriptor)
        vq = [0] * n_clusters
        for feature in descriptor:
            vq[feature] = vq[feature] + 1  # load the model from disk
        # Predict the result
        result = clf.predict([vq])
    # -----------------Draw Keyboard----------------------
    # img = drawAll(img, buttonList)
    # ------------------------------------------------------------------------
    # -------Calculate distance between fingers to check if clicked-----------
    # ------------------------------------------------------------------------
    # if results.multi_hand_landmarks:
    #     hand = results.multi_hand_landmarks[0].landmark
    #     x1 = hand[8].x * 1000  # tarf awel soba3
    #     x2 = hand[12].x * 1000  # tarf tany soba3
    #     y1 = hand[8].y * 600
    #     y2 = hand[12].y * 600
    #     distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    #     # print(distance)
    #     if distance < 60:
    #         cv2.putText(img, 'Clicked', (40, 80), cv2.FONT_HERSHEY_SIMPLEX,
    #                     3, (0, 0, 255), 6)
    #         finalText += checkIfClickedOnKeyboard((x1+x2)//2, (y1+y2)//2)
    #         sleep(0.15)
    #     else:
    #         cv2.putText(img, 'Not clicked', (40, 80), cv2.FONT_HERSHEY_SIMPLEX,
    #                     3, (0, 0, 255), 6)

    # ---------------------DRAW GESTURE PREDICTION-------------------------
    # 1 none none = none
    cv2.putText(img, f'{int(time.time() - start_time)}/10', (700, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
    if(inputsCount == 0):
        operand1 = int(result[0])
        cv2.putText(img, f'{operand1}', (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
    elif(inputsCount == 1):
        operation = int(result[0])
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
            operationStr = "+"
        cv2.putText(img, f'{operand1} {operationStr}', (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
    elif(inputsCount == 2):
        operand2 = int(result[0])
        cv2.putText(img, f'{operand1} {operationStr} {operand2}', (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
    elif(inputsCount == 3):
        # print("count = 3")
        if(operation == 6):
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

        print("finalResult = ", finalResult)
        cv2.putText(img, f'{operand1} {operationStr} {operand2} = {finalResult}', (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
    elif(inputsCount == 4):
        cv2.putText(img, f'{operand1} {operationStr} {operand2} = {finalResult}', (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
    # cv2.putText(img, f'{result[0]}', (40, 80),
    #             cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
    # cv2.putText(img, f'finalResult =  {finalResult}', (60, 100), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0, 255), 6)

    # ----------Draw rectangle that contains the output word---------
    # cv2.rectangle(img, (50, 450), (600, 550), (175, 0, 175), cv2.FILLED)
    # cv2.putText(img, finalText, (60, 500),
    #             cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    # --------------------Draw output frame--------------------------
    cv2.imshow('Hand Tracker', img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xff == ord('s'):
        capture = True
        # break
    if cv2.waitKey(1) & 0xff == 27:
        break
