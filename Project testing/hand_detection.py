import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, skeletonize, thin
import cv2
import mediapipe as mp
import cvzone
from time import sleep
from pynput.keyboard import Controller
from cvzone.HandTrackingModule import HandDetector
import math


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


cap = cv2.VideoCapture(0)

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


# take first frame of the video
ret, frame = cap.read()
# setup initial location of window
x, y, w, h = 0, 0, 300, 400  # simply hardcoded the values
track_window = (x, y, w, h)
# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                   np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1000, 600))
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)  # TBD

    lmList = []
    if success == True:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x, y, w, h = track_window
        img = cv2.rectangle(img, (x, y), (x+w, y+h), 255, 2)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

    img = drawAll(img, buttonList)
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0].landmark
        x1 = hand[8].x*1000  # tarf awel soba3
        x2 = hand[12].x*1000  # tarf tany soba3
        y1 = hand[8].y*600
        y2 = hand[12].y*600
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        # print(distance)
        if distance < 60:
            cv2.putText(img, 'Clicked', (40, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 0, 255), 6)
            finalText += checkIfClickedOnKeyboard((x1+x2)//2, (y1+y2)//2)
            sleep(0.15)
        else:
            cv2.putText(img, 'Not clicked', (40, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 0, 255), 6)

    # print the text on screen
    cv2.rectangle(img, (50, 450), (1000, 550), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 500),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow('Hand Tracker', img)
    cv2.waitKey(1)
    if cv2.waitKey(5) & 0xff == 27:
        break
