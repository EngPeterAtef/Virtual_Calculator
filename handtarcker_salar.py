## Salar Brefki

import cv2
import mediapipe as mp
import cvzone
from time import sleep
from pynput.keyboard import Controller
from cvzone.HandTrackingModule import HandDetector
import math
## Facbook - https://www.facebook.com/salar.brefki/
## Instagram - https://www.instagram.com/salarbrefki/
keyboard = Controller()

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
      cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
      cv2.putText(img, button.text, (x + 10, y + 45),
                  cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
   return img

def checkIfClickedOnKeyboard(x_finger,y_finger):
   for button in buttonList:
      x, y = button.pos
      thres = 60
      if(abs(x - x_finger) < thres and abs(y - y_finger) < thres-30):
         return button.text
   return ''

while True:

   success, img = cap.read()
   img = cv2.resize(img, (1000, 600))
   img = cv2.flip(img, 1)

   imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   results = hands.process(imgRGB) # TBD
   img = drawAll(img, buttonList)

   lmList = []

   if results.multi_hand_landmarks:
      hand = results.multi_hand_landmarks[0].landmark
      x1 = hand[8].x*1000 # tarf awel soba3
      x2 = hand[12].x*1000 # tarf tany soba3
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
   cv2.putText(img, finalText, (60, 500),cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)   

   cv2.imshow('Hand Tracker', img)
   cv2.waitKey(1)
   if cv2.waitKey(5) & 0xff == 27:
      break