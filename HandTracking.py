import cv2
import mediapipe as mp
import time

# First setting up the which camera to use - this is the video object:
cap = cv2.VideoCapture(0) # 0 indicates which camera to use

mpHands = mp.solutions.hands # Hands object - Formality
hands = mpHands.Hands() # Hands object - frames


while True:
    success, img = cap.read() # Frame
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    results = hands.process(imgRGB) # Process the image

    cv2.imshow("Image", img)
    cv2.waitKey(1)

# The steps above are typical to run a webcam
       