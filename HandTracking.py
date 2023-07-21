import cv2
import mediapipe as mp
import time

# First setting up the which camera to use - this is the video object:
cap = cv2.VideoCapture(0) # 0 indicates which camera to use

while True:
    success, img = cap.read() # Frame
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)

# The steps above are typical to run a webcam
       