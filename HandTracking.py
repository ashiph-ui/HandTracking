import cv2
import mediapipe as mp
import time

# First setting up the which camera to use - this is the video object:
cap = cv2.VideoCapture(0) # 0 indicates which camera to use

mpHands = mp.solutions.hands # Hands object - Formality
hands = mpHands.Hands() # Hands object - frames
mpDraw = mp.solutions.drawing_utils # Drawing utilities - this will 

while True:
    success, img = cap.read() # Frame
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image to RGB
    results = hands.process(imgRGB) # Process the image
    print(results.multi_hand_landmarks) # Print the results - This will specifcally print the landmarks of the hand
    
    if results.multi_hand_landmarks:
        for handsLM in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handsLM, mpHands.HAND_CONNECTIONS)

    
    cv2.imshow("Image", img)
    cv2.waitKey(1)

# The steps above are typical to run a webcam
       