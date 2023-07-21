import cv2
import mediapipe as mp
import time

# First setting up the which camera to use - this is the video object:
cap = cv2.VideoCapture(0) # 0 indicates which camera to use

mpHands = mp.solutions.hands # Hands object - Formality
hands = mpHands.Hands() # Hands object - frames
mpDraw = mp.solutions.drawing_utils # Drawing utilities - this will 

pTime = 0 # Previous time
cTime = 0 # Current time

while True:
    success, img = cap.read() # Frame
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image to RGB
    results = hands.process(imgRGB) # Process the image
    print(results.multi_hand_landmarks) # Print the results - This will specifcally print the landmarks of the hand
    
    if results.multi_hand_landmarks:
        for handsLM in results.multi_hand_landmarks:
            for id, lm in enumerate(handsLM.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                # Only drawing a circle for id number 5
                if id == 4: # This is the thumb
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handsLM, mpHands.HAND_CONNECTIONS)

    cTime = time.time() # Current time
    fps = 1/(cTime-pTime) # Frames per second
    pTime = cTime # Previous time become Current time

    cv2.putText(img, "fps: "+ str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

       