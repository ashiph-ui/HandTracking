import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5):
        # Initialising the Parameters/variables
        self.mode = mode # Static image mode
        self.maxHands = maxHands # Maximum number of hands
        self.model_complexity = model_complexity # Complexity of the hand landmark model: 0 or 1
        self.detectionCon = detectionCon # Minimum confidence value for hand detection
        self.trackCon = trackCon # Minimum confidence value for hand tracking

        # Initialising the hands object within the class
        self.mpHands = mp.solutions.hands # Hands object - Formality
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity,self.detectionCon, self.trackCon) # Hands object - frames
        self.mpDraw = mp.solutions.drawing_utils 

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image to RGB
        self.results = self.hands.process(imgRGB) # Process the image
        print(self.results.multi_hand_landmarks) # Print the results - This will specifcally print the landmarks of the hand
        
        if self.results.multi_hand_landmarks:
            for handsLM in self.results.multi_hand_landmarks: 
                if draw:
                    self.mpDraw.draw_landmarks(img, handsLM, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = [] # Landmark list
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                            # Only drawing a circle for id number 5
                if draw: # This is the thumb
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)
        return lmList


# This main function shows how to set up the camera and use the class
def main():
    pTime = 0 # Previous time
    cTime = 0 # Current time
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read() # Frame
        img = detector.findHands(img)
        lmlist = detector.findPosition(img, draw=False)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time() # Current time
        fps = 1/(cTime-pTime) # Frames per second
        pTime = cTime

        cv2.putText(img, "fps: " + str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
         (255, 0, 255))
         
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()