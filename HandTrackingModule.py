import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Constructor for the handDetector class with default parameters.
        # 'mode' is a boolean indicating whether to treat the input as a static image (False) or stream (True).
        self.mode = mode

        # 'maxHands' specifies the maximum number of hands to detect.
        self.maxHands = maxHands

        # 'detectionCon' is the minimum detection confidence threshold.
        self.detectionCon = detectionCon

        # 'trackCon' is the tracking confidence. It helps in determining when to stop tracking a hand.
        self.trackCon = trackCon

        # 'mpHands' is an instance of the MediaPipe Hands solution.
        self.mpHands = mp.solutions.hands

        # 'hands' is the actual hand detector object. It is configured with the parameters passed to the constructor.
        # This object will be used for hand detection in images.
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)

        # 'mpDraw' is a utility from MediaPipe for drawing; used for drawing hand landmarks on the image.
        self.mpDraw = mp.solutions.drawing_utils

        # 'tipIds' is a list containing ids of the fingertip landmarks (thumb to pinky).
        # These ids correspond to specific landmarks on each finger.
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        # Convert the image from BGR (Blue, Green, Red - OpenCV's default color format) to RGB color format.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the RGB image using the MediaPipe Hands model. This will detect hands and their landmarks in the image.
        # The results are stored in 'self.results'.
        self.results = self.hands.process(imgRGB)
        # The above line could potentially print the landmarks found, but it's commented out.

        # Check if any hands are detected in the image.
        if self.results.multi_hand_landmarks:
            # Iterate through each detected hand.
            for handLms in self.results.multi_hand_landmarks:
                # If 'draw' is True, draw landmarks and connections on the hand.
                # This visualizes the hand landmarks (like joints of the fingers) on the image.
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        # Return the image with drawn landmarks (if 'draw' is True) or the original image (if 'draw' is False).
        return img

    def findPosition(self, img, handNo=0, draw=True):
        # Initialize lists for storing x and y coordinates and a list for the bounding box (bbox).
        xList = []
        yList = []
        bbox = []
        # Initialize a list to store the landmark information.
        self.lmList = []
        # Initialize 'myHand' to None. It will later be used to store landmarks of the specific hand we are interested in.
        myHand = None  # Initialize myHand to None

        # Check if any hand landmarks were detected in the previous process.
        if self.results.multi_hand_landmarks:
            # Select the landmarks of the specified hand number (default is the first hand, handNo=0).
            myHand = self.results.multi_hand_landmarks[handNo]
            # Iterate over each landmark in the hand.
            for id, lm in enumerate(myHand.landmark):
                # Extract the width, height, and channels of the image for scaling the landmark coordinates.
                h, w, c = img.shape
                # Scale the landmark coordinates to match the size of the image.
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Append the scaled coordinates to their respective lists.
                xList.append(cx)
                yList.append(cy)
                # Append the landmark ID and its coordinates to the landmarks list.
                self.lmList.append([id, cx, cy])
                # If drawing is enabled, draw a circle at each landmark position.
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            # Calculate the bounding box around the hand by finding the min and max x, y coordinates.
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            # If drawing is enabled, draw a rectangle around the hand using the bounding box coordinates.
            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        # Return the list of landmarks (with their IDs and coordinates) and the bounding box for the specified hand.
        return self.lmList, bbox

    def fingersUp(self):
        # Initialize a list to store the status of each finger (up or down).
        fingers = []

        # Check for the thumb:
        # Compares the x-coordinate of the thumb tip with the x-coordinate of its lower joint.
        # This is specific to the thumb due to its unique orientation and movement.
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            # If the tip of the thumb is to the right of the lower joint (for the right hand), the thumb is considered up.
            fingers.append(1)
        else:
            # Otherwise, the thumb is considered down.
            fingers.append(0)

        # Check for the other four fingers (index, middle, ring, pinky):
        for id in range(1, 5):
            # Compares the y-coordinate of each fingertip with the y-coordinate of its corresponding middle joint.
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                # If the tip is above the middle joint, the finger is considered up.
                fingers.append(1)
            else:
                # Otherwise, the finger is considered down.
                fingers.append(0)

        # Return the list containing the status of each finger.
        # '1' indicates a finger is up, and '0' indicates it is down.
        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        # Retrieve the coordinates (x1, y1) of the first point (p1) from the landmarks list (lmList).
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]

        # Retrieve the coordinates (x2, y2) of the second point (p2) from the landmarks list.
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]

        # Calculate the center point (cx, cy) between the two points.
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # If 'draw' is set to True, then visualize the points and the line connecting them on the image.
        if draw:
            # Draw a circle at the first point (p1).
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            # Draw a circle at the second point (p2).
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            # Draw a line connecting the two points.
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # Draw a circle at the center point between p1 and p2.
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # Calculate the Euclidean distance between the two points.
        length = math.hypot(x2 - x1, y2 - y1)

        # Return the calculated distance, the image with drawn elements (if draw=True), and the list of coordinates.
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    # Initialize a variable to store the previous time, used later for FPS calculation.
    pTime = 0

    # Start capturing video from the default camera (usually the webcam).
    cap = cv2.VideoCapture(0)

    # Create an instance of the handDetector class.
    detector = handDetector()

    # Start an infinite loop to continuously process video frames.
    while True:
        # Read a frame from the video capture.
        success, img = cap.read()

        # Use the handDetector to find hands in the captured frame.
        img = detector.findHands(img)

        # Find the position of hand landmarks in the frame.
        lmList = detector.findPosition(img)

        # Check if any landmarks are found in the frame.
        if len(lmList) != 0:
            # Print the coordinates of the 5th landmark (index 4) if landmarks are detected.
            print(lmList[4])

        # Calculate the current time for FPS calculation.
        cTime = time.time()
        # Calculate the Frames Per Second (FPS).
        fps = 1 / (cTime - pTime)
        # Update the previous time to the current time.
        pTime = cTime

        # Put the FPS value on the image as text.
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        # Display the image in a window named 'Image'.
        cv2.imshow("Image", img)
        # Wait for 1 millisecond. If a key is pressed, break the loop (not explicitly written here, but it's a common practice).
        cv2.waitKey(1)


# Check if the script is run directly (not imported) and then call the main function.
if __name__ == "__main__":
    main()
