# Import necessary libraries.
import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Set webcam dimensions.
wCam, hCam = 640, 480

# Initialize webcam capture with set dimensions.
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize a variable for frame time calculation (for FPS).
pTime = 0

# Initialize the hand detector with specific confidence and max number of hands to detect.
detector = htm.handDetector(detectionCon=1, maxHands=1)

# Setup for accessing the speaker's volume control.
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get the volume range from the speaker.
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Initialize volume, volume bar, volume percentage, and area variables.
vol = 0
volBar = 400
volPer = 0
area = 0
colorVol = (255, 0, 0)

# Start an infinite loop for continuous frame capture and processing.
while True:
    # Read an image frame from the webcam.
    success, img = cap.read()

    # Process the image to find hands and their positions.
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)

    # Check if any landmarks were detected.
    if len(lmList) != 0:

        # Calculate the area of the bounding box to filter hand size.
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100

        # Proceed if the hand area is within a certain range.
        if 250 < area < 1000:

            # Find the distance between the index finger and thumb.
            length, img, lineInfo = detector.findDistance(4, 8, img)

            # Convert the length to a volume level and volume percentage.
            volBar = np.interp(length, [50, 200], [400, 150])
            volPer = np.interp(length, [50, 200], [0, 100])

            # Smooth the volume change.
            smoothness = 10
            volPer = smoothness * round(volPer / smoothness)

            # Check which fingers are up.
            fingers = detector.fingersUp()

            # Change the system volume if the pinky is down.
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                colorVol = (0, 255, 0)
            else:
                colorVol = (255, 0, 0)

    # Draw the volume bar and percentage on the image.
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    # Display the current system volume on the image.
    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, colorVol, 3)

    # Calculate and display the current FPS.
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    # Show the image in a window.
    cv2.imshow("Img", img)
    # Wait for a key press, with a 1 millisecond delay.allowing the window to update continuously.
    cv2.waitKey(1)
