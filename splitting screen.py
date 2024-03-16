import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

while True:
    success, img = cap.read()
    height, width, _ = img.shape

    # Splitting the frame
    img1 = img[:, :width//2]  # Left half
    img2 = img[:, width//2:]  # Right half

    cv2.imshow('Left Half', img1)
    cv2.imshow('Right Half', img2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
