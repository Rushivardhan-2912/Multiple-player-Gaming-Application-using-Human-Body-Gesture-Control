import cv2
import left
import right
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Split the frame into two
    height, width, _ = frame.shape
    img1 = frame[:, :width // 2  ]  # Left half
    img2 = frame[:, width // 2 :]  # Right half

    # Check the posture for both left and right players
    img1 = left.check_left_player_posture(img1)
    img2 = right.check_right_player_posture(img2)

    # Combining the frame again for displaying
    frame = np.hstack((img1, img2))

    cv2.imshow('Game Frame', frame)
    cv2.imshow('left', img1)
    cv2.imshow('right', img2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()