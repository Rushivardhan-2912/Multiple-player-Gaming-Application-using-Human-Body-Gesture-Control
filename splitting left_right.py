import cv2
import mediapipe as mp
import numpy as np

# Initializing MediaPipe's Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate the angle between three points
def angle_between_points(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    dot_product = np.dot(v1, v2)
    cross_product = np.linalg.norm(np.cross(v1, v2))
    angle = np.arctan2(cross_product, dot_product)
    return abs(np.degrees(angle))

def check_left_player_posture(img):
    return check_posture(img, 'Player1')

def check_right_player_posture(img):
    return check_posture(img, 'Player2')

def check_posture(img, player_name):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_angle = angle_between_points(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_EAR.value])
        right_angle = angle_between_points(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value])

        # Check if player is standing straight
        if 85 <= left_angle <= 95 and 85 <= right_angle <= 95:
            cv2.putText(img, f"{player_name} is Upright", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, f"{player_name}: Stand Upright", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # Visualize the detected keypoints
        mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        cv2.putText(img, f"{player_name} not found!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return img

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Split the frame into two
    height, width, _ = frame.shape
    img1 = frame[:, :width // 2]  # Left half
    img2 = frame[:, width // 2:]  # Right half

    # Check the posture for img1 (left side player) and img2 (right side player)
    img1 = check_left_player_posture(img1)
    img2 = check_right_player_posture(img2)

    # Combining the frame again for displaying (if needed)
    frame = np.hstack((img1, img2))

    cv2.imshow('Game Frame', frame)
    cv2.imshow('Left Half', img1)
    cv2.imshow('Right Half', img2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
