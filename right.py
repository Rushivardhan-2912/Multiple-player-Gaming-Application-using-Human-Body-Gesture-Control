import cv2
import mediapipe as mp
import numpy as np
import pyautogui


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

def check_right_player_posture(img):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame and get the landmarks
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_angle = angle_between_points(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

        right_angle = angle_between_points(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

        # Check for horizontalness
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        threshold = 0.05  # Adjust this threshold as needed

##
        # if 165 <= left_angle <= 195:
        #     cv2.putText(img, "Left Arm ~180", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 200, 0), 2, cv2.LINE_AA)
        # if 165 <= right_angle <= 195:
        #     cv2.putText(img, "Right Arm ~180", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 200, 0), 2, cv2.LINE_AA)

        # Check for horizontalness of left arm
        left_elbow_y = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        # if abs(left_wrist_y - left_elbow_y) < threshold:
        #     cv2.putText(img, "Left Hand Horizontal", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 50), 2,
        #                 cv2.LINE_AA)

        # Check for horizontalness of right arm
        right_elbow_y = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
        # if abs(right_wrist_y - right_elbow_y) < threshold:
        #     cv2.putText(img, "Right Hand Horizontal", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 50), 2,
        #                 cv2.LINE_AA)


        if 165 <= left_angle <= 195 and abs(left_wrist_y - left_elbow_y) < threshold:
            pyautogui.press('num1')
        if 165 <= right_angle <= 195 and abs(right_wrist_y - right_elbow_y) < threshold:
            pyautogui.press('num2')

        # Visualize the detected keypoints
        # Define custom connections for the arms
        # Define custom connections for the arms
        ARM_CONNECTIONS = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
        ]

        # Visualize only the arm landmarks
        mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, ARM_CONNECTIONS)

    else:
        cv2.putText(img, "Player2 not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 0), 2, cv2.LINE_AA)

    return img
