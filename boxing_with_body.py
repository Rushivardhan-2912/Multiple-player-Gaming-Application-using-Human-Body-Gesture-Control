import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Extract coordinates from landmark
def extract_coordinates(landmark):
    return np.array([landmark.x, landmark.y])

# Calculate the angle between three points
def angle_between_points(p1, p2, p3):
    p1 = extract_coordinates(p1)
    p2 = extract_coordinates(p2)
    p3 = extract_coordinates(p3)

    v1 = p1 - p2
    v2 = p3 - p2

    dot_product = np.dot(v1, v2)
    cross_product = np.cross(v1, v2)
    angle = np.arctan2(cross_product, dot_product)

    return np.degrees(angle)

def identify_players(heads):
    if len(heads) > 1:
        left_player = min(heads, key=lambda x: x.x)
        right_player = max(heads, key=lambda x: x.x)
        return left_player, right_player
    return None, None

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(image)

    heads_detected = []

    if results.pose_landmarks:
        nose_index = mp_holistic.PoseLandmark.NOSE.value
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if landmark.visibility > 0.5 and i == nose_index:
                heads_detected.append(landmark)

        if len(heads_detected) < 2:
            print(len(heads_detected))
            print('-'*70)
            cv2.putText(frame,str(len(heads_detected)) + 'Two players required to play this game ', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            # If more than two heads detected, we consider two major heads
            if len(heads_detected) > 2:
                cv2.putText(frame, 'Major two players considered', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            player1, player2 = identify_players(heads_detected)
            cv2.putText(frame, 'Player 1', (int(player1.x*frame.shape[1]), int(player1.y*frame.shape[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Player 2', (int(player2.x*frame.shape[1]), int(player2.y*frame.shape[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            landmarks = results.pose_landmarks.landmark

            # Check arm angles and press keys
            player1_rh = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value],
                          landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value],
                          landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value]]

            player1_lh = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value],
                          landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value],
                          landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value]]

            if 175 <= angle_between_points(player1_rh[0], player1_rh[1], player1_rh[2]) <= 185:
                pyautogui.press('U')

            if 175 <= angle_between_points(player1_lh[0], player1_lh[1], player1_lh[2]) <= 185:
                pyautogui.press('Y')

            player2_rh = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value],
                          landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value],
                          landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value]]

            player2_lh = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value],
                          landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value],
                          landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value]]

            if 175 <= angle_between_points(player2_rh[0], player2_rh[1], player2_rh[2]) <= 185:
                pyautogui.press('num1')

            if 175 <= angle_between_points(player2_lh[0], player2_lh[1], player2_lh[2]) <= 185:
                pyautogui.press('num2')

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
