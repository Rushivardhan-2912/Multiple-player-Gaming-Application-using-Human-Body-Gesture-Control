import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe's Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Haar cascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Function to calculate the angle between three points
def angle_between_points(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    dot_product = np.dot(v1, v2)
    cross_product = np.linalg.norm(np.cross(v1, v2))
    angle = np.arctan2(cross_product, dot_product)
    return abs(np.degrees(angle))


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi = frame[y:y + h, x:x + w]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_roi)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_angle = angle_between_points(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_EAR.value])

            right_angle = angle_between_points(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value])

            # Check if person is standing straight
            if 85 <= left_angle <= 95 and 85 <= right_angle <= 95:
                cv2.putText(roi, "Upright", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(roi, "Adjust", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            mp.solutions.drawing_utils.draw_landmarks(roi, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Game Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
