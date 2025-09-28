# Situps Counter with Mediapipe and OpenCV
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

# Ask user for sit-up target
target_situps = int(input("Enter the number of sit-ups you want to do: "))

# Mediapipe initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Video capture
cap = cv2.VideoCapture(0)

# Sit-up count variables
situp_count = 0
situp_stage = None  # 'up' or 'down'

# Function to calculate angle between 3 points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    # Get current date and time in 12-hour format
    current_time = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

    if results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
        )

        landmarks = results.pose_landmarks.landmark

        # Get required landmarks
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * h]

        # Calculate hip angle
        hip_angle = calculate_angle(shoulder, hip, knee)

        # Sit-up detection logic using hip angle
        if hip_angle > 150:
            situp_stage = 'down'
        if hip_angle < 90 and situp_stage == 'down':
            situp_stage = 'up'
            situp_count += 1

        # Display sit-up count
        cv2.putText(frame, f'Sit-ups: {situp_count}/{target_situps}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        # Stop when target sit-ups are reached
        if situp_count >= target_situps:
            cv2.putText(frame, 'Workout Complete!', (w//4, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.imshow('AthletesX', frame)
            cv2.waitKey(3000)
            break

    # Show current time and date in top-right corner
    text_size, _ = cv2.getTextSize(current_time, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = w - text_size[0] - 10
    text_y = 30
    cv2.putText(frame, current_time, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 2, cv2.LINE_AA)

    # Show result
    cv2.imshow('AthletesX', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # Esc key to stop early
        break

cap.release()
cv2.destroyAllWindows()
