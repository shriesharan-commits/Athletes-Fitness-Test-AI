import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

# Ask user for crunch target
target_crunches = int(input("Enter the number of crunches you want to do: "))

# Mediapipe initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Video capture
cap = cv2.VideoCapture(0)

# Crunch count variables
crunch_count = 0
crunch_stage = None  # 'up' or 'down'

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    # ✅ Get current date and time (formatted separately with AM/PM)
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y")      # e.g., "September 16, 2025"
    time_str = now.strftime("%I:%M:%S %p")    # e.g., "07:42:10 PM"

    if results.pose_landmarks:
        # ✅ Draw pose landmarks
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
        )

        landmarks = results.pose_landmarks.landmark

        # Get Y positions of nose, shoulder, and hip
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
        shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y

        # Convert to pixel positions
        nose_y_pixel = int(nose_y * h)
        shoulder_y_pixel = int(shoulder_y * h)
        hip_y_pixel = int(hip_y * h)

        # Draw a threshold line for crunch detection (midway between shoulder and hip)
        crunch_threshold_y = int(((shoulder_y + hip_y) / 2) * h)
        cv2.line(frame, (0, crunch_threshold_y), (w, crunch_threshold_y), (255, 255, 0), 2)
        cv2.putText(frame, "Crunch Line", (10, crunch_threshold_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Crunch detection logic
        if nose_y < crunch_threshold_y / h:  # Nose goes above threshold (laying back)
            if crunch_stage == 'down':
                crunch_count += 1
                crunch_stage = 'up'
        elif nose_y > shoulder_y:  # Nose goes below/near shoulder (crunching up)
            crunch_stage = 'down'

        # Display crunch count
        cv2.putText(frame, f'Crunches: {crunch_count}/{target_crunches}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        # If target reached, show "Workout Complete"
        if crunch_count >= target_crunches:
            cv2.putText(frame, 'Workout Complete!', (w // 4, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.imshow('Crunch Tracker', frame)
            cv2.waitKey(3000)
            break

    # ✅ Show date and time in top-right corner
    date_size, _ = cv2.getTextSize(date_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    time_size, _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = w - max(date_size[0], time_size[0]) - 10
    cv2.putText(frame, date_str, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, time_str, (text_x, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Crunch Tracker', frame)

    # Press ESC to exit early
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
