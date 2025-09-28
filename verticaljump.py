import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import pandas as pd
from playsound import playsound
import os

# === CONFIGURATION ===

CSV_FILENAME = 'vertical_jump_log.csv'
SOUND_FILE = 'complete_sound.mp3'  # sound to play on completion
VELOCITY_THRESHOLD = 20  # change in hip pixel position between frames to detect movement
GROUND_FRAMES = 30  # number of initial frames to average as ground ankle position

# === USER INPUT ===

target_jumps = int(input("Enter the number of jumps you want to do: "))
user_height_cm = float(input("Enter your height in cm (for better jump height estimate): "))

# === INIT Mediapipe ===

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

# === VARIABLES for tracking ===

jump_count = 0
jump_stage = None  # can be 'up' or 'down' or None
previous_hip_y = None

ground_ankle_ys = []
ground_ankle_y = None

max_jump_height_cm = 0.0
last_jump_height_cm = 0.0
attempts = 0

# === CSV logging initialization ===

if not os.path.isfile(CSV_FILENAME):
    df_init = pd.DataFrame(columns=["Timestamp", "Total_Jumps", "Max_Jump_Height_cm", "User_Height_cm", "Attempts"])
    df_init.to_csv(CSV_FILENAME, index=False)

# === MAIN LOOP ===

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)
    current_time = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
        )

        landmarks = results.pose_landmarks.landmark

        # === Get ankle (foot) position ===
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        ankle_y = left_ankle.y
        ankle_y_pixel = int(ankle_y * h)

        # Gather ground ankle positions initially
        if ground_ankle_y is None:
            ground_ankle_ys.append(ankle_y_pixel)
            if len(ground_ankle_ys) >= GROUND_FRAMES:
                # compute average ground level
                ground_ankle_y = int(sum(ground_ankle_ys) / len(ground_ankle_ys))
                print(f"Ground ankle level established at pixel y = {ground_ankle_y}")

        # Draw ground line if known
        if ground_ankle_y is not None:
            cv2.line(frame, (0, ground_ankle_y), (w, ground_ankle_y), (255, 255, 0), 2)
            cv2.putText(frame, "Ground Level", (10, ground_ankle_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # === Use hip to detect movement up/down ===
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        hip_y = hip.y
        hip_y_pixel = int(hip_y * h)

        if previous_hip_y is not None and ground_ankle_y is not None:
            delta = previous_hip_y - hip_y_pixel  # positive if hip moves up

            # Going up
            if delta > VELOCITY_THRESHOLD:
                if jump_stage == 'down' or jump_stage is None:
                    jump_stage = 'up'

            # Coming down
            elif delta < -VELOCITY_THRESHOLD:
                if jump_stage == 'up':
                    # Completed one jump
                    jump_count += 1
                    attempts += 1
                    jump_stage = 'down'

                    # Estimate jump height in cm
                    pixel_jump = ground_ankle_y - ankle_y_pixel  # how many pixels ankle rose
                    if pixel_jump < 0:
                        pixel_jump = 0  # no negative heights

                    # ratio of pixel movement to frame height
                    jump_height_ratio = pixel_jump / h
                    jump_height_cm = jump_height_ratio * user_height_cm

                    last_jump_height_cm = round(jump_height_cm, 2)
                    if last_jump_height_cm > max_jump_height_cm:
                        max_jump_height_cm = last_jump_height_cm

        previous_hip_y = hip_y_pixel

        # === Overlay information ===
        cv2.putText(frame, f'Jumps: {jump_count}/{target_jumps}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.putText(frame, f'Last Jump: {last_jump_height_cm:.2f} cm', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f'Max Jump: {max_jump_height_cm:.2f} cm', (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f'Attempts: {attempts}', (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 100), 2, cv2.LINE_AA)

        # === Completion logic ===
        if jump_count >= target_jumps:
            cv2.putText(frame, 'Workout Complete!', (w // 4, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

            # Play sound (non-blocking or blocking)
            try:
                playsound(SOUND_FILE)
            except Exception as e:
                print("Error playing sound:", e)

            # Log to CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df_new = pd.DataFrame([{
                "Timestamp": timestamp,
                "Total_Jumps": jump_count,
                "Max_Jump_Height_cm": max_jump_height_cm,
                "User_Height_cm": user_height_cm,
                "Attempts": attempts
            }])
            df_new.to_csv(CSV_FILENAME, mode='a', header=False, index=False)

            cv2.imshow('Vertical Jump Tracker', frame)
            cv2.waitKey(3000)
            break

    # Display current time
    text_size, _ = cv2.getTextSize(current_time, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = w - text_size[0] - 10
    cv2.putText(frame, current_time, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 2, cv2.LINE_AA)

    cv2.imshow('Vertical Jump Tracker', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit early
        break

cap.release()
cv2.destroyAllWindows()
