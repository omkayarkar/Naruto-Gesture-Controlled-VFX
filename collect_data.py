import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ==============================
# INPUT GESTURE NAME
# ==============================
gesture_name = input("Enter gesture name: ").strip()

if not gesture_name:
    print("Gesture name cannot be empty.")
    exit()

# ==============================
# CREATE DATA FOLDER
# ==============================
if not os.path.exists("data"):
    os.makedirs("data")

save_path = os.path.join("data", f"{gesture_name}.npy")

# ==============================
# MEDIAPIPE SETUP (VIDEO MODE)
# ==============================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# ==============================
# CAMERA START
# ==============================
cap = cv2.VideoCapture(0)

data = []
count = 0
start_time = time.time()

print("\nPress 's' to save sample")
print("Press 'q' to quit\n")

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not detected.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Proper timestamp in milliseconds
        timestamp = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        landmarks = None

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            h, w, _ = frame.shape
            landmarks = []

            # Draw landmarks
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                landmarks.extend([lm.x, lm.y, lm.z])

            # Draw connections
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (5,9),(9,10),(10,11),(11,12),
                (9,13),(13,14),(14,15),(15,16),
                (13,17),(17,18),(18,19),(19,20),
                (0,17)
            ]

            for start_idx, end_idx in connections:
                x1 = int(hand_landmarks[start_idx].x * w)
                y1 = int(hand_landmarks[start_idx].y * h)
                x2 = int(hand_landmarks[end_idx].x * w)
                y2 = int(hand_landmarks[end_idx].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Display counter
        cv2.putText(frame, f"Samples: {count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Collect Data", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if landmarks is not None:
                data.append(landmarks)
                count += 1
                print(f"Sample saved: {count}")
            else:
                print("No hand detected. Try again.")

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# ==============================
# SAVE ONLY IF DATA EXISTS
# ==============================
if count > 0:
    np.save(save_path, np.array(data))
    print("\nData saved successfully!")
    print("Total samples collected:", count)
else:
    print("\nNo samples collected. Nothing saved.")

"""python collect_data.py"""
"""python train_model.py"""