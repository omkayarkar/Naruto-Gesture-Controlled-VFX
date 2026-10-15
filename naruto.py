import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import math
import random
import pygame

# ==========================
# Load trained model
# ==========================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ==========================
# Load Rasengan sprite
# ==========================
rasengan_sprite = cv2.imread("rasengan.png", cv2.IMREAD_UNCHANGED)
if rasengan_sprite is None:
    print("rasengan.png not found!")
    exit()

# ==========================
# Initialize sound
# ==========================
pygame.mixer.init()
try:
    pygame.mixer.music.load("rasengan.mp3")
except:
    print("rasengan.mp3 not found!")
    exit()

# ==========================
# MediaPipe setup
# ==========================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

def normalize(label):
    return label.lower().replace(" ", "_")

SEQUENCE = [
    normalize("chakra_focus"),
    normalize("chakra_rotate"),
    normalize("rasengan_ready")
]

sequence_index = 0
sequence_start_time = None
sequence_timeout = 6
CONFIDENCE_THRESHOLD = 0.60

animation_active = False
animation_start = 0
animation_duration = 6.0

# ==========================
# Circular masked overlay
# ==========================
def overlay_circle(background, overlay, cx, cy, size, angle):

    overlay_resized = cv2.resize(overlay, (size, size), interpolation=cv2.INTER_CUBIC)

    M = cv2.getRotationMatrix2D((size//2, size//2), angle, 1)
    overlay_rotated = cv2.warpAffine(
        overlay_resized, M, (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT
    )

    if overlay_rotated.shape[2] == 3:
        b,g,r = cv2.split(overlay_rotated)
        alpha = np.ones_like(b) * 255
        overlay_rotated = cv2.merge((b,g,r,alpha))

    b,g,r,a = cv2.split(overlay_rotated)

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size//2, size//2), size//2, 255, -1)

    alpha = (a * (mask/255.0)) / 255.0

    h, w = background.shape[:2]
    x = cx - size//2
    y = cy - size//2

    y1, y2 = max(0,y), min(h,y+size)
    x1, x2 = max(0,x), min(w,x+size)

    overlay_crop = overlay_rotated[0:(y2-y1),0:(x2-x1)]
    alpha_crop = alpha[0:(y2-y1),0:(x2-x1)]

    for c in range(3):
        background[y1:y2, x1:x2, c] = (
            alpha_crop * overlay_crop[:,:,c] +
            (1 - alpha_crop) * background[y1:y2, x1:x2, c]
        )

    return background

# ==========================
# Camera
# ==========================
cap = cv2.VideoCapture(0)
start_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        current_time = time.time()

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            h, w, _ = frame.shape

            landmarks = []
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x,y), 5, (0,0,255), -1)
                landmarks.extend([lm.x, lm.y, lm.z])

            data = np.array(landmarks).reshape(1,-1)
            raw_prediction = model.predict(data)[0]
            probs = model.predict_proba(data)[0]
            confidence = np.max(probs)

            prediction = normalize(raw_prediction)

            if confidence >= CONFIDENCE_THRESHOLD:
                if sequence_index == 0:
                    sequence_start_time = current_time

                if sequence_start_time and current_time - sequence_start_time <= sequence_timeout:
                    if prediction == SEQUENCE[sequence_index]:
                        sequence_index += 1
                        time.sleep(0.25)
                else:
                    sequence_index = 0
                    sequence_start_time = None

            if sequence_index == len(SEQUENCE):
                animation_active = True
                animation_start = current_time
                sequence_index = 0
                sequence_start_time = None

                pygame.mixer.music.stop()
                pygame.mixer.music.play()

            if animation_active:

                progress = (current_time - animation_start) / animation_duration

                if progress <= 1:

                    cx = int(hand_landmarks[9].x * w)
                    cy = int(hand_landmarks[9].y * h)

                    # -------- FIXED SIZE (NO GROWTH) --------
                    wrist = hand_landmarks[0]
                    palm  = hand_landmarks[9]
                    hand_size = math.hypot(
                        (palm.x - wrist.x) * w,
                        (palm.y - wrist.y) * h
                    )

                    size = int(hand_size * 5)   # Constant size

                    angle = progress * 2160  # Fast spin

                    frame = overlay_circle(
                        frame,
                        rasengan_sprite,
                        cx,
                        cy,
                        size,
                        angle
                    )

                    # Impact flash
                    if progress > 0.85:
                        white = np.ones_like(frame) * 255
                        frame = cv2.addWeighted(frame, 0.7, white, 0.3, 0)

                    # Screen shake
                    shake_x = random.randint(-10,10)
                    shake_y = random.randint(-10,10)
                    M = np.float32([[1,0,shake_x],[0,1,shake_y]])
                    frame = cv2.warpAffine(frame, M, (w,h))

                else:
                    animation_active = False

        cv2.imshow("Naruto AI - Stable Rasengan", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
pygame.quit()

"""python naruto.py"""