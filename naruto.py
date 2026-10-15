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

RASENGAN_SEQUENCE = [
    normalize("Chakra Focus"),
    normalize("Chakra Rotate"),
    normalize("Rasengan Ready")
]
CHIDORI_SEQUENCE = [
    normalize("Chakra Focus"),
    normalize("Lightning Chakra"),
    normalize("Chidori Ready")
]

sequence_timeout = 6.0
CONFIDENCE_THRESHOLD = 0.60
sequence_states = {
    "rasengan": {"sequence": RASENGAN_SEQUENCE, "index": 0, "start": None},
    "chidori": {"sequence": CHIDORI_SEQUENCE, "index": 0, "start": None},
}

animation_active = False
animation_start = 0
animation_duration = 6.0

chidori_video = cv2.VideoCapture("chidori.mp4")
if not chidori_video.isOpened():
    raise SystemExit("Could not open chidori.mp4")
video_width = int(chidori_video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(chidori_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
if video_width <= 0 or video_height <= 0:
    chidori_video.release()
    raise SystemExit("Invalid chidori.mp4 dimensions")
video_ratio = video_height / video_width
chidori_sound = pygame.mixer.Sound("chidori.mp3")
chidori_active = False
chidori_start = 0.0


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

# Chidori green-screen overlay helpers
def center_crop(frame, percent=0.45):
    h, w = frame.shape[:2]
    cw, ch = max(1, int(w * percent)), max(1, int(h * percent))
    x, y = max(0, (w - cw) // 2), max(0, (h - ch) // 2)
    return frame[y:y + ch, x:x + cw]


def remove_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(
        hsv, np.array([35, 70, 70]), np.array([85, 255, 255])
    )
    kernel = np.ones((3, 3), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    alpha = cv2.GaussianBlur(
        cv2.bitwise_not(mask_green), (5, 5), 0
    ).astype(np.float32) / 255.0
    b, g, r = cv2.split(frame)
    g = np.minimum(g, np.maximum(r, b))
    return cv2.merge((b, g, r)), alpha


def overlay_video(frame, video_frame, cx, cy, scale_width):
    video_frame = center_crop(video_frame, 0.45)
    nw = max(1, int(scale_width))
    nh = max(1, int(nw * video_ratio))
    video_frame = cv2.resize(video_frame, (nw, nh), interpolation=cv2.INTER_CUBIC)
    processed, alpha = remove_green(video_frame)

    h, w = frame.shape[:2]
    x = cx - nw // 2 + random.randint(-6, 6)
    y = cy - nh // 2 - int(nh * 0.15) + random.randint(-6, 6)
    x1, x2 = max(0, x), min(w, x + nw)
    y1, y2 = max(0, y), min(h, y + nh)
    if x1 >= x2 or y1 >= y2:
        return frame

    crop = processed[y1-y:y2-y, x1-x:x2-x].astype(np.float32)
    a = alpha[y1-y:y2-y, x1-x:x2-x, None]
    region = frame[y1:y2, x1:x2].astype(np.float32)
    blended = a * crop + (1.0 - a) * region
    glow = cv2.GaussianBlur(crop, (21, 21), 0)
    frame[y1:y2, x1:x2] = np.clip(blended + glow * a * 0.25, 0, 255).astype(np.uint8)
    return frame


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
        chidori_center = None
        dynamic_width = 400

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            h, w, _ = frame.shape
            xs = [int(lm.x * w) for lm in hand_landmarks]
            ys = [int(lm.y * h) for lm in hand_landmarks]
            hand_diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            dynamic_width = max(350, min(int(hand_diagonal * 6.5), 1000))
            chidori_center = (
                int(hand_landmarks[9].x * w),
                int(hand_landmarks[9].y * h)
            )

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
                for effect_name, state in sequence_states.items():
                    seq = state["sequence"]
                    idx = state["index"]

                    if state["start"] is not None and current_time - state["start"] > sequence_timeout:
                        state["index"] = 0
                        state["start"] = None
                        idx = 0

                    if prediction == seq[idx]:
                        if idx == 0:
                            state["start"] = current_time
                        state["index"] += 1

                        if state["index"] == len(seq):
                            state["index"] = 0
                            state["start"] = None

                            if effect_name == "rasengan":
                                animation_active = True
                                animation_start = current_time
                                pygame.mixer.music.stop()
                                pygame.mixer.music.play()
                            else:
                                chidori_active = True
                                chidori_start = current_time
                                chidori_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                chidori_sound.stop()
                                chidori_sound.play()

                    elif prediction == seq[0]:
                        state["index"] = 1
                        state["start"] = current_time

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

        if chidori_active:
            if current_time - chidori_start >= animation_duration:
                chidori_active = False
                chidori_sound.stop()
            elif chidori_center is not None:
                ok, video_frame = chidori_video.read()
                if not ok:
                    chidori_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, video_frame = chidori_video.read()
                if ok:
                    frame = overlay_video(
                        frame, video_frame,
                        chidori_center[0], chidori_center[1],
                        dynamic_width
                    )

        cv2.imshow("Naruto AI - Stable Rasengan", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
chidori_video.release()
cv2.destroyAllWindows()
pygame.quit()

"""python naruto.py"""