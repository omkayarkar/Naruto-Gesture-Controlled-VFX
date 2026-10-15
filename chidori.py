import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import pygame
import random
import math

# ==========================
# LOAD MODEL
# ==========================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ==========================
# LOAD VIDEO
# ==========================
chidori_video = cv2.VideoCapture("chidori.mp4")
if not chidori_video.isOpened():
    print("chidori.mp4 not found")
    exit()

video_width  = int(chidori_video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(chidori_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_ratio  = video_height / video_width

# ==========================
# LOAD SOUND
# ==========================
pygame.mixer.init()
chidori_sound = pygame.mixer.Sound("chidori.mp3")

# ==========================
# MEDIAPIPE SETUP
# ==========================
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

def normalize(label):
    return label.lower().replace(" ", "_")

# ==========================
# SEQUENCE
# ==========================
CHIDORI_SEQ = [
    normalize("Chakra Focus"),
    normalize("Lightning Chakra"),
    normalize("Chidori Ready")
]

sequence_index = 0
sequence_start_time = None

SEQ_TIMEOUT = 6
CONF_THRESHOLD = 0.50

chidori_active = False
chidori_start = 0
sound_start = 0

ANIM_DURATION = 6.0

# ==========================
# CENTER CROP (ZOOM)
# ==========================
def center_crop(frame, percent=0.45):
    h, w = frame.shape[:2]
    crop_w = int(w * percent)
    crop_h = int(h * percent)
    start_x = w//2 - crop_w//2
    start_y = h//2 - crop_h//2
    return frame[start_y:start_y+crop_h, start_x:start_x+crop_w]

# ==========================
# CLEAN GREEN REMOVAL
# ==========================
def remove_green(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 70, 70])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3,3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    mask = cv2.bitwise_not(green_mask)

    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = mask.astype(float) / 255.0

    # Spill suppression
    b, g, r = cv2.split(frame)
    g = np.minimum(g, np.maximum(r, b))
    frame = cv2.merge((b, g, r))

    return frame, mask

# ==========================
# OVERLAY FUNCTION
# ==========================
def overlay_video(frame, video_frame, cx, cy, scale_width):

    video_frame = center_crop(video_frame, percent=0.45)

    new_w = int(scale_width)
    new_h = int(scale_width * video_ratio)

    video_frame = cv2.resize(
        video_frame,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC
    )

    processed_frame, mask = remove_green(video_frame)

    h, w = frame.shape[:2]

    offset_y = int(new_h * 0.15)

    jitter_x = random.randint(-6, 6)
    jitter_y = random.randint(-6, 6)

    x = cx - new_w//2 + jitter_x
    y = cy - new_h//2 - offset_y + jitter_y

    y1 = max(0, y)
    y2 = min(h, y + new_h)
    x1 = max(0, x)
    x2 = min(w, x + new_w)

    overlay_crop = processed_frame[y1-y:y2-y, x1-x:x2-x]
    mask_crop = mask[y1-y:y2-y, x1-x:x2-x]

    # Proper alpha blend (sharp)
    for c in range(3):
        frame[y1:y2, x1:x2, c] = (
            mask_crop * overlay_crop[:,:,c] +
            (1 - mask_crop) * frame[y1:y2, x1:x2, c]
        )

    # Controlled glow layer
    glow = cv2.GaussianBlur(overlay_crop, (21,21), 0)

    for c in range(3):
        frame[y1:y2, x1:x2, c] = np.clip(
            frame[y1:y2, x1:x2, c] +
            glow[:,:,c] * mask_crop * 0.25,
            0, 255
        )

    return frame

# ==========================
# CAMERA
# ==========================
cap = cv2.VideoCapture(0)
start_time = time.time()

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int((time.time()-start_time)*1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        current_time = time.time()
        chidori_center = None
        dynamic_width = 400

        if result.hand_landmarks:

            hand_landmarks = result.hand_landmarks[0]
            h,w,_ = frame.shape

            for lm in hand_landmarks:
                x = int(lm.x*w)
                y = int(lm.y*h)
                cv2.circle(frame,(x,y),5,(0,0,255),-1)

            wrist = hand_landmarks[0]
            palm  = hand_landmarks[9]

            x1 = int(wrist.x*w)
            y1 = int(wrist.y*h)
            x2 = int(palm.x*w)
            y2 = int(palm.y*h)

            hand_size = math.hypot(x2-x1, y2-y1)

            dynamic_width = int(hand_size * 6.5)
            dynamic_width = max(350, min(dynamic_width, 1000))

            landmarks = []
            for lm in hand_landmarks:
                landmarks.extend([lm.x,lm.y,lm.z])

            data = np.array(landmarks).reshape(1,-1)
            raw_pred = model.predict(data)[0]
            probs = model.predict_proba(data)[0]
            confidence = np.max(probs)

            pred = normalize(raw_pred)

            cx = int(hand_landmarks[9].x*w)
            cy = int(hand_landmarks[9].y*h)

            if confidence >= CONF_THRESHOLD:

                if sequence_index == 0:
                    sequence_start_time = current_time

                if sequence_start_time and current_time-sequence_start_time <= SEQ_TIMEOUT:

                    if pred == CHIDORI_SEQ[sequence_index]:
                        sequence_index += 1

                else:
                    sequence_index = 0
                    sequence_start_time = None

                if sequence_index == len(CHIDORI_SEQ):
                    chidori_active = True
                    chidori_start = current_time
                    sound_start = current_time
                    chidori_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    chidori_sound.play()
                    sequence_index = 0
                    sequence_start_time = None

            if chidori_active:
                chidori_center = (cx,cy)

        if chidori_active and current_time - sound_start > 6:
            chidori_sound.stop()

        if chidori_active and chidori_center:

            progress = (current_time - chidori_start)/ANIM_DURATION

            if progress <= 1:

                ret2, video_frame = chidori_video.read()
                if not ret2:
                    chidori_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret2, video_frame = chidori_video.read()

                frame = overlay_video(
                    frame,
                    video_frame,
                    chidori_center[0],
                    chidori_center[1],
                    dynamic_width
                )
            else:
                chidori_active = False
                chidori_sound.stop()

        cv2.imshow("Chidori - Sharp Clean Anime Version", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
chidori_video.release()
cv2.destroyAllWindows()
pygame.quit()


"""python chidori.py"""