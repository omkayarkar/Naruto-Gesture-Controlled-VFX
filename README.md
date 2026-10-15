# Naruto Gesture-Controlled VFX

Control Naruto-inspired visual effects using hand gestures and your webcam! This Python project combines MediaPipe hand tracking, a trained machine-learning classifier, OpenCV overlays, and sound effects.

## Features

- **Rasengan:** rotating energy effect, sound, impact flash, and screen shake.
- **Chidori:** lightning effect composited from a video, with sound.
- Real-time webcam hand tracking using MediaPipe.
- Gesture-sequence recognition using a trained classifier.
- Separate Rasengan and Chidori scripts.

## Requirements

- Windows 10/11
- Python 3.10 or 3.11 recommended
- A working webcam
- Speakers or headphones

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/omkayarkar/Naruto-Gesture-Controlled-VFX.git
cd Naruto-Gesture-Controlled-VFX
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, open a new terminal or use this command for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If MediaPipe is missing, make sure the virtual environment is activated and install it with:

```powershell
python -m pip install mediapipe
```

## Run the application

### Combined Rasengan and Chidori app

```powershell
python naruto.py
```

Allow webcam access if Windows asks for permission. Keep your hand visible, well-lit, and inside the camera frame.

### Chidori standalone app

```powershell
python chidori.py
```

The standalone script runs the Chidori effect independently.

Press **Q** while the camera window is focused to quit.

## Gesture sequences

Perform the gestures in the listed order. The app expects each sequence to be completed within approximately six seconds.

| Effect | Gesture sequence |
|---|---|
| Rasengan | `Chakra Focus` → `Chakra Rotate` → `Rasengan Ready` |
| Chidori | `Chakra Focus` → `Lightning Chakra` → `Chidori Ready` |

The gesture classifier must recognize these labels. Hold each gesture clearly and change to the next gesture deliberately. Recognition depends on lighting, camera positioning, and the trained model.

## Important project files

| File | Purpose |
|---|---|
| `naruto.py` | Main combined VFX application |
| `chidori.py` | Standalone Chidori application |
| `collect_data.py` | Collect gesture landmark samples |
| `train_model.py` | Train the gesture classifier |
| `model.pkl` | Trained gesture classification model |
| `hand_landmarker.task` | MediaPipe hand-landmark model |
| `rasengan.png` | Rasengan visual asset |
| `rasengan.mp3` | Rasengan sound |
| `chidori.mp4` | Chidori visual-effect video |
| `chidori.mp3` | Chidori sound |
| `requirements.txt` | Python dependencies |

Keep the asset files in the project root alongside the Python scripts. The application loads them using relative paths.

## Troubleshooting

### `ModuleNotFoundError: No module named 'mediapipe'`

Activate the virtual environment and run:

```powershell
python -m pip install -r requirements.txt
```

### Webcam does not open

- Close other applications using the camera.
- Check Windows camera permissions.
- If you have multiple cameras, update `cv2.VideoCapture(0)` in the script to use the correct camera index.

### Effects do not trigger

- Ensure your hand is fully visible and well-lit.
- Perform the required gesture sequence in order and within the timeout.
- The classifier must predict the expected gesture labels with sufficient confidence.
- The included model was trained on specific gesture samples; different users or hand positions may reduce accuracy.

### Sound is not playing

Check Windows output-device settings and confirm that the required MP3 files are present.

## Development workflow

To collect more training samples:

```powershell
python collect_data.py
```

After collecting the required gesture samples, retrain the model:

```powershell
python train_model.py
```

Retraining overwrites `model.pkl`. Keep a backup if you want to preserve the current trained model.

## Notes

- This is an experimental computer-vision project, not a production-grade gesture-recognition system.
- Performance depends on the webcam, lighting, hardware, and classifier accuracy.
- Use only media assets you have permission to distribute.

## License

No license has been specified yet. Unless a license is added, standard copyright restrictions apply to this repository.
