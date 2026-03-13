# Gesture Recognition Android App

Real-time hand gesture recognition using MediaPipe Hands and ONNX Runtime with TCN model.

## 📋 Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Setup Instructions](#setup-instructions)
- [Adding Your Model](#adding-your-model)
- [Building APK](#building-apk)
- [Installation](#installation)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)

---

## 🎯 Overview

This Android app performs real-time gesture recognition using:
- **MediaPipe Hands**: 21-point hand landmark detection
- **ONNX Runtime**: INT8 quantized TCN model inference
- **CameraX**: Modern Android camera API

**Supported Gestures (11 classes):**
1. doing_other_things
2. swipe_left
3. swipe_right
4. thumb_down
5. thumb_up
6. v_gesture
7. top
8. left_gesture
9. right_gesture
10. stop_sign
11. heart

---

## 🚀 Quick Start

### Option 1: Build with GitHub Actions (RECOMMENDED - No Installation)

1. **Create GitHub Account** (if you don't have one)
   - Go to https://github.com
   - Sign up for free

2. **Create New Repository**
   - Click "New Repository"
   - Name: `gesture-recognition-android`
   - Visibility: Public or Private (both work)
   - Click "Create repository"

3. **Upload Project**
   - Download this project as ZIP
   - Go to your repository
   - Click "Add file" → "Upload files"
   - Drag and drop all project files
   - Commit changes

4. **Add Your Model Files** (IMPORTANT!)
   ```
   app/src/main/assets/
   ├── gesture_model.onnx          ← YOUR MODEL HERE
   ├── gesture_model.onnx.data     ← YOUR MODEL DATA HERE (if separate)
   └── hand_landmarker.task        ← Download from MediaPipe
   ```

5. **Download hand_landmarker.task**
   - Go to: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   - Save to `app/src/main/assets/`

6. **Trigger Build**
   - GitHub automatically builds when you push
   - Or go to "Actions" tab → "Build Android APK" → "Run workflow"

7. **Download APK**
   - Wait 5-10 minutes for build
   - Go to "Actions" tab
   - Click latest workflow run (green checkmark)
   - Scroll to "Artifacts" section
   - Download `gesture-recognition-debug`
   - Extract ZIP to get `app-debug.apk`

8. **Install on Tablet**
   - Transfer APK to tablet (USB, Google Drive, etc.)
   - Enable "Install from Unknown Sources" in Settings
   - Tap APK file → Install

---

### Option 2: Build Locally with Android Studio

**Prerequisites:**
- Android Studio (latest version)
- JDK 17
- Android SDK 24+

**Steps:**

1. **Open Project**
   ```bash
   # Extract project
   unzip gesture-recognition-android.zip
   cd gesture-recognition-android
   
   # Open in Android Studio
   # File → Open → Select project folder
   ```

2. **Add Model Files**
   ```
   Place in: app/src/main/assets/
   - gesture_model.onnx
   - gesture_model.onnx.data (if separate)
   - hand_landmarker.task (download from MediaPipe)
   ```

3. **Sync Gradle**
   - Android Studio will prompt to sync
   - Click "Sync Now"
   - Wait for dependencies to download

4. **Build APK**
   - Build → Build Bundle(s)/APK(s) → Build APK(s)
   - APK location: `app/build/outputs/apk/debug/app-debug.apk`

5. **Install**
   - Connect tablet via USB
   - Enable USB Debugging on tablet
   - Click "Run" in Android Studio

---

## 📦 Adding Your Model

### CRITICAL: You Must Add These Files

The app **will not work** without these files:

#### 1. Your ONNX Model
```
Source: Your trained model
Location: app/src/main/assets/gesture_model.onnx
```

**How to add:**
- If building locally: Copy to `app/src/main/assets/`
- If using GitHub: Upload to repository at this path

#### 2. MediaPipe Hand Landmarker
```
Download: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
Location: app/src/main/assets/hand_landmarker.task
Size: ~22 MB
```

**Why needed:** For hand landmark detection (21 points)

#### 3. Model Data File (if separate)
```
Source: Your model (if .onnx.data exists)
Location: app/src/main/assets/gesture_model.onnx.data
```

### File Structure Check
```
app/src/main/assets/
├── gesture_model.onnx           ✓ Must exist
├── gesture_model.onnx.data      ? Optional (if your model has it)
└── hand_landmarker.task         ✓ Must exist
```

---

## 🔨 Building APK

### Method 1: GitHub Actions (Zero Installation)

**When it builds:**
- Automatically when you push code
- Manual trigger from Actions tab

**Build time:** 5-10 minutes

**Output:**
- artifact: `gesture-recognition-debug.zip`
- Contains: `app-debug.apk`

**To download:**
1. Go to repository → Actions tab
2. Click latest workflow (green checkmark)
3. Scroll to "Artifacts"
4. Download and extract

---

### Method 2: Command Line (Lightweight)

**Requirements:**
- Java JDK 17
- Android Command Line Tools

**Build command:**
```bash
./gradlew assembleDebug
```

**Output:**
```
app/build/outputs/apk/debug/app-debug.apk
```

---

### Method 3: Android Studio (Full IDE)

**Build:**
1. Build → Build Bundle(s)/APK(s) → Build APK(s)
2. Wait for build to complete
3. Click "locate" in notification

**Run on device:**
1. Connect tablet via USB
2. Enable USB Debugging
3. Click "Run" (green triangle)

---

## 📱 Installation on Tablet

### Step 1: Transfer APK

**USB Method:**
```bash
adb install app-debug.apk
```

**Google Drive Method:**
1. Upload APK to Google Drive
2. Open Drive on tablet
3. Download APK

**Direct Transfer:**
1. Connect tablet to PC
2. Copy APK to Downloads folder
3. Disconnect

### Step 2: Enable Unknown Sources

1. Settings → Security
2. Enable "Install from Unknown Sources"
3. Or: Allow for specific app (Chrome, Files, etc.)

### Step 3: Install

1. Open Files app
2. Navigate to Downloads
3. Tap `app-debug.apk`
4. Tap "Install"
5. Wait for installation
6. Tap "Open"

### Step 4: Grant Permissions

1. App will request Camera permission
2. Tap "Allow"
3. App starts

---

## 🐛 Troubleshooting

### Build Errors

**Error: "Model file not found"**
```
Solution: Add gesture_model.onnx to app/src/main/assets/
Check: File exists and name matches exactly
```

**Error: "hand_landmarker.task not found"**
```
Solution: Download from MediaPipe and add to assets
URL: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

**Error: "Failed to create ONNX session"**
```
Possible causes:
1. Model file corrupted → Re-export from training
2. Incompatible model format → Check ONNX version
3. Model too large → Check INT8 quantization applied
```

**Error: GitHub Actions build fails**
```
Solutions:
1. Check Actions tab for error log
2. Ensure all files committed properly
3. Check file paths match exactly
4. Verify gradle.properties correct
```

### Runtime Errors

**App crashes on launch**
```
Check:
1. Camera permission granted
2. Model files in assets folder
3. Minimum Android version (API 24+)
4. Check logcat: adb logcat | grep GestureRecognition
```

**Low FPS / Lag**
```
Solutions:
1. Reduce camera resolution in Config.kt
2. Use GPU delegate (if supported)
3. Check tablet specs (need 2GB+ RAM)
4. Close background apps
```

**Inaccurate predictions**
```
Check:
1. Normalization matches training exactly
2. Model is INT8 quantized correctly
3. Lighting conditions adequate
4. Hand fully visible in frame
```

**"No hand detected" always shown**
```
Check:
1. hand_landmarker.task file exists
2. Camera working (test with Camera app)
3. Hand within frame boundaries
4. Adequate lighting
```

---

## 🔧 Technical Details

### Architecture

```
Camera Feed (YUV)
    ↓
CameraX (1280x720 @ 30fps)
    ↓
MediaPipe Hands → 21 landmarks (x,y,z) = 63 features
    ↓
LandmarkNormalizer → 4-level normalization:
    1. Wrist-relative (position invariance)
    2. Scale by hand size (size invariance)
    3. Outlier clipping ([-2.0, 2.0])
    4. Flatten to 63 features
    ↓
SequenceBuffer → Circular buffer (15 frames)
    ↓
ONNX Runtime → TCN model inference
    Input: [1, 15, 63]
    Output: [1, 11] (probabilities)
    ↓
PredictionSmoother → Majority voting (last 5 predictions)
    ↓
UI Display → Gesture + Confidence + FPS
```

### Model Requirements

**Format:** ONNX (INT8 quantized recommended)

**Input Shape:** `[batch=1, sequence=15, features=63]`

**Output Shape:** `[batch=1, classes=11]`

**Input Data:**
- 15 consecutive frames
- Each frame: 21 landmarks × 3 coordinates (x, y, z)
- Normalized: wrist-relative, scaled, clipped

**Output Data:**
- Softmax probabilities for 11 gesture classes
- Argmax for predicted class

### Normalization (Must Match Training)

```kotlin
// Step 1: Wrist-relative
relative = landmarks - wrist

// Step 2: Scale by hand size
hand_scale = max(x_range, y_range)
if (hand_scale < 0.01) hand_scale = 0.01
scaled = relative / hand_scale

// Step 3: Outlier clipping
clipped = clip(scaled, -2.0, 2.0)

// Step 4: Flatten
output = clipped.flatten()  // 63 values
```

### Performance Benchmarks

**Expected Performance:**
- FPS: 20-30 on mid-range tablets
- Latency: 80-120ms per prediction
- Battery: 3-4 hours continuous use
- APK Size: 35-45 MB

**Tested on:**
- Samsung Galaxy Tab (2019+)
- Android 10+
- 3GB+ RAM

### Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| CameraX | 1.3.1 | Camera handling |
| MediaPipe | 0.10.9 | Hand landmark detection |
| ONNX Runtime | 1.17.0 | Model inference |
| Kotlin Coroutines | 1.7.3 | Async processing |

---

## 📝 Configuration

Edit `Config.kt` to modify:

```kotlin
// Sequence parameters
SEQUENCE_LENGTH = 15        // Frames per prediction

// MediaPipe confidence
MP_HANDS_CONFIDENCE = 0.5f
MP_HANDS_TRACKING_CONFIDENCE = 0.5f

// Normalization
NORMALIZATION_CLIP_RANGE = 2.0f
MIN_HAND_SCALE = 0.01f

// Inference
CONFIDENCE_THRESHOLD = 0.6f
PREDICTION_SMOOTHING_WINDOW = 5

// Camera
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
```

---

## 📄 License

This project is provided as-is for your gesture recognition application.

---

## 🆘 Support

**Having issues?**

1. Check this README thoroughly
2. Review Troubleshooting section
3. Check GitHub Issues (if available)
4. Review build logs in GitHub Actions

**Common Issues Checklist:**
- ✅ Model files in assets folder
- ✅ hand_landmarker.task downloaded
- ✅ Camera permission granted
- ✅ Android version 7.0+ (API 24+)
- ✅ 2GB+ RAM
- ✅ Adequate lighting for camera

---

**Built with ❤️ for Real-Time Gesture Recognition**

*Version: 1.0*
*Last Updated: January 2026*
