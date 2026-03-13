# ⚡ Quick Start Guide - Get APK in 30 Minutes

**Goal:** Build and install your gesture recognition APK with minimal effort.

---

## 🎯 Fastest Path (GitHub Actions - No Installation)

### Step 1: Create GitHub Account (2 minutes)

Already have one? Skip to Step 2.

1. Go to https://github.com
2. Click "Sign up"
3. Follow prompts (username, email, password)
4. Verify email

**Done? ✅ Continue to Step 2**

---

### Step 2: Upload Project (5 minutes)

1. **Download this project**
   - You should have: `gesture-recognition-android.zip`
   - Extract it to your desktop

2. **Create repository on GitHub**
   - Click green "New" button (top left)
   - Repository name: `gesture-recognition-android`
   - Keep "Public" selected
   - Click "Create repository"

3. **Upload files**
   - On the new repo page, click "uploading an existing file"
   - Open extracted folder
   - Select ALL files and folders
   - Drag into GitHub upload area
   - Wait for upload (may take 2-3 minutes)
   - Commit message: "Initial commit"
   - Click "Commit changes"

**Done? ✅ Continue to Step 3**

---

### Step 3: Add Model Files (10 minutes)

⚠️ **CRITICAL STEP** - App won't work without these!

#### 3.1 Add Your ONNX Model

1. **On your laptop, find:**
   ```
   output_session_3_ph_2/gesture_model.onnx
   ```

2. **Upload to GitHub:**
   - In your repository, click: `app` → `src` → `main` → `assets`
   - Click "Add file" → "Upload files"
   - Drag `gesture_model.onnx` here
   - Commit message: "Add gesture model"
   - Click "Commit changes"

**If you have `gesture_model.onnx.data` too:**
   - Upload it the same way

#### 3.2 Add MediaPipe Model

1. **Download MediaPipe hand model:**
   - Link: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   - Right-click → Save as: `hand_landmarker.task`

2. **Upload to GitHub:**
   - Same location: `app/src/main/assets/`
   - Upload `hand_landmarker.task`
   - Commit: "Add MediaPipe model"

#### 3.3 Verify Files

In `app/src/main/assets/` you should see:
- ✅ gesture_model.onnx
- ✅ hand_landmarker.task
- ✅ (optional) gesture_model.onnx.data

**Done? ✅ Continue to Step 4**

---

### Step 4: Build APK (5-10 minutes)

**Automatic build already started!** GitHub Actions builds automatically when you commit.

1. **Check build status:**
   - Go to "Actions" tab (top of repo)
   - You should see a yellow circle (building) or green checkmark (done)
   - Click on the running/completed workflow

2. **Wait for build:**
   - Watch the progress (optional - can close and check later)
   - Takes 5-10 minutes
   - Green checkmark = Success ✅
   - Red X = Failed ❌ (see Troubleshooting below)

**Done? ✅ Continue to Step 5**

---

### Step 5: Download APK (2 minutes)

1. **Go to Actions tab**
2. **Click latest workflow** (should have green checkmark)
3. **Scroll down to "Artifacts" section**
4. **Download: `gesture-recognition-debug`**
   - This downloads a ZIP file
5. **Extract the ZIP**
   - Inside you'll find: `app-debug.apk`

**Got the APK? ✅ Continue to Step 6**

---

### Step 6: Install on Tablet (5 minutes)

#### 6.1 Transfer APK to Tablet

**Method A: USB Cable**
```
1. Connect tablet to computer
2. Copy app-debug.apk to tablet's Downloads folder
3. Disconnect
```

**Method B: Google Drive**
```
1. Upload app-debug.apk to Google Drive
2. Open Drive on tablet
3. Download the APK
```

**Method C: Email**
```
1. Email the APK to yourself
2. Open email on tablet
3. Download attachment
```

#### 6.2 Install APK

1. **Open Files app** on tablet
2. **Navigate to Downloads**
3. **Tap `app-debug.apk`**
4. **If prompted:**
   - "Install from Unknown Sources" → Allow
   - Or: Settings → Security → Enable "Unknown Sources"
5. **Tap "Install"**
6. **Wait for installation** (30 seconds)
7. **Tap "Open"**

#### 6.3 Grant Permission

1. App will ask for Camera permission
2. Tap "Allow"
3. App starts!

**Done! 🎉 Your app is running!**

---

## 🎮 Using the App

1. **Hold tablet with camera facing you**
2. **Show your hand to camera**
3. **Wait 2-3 seconds** (collecting frames)
4. **Perform a gesture**
5. **See prediction** at bottom of screen

**Gestures to try:**
- 👍 Thumb up
- 👎 Thumb down
- ✋ Stop sign
- ✌️ V gesture
- ➡️ Swipe right
- ⬅️ Swipe left

---

## ⚠️ Troubleshooting

### Build Failed (Red X in Actions)

1. **Click the failed workflow**
2. **Click "build" job**
3. **Expand "Build Debug APK" step**
4. **Look for error message**

**Common errors:**

**"gesture_model.onnx not found"**
```
Solution: Go to Step 3 → Verify model files uploaded
```

**"hand_landmarker.task not found"**
```
Solution: Download and upload MediaPipe model (Step 3.2)
```

**"Out of memory"**
```
Solution: Model too large, check if INT8 quantized
```

---

### App Crashes on Launch

**Error: "Failed to load model"**
```
Cause: Model files missing or corrupted
Solution: Re-upload model files, rebuild APK
```

**Error: "Camera permission denied"**
```
Solution: Settings → Apps → Gesture Recognition → Permissions → Enable Camera
```

---

### No Hand Detected

```
Check:
1. Camera working? (test with Camera app)
2. Hand in frame?
3. Good lighting?
4. hand_landmarker.task uploaded correctly?
```

---

### Low FPS / Laggy

```
Solutions:
1. Close background apps
2. Restart tablet
3. Use newer tablet (2019+, 2GB+ RAM)
```

---

## 📊 Quick Reference

| Step | Time | What You Need |
|------|------|---------------|
| 1. GitHub account | 2 min | Email address |
| 2. Upload project | 5 min | Project ZIP |
| 3. Add models | 10 min | gesture_model.onnx + internet |
| 4. Build | 5-10 min | Wait |
| 5. Download | 2 min | - |
| 6. Install | 5 min | Tablet |
| **TOTAL** | **~30 min** | |

---

## 🔄 Making Changes

### Want to rebuild with changes?

1. **Edit files on GitHub:**
   - Navigate to file
   - Click pencil icon (edit)
   - Make changes
   - Commit

2. **Build automatically triggers**
   - Go to Actions tab
   - Wait for new build
   - Download new APK

### Want to change gestures or settings?

**Edit:** `app/src/main/java/com/gesture/recognition/Config.kt`

**Change these values:**
```kotlin
CONFIDENCE_THRESHOLD = 0.6f    // Lower = more sensitive
SEQUENCE_LENGTH = 15           // Frames per prediction
```

**Then:** Commit → Rebuild → Download new APK

---

## ✅ Success Checklist

Confirm everything works:

- [ ] App installs without errors
- [ ] Camera permission granted
- [ ] Camera feed shows on screen
- [ ] Hand detected (can see detection)
- [ ] Gestures recognized correctly
- [ ] FPS shows 20-30
- [ ] No crashes

**All checked? You're done! 🎉**

---

## 📚 Next Steps

- **Want better accuracy?** Retrain model with more data
- **Want more gestures?** Update Config.kt and retrain
- **Want to customize UI?** Edit activity_main.xml
- **Want to optimize?** Check Performance section in README.md

---

## 🆘 Need Help?

1. **Re-read this guide carefully**
2. **Check Troubleshooting section**
3. **Review MODEL_SETUP.md for model file issues**
4. **Check Actions build log for errors**
5. **Verify all files in correct location**

**Most issues are:**
- ❌ Model files not uploaded
- ❌ Wrong file location
- ❌ Camera permission not granted
- ❌ Old Android version (need 7.0+)

---

**You're all set! Go build your gesture recognition app! 🚀**
