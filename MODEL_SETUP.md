# 📦 Model Files Setup

## ⚠️ CRITICAL: Required Files

Your app will **NOT work** without these files. Please read carefully.

---

## 📋 Files You Need

### 1. Your ONNX Gesture Model ✅

**File:** `gesture_model.onnx`

**Source:** Your trained TCN model (from your laptop)

**Location on your laptop:**
```
output_session_3_ph_2/gesture_model.onnx
```

**Where to put it:**
```
gesture-recognition-android/
└── app/
    └── src/
        └── main/
            └── assets/
                └── gesture_model.onnx  ← PUT IT HERE
```

**How to add:**

**If using GitHub Actions:**
1. Go to your GitHub repository
2. Navigate to: `app/src/main/assets/`
3. Click "Add file" → "Upload files"
4. Upload `gesture_model.onnx`
5. Commit changes

**If using Android Studio:**
1. Open project in Android Studio
2. In Project view, expand: `app → src → main → assets`
3. Right-click `assets` → New → Directory (if doesn't exist)
4. Copy your `gesture_model.onnx` here

**File size:** ~5-10 MB (INT8 quantized)

---

### 2. ONNX Model Data (If Separate) ⚠️

**File:** `gesture_model.onnx.data`

**Check if you have it:**
```bash
ls output_session_3_ph_2/
# If you see both .onnx and .onnx.data, you need both
```

**Where to put it:**
```
gesture-recognition-android/
└── app/
    └── src/
        └── main/
            └── assets/
                ├── gesture_model.onnx
                └── gesture_model.onnx.data  ← IF IT EXISTS
```

**Note:** Some models have weights in a separate `.data` file. If you have it, upload it too.

---

### 3. MediaPipe Hand Landmarker ✅

**File:** `hand_landmarker.task`

**Download link:**
```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

**How to download:**

**Method 1: Direct download**
```bash
cd gesture-recognition-android/app/src/main/assets/
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

**Method 2: Browser**
1. Open link in browser
2. Save as `hand_landmarker.task`
3. Move to `app/src/main/assets/`

**File size:** ~22 MB

**Why needed:** This detects the 21 hand landmarks that your model needs

---

## ✅ Verification Checklist

After adding files, verify:

```bash
cd gesture-recognition-android/app/src/main/assets/
ls -lh
```

**You should see:**
```
-rw-r--r--  gesture_model.onnx       (~5-10 MB)
-rw-r--r--  gesture_model.onnx.data  (~5-10 MB, optional)
-rw-r--r--  hand_landmarker.task     (~22 MB)
```

**Total size:** ~30-45 MB

---

## 🚫 Common Mistakes

### ❌ Wrong: Files in wrong location
```
gesture-recognition-android/
├── gesture_model.onnx         ← WRONG (root)
└── app/
    └── gesture_model.onnx     ← WRONG (app root)
```

### ✅ Correct: Files in assets folder
```
gesture-recognition-android/
└── app/
    └── src/
        └── main/
            └── assets/
                ├── gesture_model.onnx     ← CORRECT
                └── hand_landmarker.task   ← CORRECT
```

---

### ❌ Wrong: Incorrect filenames
```
assets/
├── model.onnx                    ← WRONG NAME
├── gesture_recognition.onnx      ← WRONG NAME
└── hand_landmarker_v1.task       ← WRONG NAME
```

### ✅ Correct: Exact filenames
```
assets/
├── gesture_model.onnx            ← EXACT NAME
└── hand_landmarker.task          ← EXACT NAME
```

---

### ❌ Wrong: Forgot MediaPipe model
```
assets/
└── gesture_model.onnx            ← Missing hand_landmarker.task!
```

### ✅ Correct: Both models present
```
assets/
├── gesture_model.onnx
└── hand_landmarker.task          ← Both needed!
```

---

## 🔍 How to Find Your Model Files

### On Your Laptop

Your files are likely in:
```bash
# Your training output directory
/NAS1/seelam_repo_falcon2/aswin_/output_session_3_ph_2/

# Or wherever you ran the training script
```

**Find them:**
```bash
cd /path/to/your/project/
find . -name "gesture_model.onnx"
find . -name "*.onnx"
```

**Copy from NAS:**
```bash
# If on NAS, copy to local first
cp /NAS1/seelam_repo_falcon2/aswin_/output_session_3_ph_2/gesture_model.onnx ~/Downloads/
```

---

## 📤 Upload Methods

### Method 1: GitHub Web Interface (Easiest)

1. Go to your repository on GitHub
2. Navigate: `app` → `src` → `main` → `assets`
3. Click "Add file" → "Upload files"
4. Drag both files:
   - `gesture_model.onnx`
   - `hand_landmarker.task`
5. Write commit message: "Add model files"
6. Click "Commit changes"

**Note:** GitHub has 100MB per file limit. Your files should be well under this.

---

### Method 2: Git Command Line

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/gesture-recognition-android.git
cd gesture-recognition-android

# Create assets directory
mkdir -p app/src/main/assets

# Copy your model
cp /path/to/your/gesture_model.onnx app/src/main/assets/

# Download MediaPipe model
cd app/src/main/assets
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# Commit and push
cd ../../../..
git add app/src/main/assets/
git commit -m "Add model files"
git push
```

---

### Method 3: Android Studio

1. Open project in Android Studio
2. Switch to "Project" view (dropdown at top left)
3. Expand: `app` → `src` → `main`
4. Right-click `main` → New → Directory
5. Name it: `assets`
6. Right-click `assets` → Show in Explorer/Finder
7. Copy your files here:
   - `gesture_model.onnx`
   - `hand_landmarker.task`
8. Sync Gradle (File → Sync Project with Gradle Files)

---

## 🧪 Test Model Files

### After adding files, test:

**GitHub Actions:**
1. Go to Actions tab
2. Click "Run workflow"
3. If build succeeds → Files are correct ✅
4. If build fails → Check error log

**Android Studio:**
1. Build → Clean Project
2. Build → Rebuild Project
3. Check Build output for errors
4. Run app on device/emulator

**Expected output:**
```
Loading ONNX model: gesture_model.onnx
Model loaded: 8234567 bytes
ONNX session created successfully
Initializing MediaPipe HandLandmarker...
MediaPipe initialized successfully
```

---

## ❓ Troubleshooting

### Build Error: "gesture_model.onnx not found"

**Cause:** File not in assets folder or wrong name

**Solution:**
1. Check exact path: `app/src/main/assets/gesture_model.onnx`
2. Check filename matches exactly (case-sensitive)
3. Rebuild project

---

### Build Error: "hand_landmarker.task not found"

**Cause:** MediaPipe model not downloaded

**Solution:**
1. Download from link above
2. Place in `app/src/main/assets/`
3. Filename must be exact: `hand_landmarker.task`

---

### Runtime Error: "Failed to create ONNX session"

**Possible causes:**
1. Model file corrupted during transfer
2. Wrong ONNX version
3. Model not INT8 quantized

**Solution:**
```bash
# Check model integrity
md5sum gesture_model.onnx  # Compare with original

# Re-export model
python export_to_onnx.py  # Or your export script

# Verify ONNX version
python -c "import onnx; print(onnx.__version__)"  # Should be 1.13+
```

---

### Large File Warning on GitHub

**If file > 50MB:**

GitHub will warn you. Solutions:

1. **Use Git LFS** (Large File Storage):
```bash
git lfs install
git lfs track "*.onnx"
git lfs track "*.task"
git add .gitattributes
git add app/src/main/assets/
git commit -m "Add model files with LFS"
git push
```

2. **Or use Release assets:**
- Build without model first
- Add model files manually after downloading APK

---

## 📊 File Size Reference

| File | Expected Size | Max Size |
|------|---------------|----------|
| gesture_model.onnx | 5-10 MB | 50 MB |
| gesture_model.onnx.data | 5-10 MB | 50 MB |
| hand_landmarker.task | ~22 MB | 30 MB |
| **Total** | **~30-45 MB** | **100 MB** |

---

## ✅ Final Checklist

Before building, confirm:

- [ ] `gesture_model.onnx` in `app/src/main/assets/`
- [ ] `hand_landmarker.task` in `app/src/main/assets/`
- [ ] Files not corrupted (check file sizes)
- [ ] Filenames exact (case-sensitive)
- [ ] Files committed to git (if using GitHub)
- [ ] Build completes without errors

**If all checked → You're ready to build! 🚀**

---

## 🆘 Still Having Issues?

1. Double-check paths (most common issue)
2. Verify file sizes match expected
3. Try clean rebuild
4. Check GitHub Actions build log
5. Verify Android Studio sees files (Project view)

**Debug command:**
```bash
# List all files in assets
find app/src/main/assets -type f -ls
```

---

**Next Step:** Go to main README.md → "Building APK" section
