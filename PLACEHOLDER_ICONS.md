# 🎨 App Icons

## Status: Using Default Android Icons

The app currently uses default Android launcher icons (`ic_launcher`).

## To Customize (Optional)

### Method 1: Android Studio

1. Right-click `res` folder
2. New → Image Asset
3. Choose icon type: "Launcher Icons"
4. Upload your icon image
5. Configure sizes
6. Click "Finish"

### Method 2: Manual

Add these files to respective folders:

```
app/src/main/res/
├── mipmap-mdpi/ic_launcher.png       (48x48)
├── mipmap-hdpi/ic_launcher.png       (72x72)
├── mipmap-xhdpi/ic_launcher.png      (96x96)
├── mipmap-xxhdpi/ic_launcher.png     (144x144)
└── mipmap-xxxhdpi/ic_launcher.png    (192x192)
```

### Method 3: Online Icon Generator

1. Go to: https://romannurik.github.io/AndroidAssetStudio/
2. Upload your logo/icon
3. Download generated pack
4. Extract to `app/src/main/res/`

## Icon Requirements

- **Format:** PNG (with transparency)
- **Recommended size:** 512x512 minimum
- **Style:** Simple, recognizable shape
- **For gesture app:** Could use hand emoji 🤚 or custom hand icon

## Current Default Icon

The app uses Android's default green robot icon. This works fine for testing but you may want to customize for production.

**Priority: Low** (does not affect functionality)
