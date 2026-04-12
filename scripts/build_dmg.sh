#!/bin/bash
set -e

# Script expects path to the extracted .app
APP_PATH=$1

if [ -z "$APP_PATH" ]; then
    echo "Usage: $0 /path/to/SwiftBuddy.app"
    exit 1
fi

APP_NAME=$(basename "$APP_PATH")

echo "=========================================="
echo "1. Applying Ad-Hoc open-source signature"
echo "=========================================="
# Force a local ad-hoc signature so the binary structure is valid for macOS execution locally
codesign --force --deep --sign - "$APP_PATH"

echo "=========================================="
echo "2. Package Ad-Hoc build into DMG"
echo "=========================================="
# Extract marketing version number directly from the compiled app's bundle
VERSION=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$APP_PATH/Contents/Info.plist" || echo "latest")
mkdir -p output
DMG_NAME="SwiftBuddy-macOS-v${VERSION}.dmg"

create-dmg \
  --volname "SwiftBuddy" \
  --volicon "$APP_PATH/Contents/Resources/AppIcon.icns" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "SwiftBuddy.app" 200 190 \
  --hide-extension "SwiftBuddy.app" \
  --app-drop-link 600 185 \
  "output/$DMG_NAME" \
  "$APP_PATH"

echo "=========================================="
echo "SUCCESS! Created UNSIGNED (Ad-Hoc) output/$DMG_NAME"
echo "=========================================="
