#!/usr/bin/env bash

# Navigate to the script's directory, then into SwiftBuddy
cd "$(dirname "$0")/SwiftBuddy" || exit

echo "🔄 Generating SwiftBuddy Xcode Project..."
python3 generate_xcodeproj.py

echo "🚀 Opening SwiftBuddy.xcodeproj in Xcode..."
open SwiftBuddy.xcodeproj
