#!/usr/bin/env bash
# Build Tohu.app from the Swift package and the exported model.
#
#   macos/scripts/build-app.sh            -> macos/build/Tohu.app
#   macos/scripts/build-app.sh --install  -> also copies it to /Applications and (re)starts it
#
# Needs models/export/ (uv run --group export python scripts/export_coreml.py).
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
EXPORT="$ROOT/models/export"
APP="$HERE/build/Tohu.app"
BUNDLE_ID="dev.tohu.Tohu"
VERSION="0.1.0"

for f in Tohu.mlmodelc lexicon.bin guard.json manifest.json; do
  [[ -e "$EXPORT/$f" ]] || { echo "missing $EXPORT/$f - run scripts/export_coreml.py first" >&2; exit 1; }
done

cd "$HERE"
swift build -c release --product TohuApp
swift build -c release --product tohu
BIN="$(swift build -c release --show-bin-path)"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Helpers" "$APP/Contents/Resources"
cp "$BIN/TohuApp" "$APP/Contents/MacOS/Tohu"
# Not next to the app binary: APFS is case-insensitive, so MacOS/tohu would overwrite MacOS/Tohu.
cp "$BIN/tohu" "$APP/Contents/Helpers/tohu"
cp -R "$EXPORT/Tohu.mlmodelc" "$APP/Contents/Resources/"
cp "$EXPORT/lexicon.bin" "$EXPORT/guard.json" "$EXPORT/manifest.json" "$APP/Contents/Resources/"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Tohu</string>
  <key>CFBundleDisplayName</key><string>Tohu</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleExecutable</key><string>Tohu</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$(date +%Y%m%d%H%M)</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSServices</key>
  <array>
    <dict>
      <key>NSMenuItem</key><dict><key>default</key><string>Restore Hungarian Accents</string></dict>
      <key>NSMessage</key><string>restoreAccents</string>
      <key>NSPortName</key><string>Tohu</string>
      <key>NSSendTypes</key><array><string>public.utf8-plain-text</string></array>
      <key>NSReturnTypes</key><array><string>public.utf8-plain-text</string></array>
    </dict>
  </array>
</dict>
</plist>
PLIST

# Ad-hoc signature. Note: macOS ties the Accessibility permission to the signature, so after a
# rebuild you may need to toggle Tohu off and on again in Privacy & Security -> Accessibility.
codesign --force --sign - "$APP/Contents/Helpers/tohu"
codesign --force --sign - "$APP"
echo "built $APP"

if [[ "${1:-}" == "--install" ]]; then
  pkill -x Tohu 2>/dev/null && sleep 0.5 || true
  rm -rf /Applications/Tohu.app
  cp -R "$APP" /Applications/
  /System/Library/CoreServices/pbs -update 2>/dev/null || true   # refresh the Services menu
  open /Applications/Tohu.app
  echo "installed /Applications/Tohu.app"
fi
