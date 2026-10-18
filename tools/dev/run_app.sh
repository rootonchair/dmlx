#!/usr/bin/env bash
# Development run: wraps the debug binary in a minimal .app (engine runs from ../engine via uv).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
(cd "$ROOT/app" && swift build --product dMLX)
BIN="$(cd "$ROOT/app" && swift build --product dMLX --show-bin-path)/dMLX"
APP="$ROOT/app/.build/dMLX-dev.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/dMLX"
sed "s/__VERSION__/dev/g; s/io.github.dmlx.app/io.github.dmlx.app.dev/" "$ROOT/app/Resources/Info.plist" > "$APP/Contents/Info.plist"
[ -f "$ROOT/app/Resources/AppIcon.icns" ] && cp "$ROOT/app/Resources/AppIcon.icns" "$APP/Contents/Resources/"
codesign --force --sign - "$APP" >/dev/null 2>&1 || true
if [ -n "${DMLX_HARNESS:-}" ]; then "$APP/Contents/MacOS/dMLX"; else open -n "$APP"; fi
