#!/usr/bin/env bash
# Build dist/dMLX.app: Swift app + a relocatable Python with the engine installed inside it.
#
#   tools/bundle_app.sh            # build the .app
#   tools/bundle_app.sh --zip      # also produce dist/dMLX-<version>-macos-arm64.zip
#
# Requirements: Xcode command line tools, uv. Signs ad-hoc (no Apple Developer ID needed).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$ROOT/dist"
APP="$DIST/dMLX.app"
PY_VERSION="3.13"
VERSION="$(grep -m1 '^version' "$ROOT/engine/pyproject.toml" | sed -E 's/version = "(.*)"/\1/')"
# Packages the engine never imports at runtime (mflux pulls them in for PyTorch-format checkpoints).
EXCLUDE='^(torch|sympy|networkx|mpmath|opencv-python|matplotlib|contourpy|cycler|kiwisolver|pyparsing|fonttools)=='

echo "==> Building Swift app (release)"
(cd "$ROOT/app" && swift build -c release --product dMLX)
BIN="$(cd "$ROOT/app" && swift build -c release --product dMLX --show-bin-path)/dMLX"

echo "==> Assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/engine"
cp "$BIN" "$APP/Contents/MacOS/dMLX"
sed "s/__VERSION__/$VERSION/g" "$ROOT/app/Resources/Info.plist" > "$APP/Contents/Info.plist"
if [ -f "$ROOT/app/Resources/AppIcon.icns" ]; then
  cp "$ROOT/app/Resources/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
fi

echo "==> Bundling Python $PY_VERSION"
uv python install "$PY_VERSION" >/dev/null
PY_SRC="$(uv python find "$PY_VERSION" --managed-python)"
PY_HOME="$(cd "$(dirname "$PY_SRC")/.." && pwd)"
cp -R "$PY_HOME" "$APP/Contents/Resources/engine/python"
PY="$APP/Contents/Resources/engine/python/bin/python3"
# Drop pieces a GUI-launched interpreter never needs.
rm -rf "$APP/Contents/Resources/engine/python/lib/python$PY_VERSION/test" \
       "$APP/Contents/Resources/engine/python/lib/python$PY_VERSION/idlelib" \
       "$APP/Contents/Resources/engine/python/lib/python$PY_VERSION/tkinter" \
       "$APP/Contents/Resources/engine/python/share"

echo "==> Installing engine dependencies (locked)"
REQ="$(mktemp)"
(cd "$ROOT/engine" && uv export --frozen --no-dev --no-hashes --no-emit-project --format requirements-txt) \
  | grep -v -E "$EXCLUDE" | grep -v '^\s*#' | grep -v '^-e' > "$REQ"
# mflux declares torch/opencv/matplotlib as hard deps; install it without deps, the rest from the lock.
grep -v '^mflux==' "$REQ" > "$REQ.rest"
uv pip install --python "$PY" --no-cache --break-system-packages -r "$REQ.rest" >/dev/null
uv pip install --python "$PY" --no-cache --break-system-packages --no-deps "$(grep '^mflux==' "$REQ")" >/dev/null
uv pip install --python "$PY" --no-cache --break-system-packages --no-deps "$ROOT/engine" >/dev/null
rm -f "$REQ" "$REQ.rest"

echo "==> Smoke-testing the bundled engine"
"$PY" -I -c "import dmlx, dmlx.api, dmlx.adapters.mflux_adapter, mflux.models.flux2.variants, mflux.models.z_image.variants.z_image; print('engine import ok', dmlx.__version__)"

echo "==> Cleaning bytecode caches"
find "$APP/Contents/Resources/engine" -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "==> Signing (ad-hoc)"
find "$APP/Contents/Resources/engine" \( -name "*.so" -o -name "*.dylib" \) -print0 | xargs -0 -n 50 codesign --force --sign - --timestamp=none 2>/dev/null
find "$APP/Contents/Resources/engine/python/bin" -type f -perm +111 -print0 | xargs -0 -n 50 codesign --force --sign - --timestamp=none 2>/dev/null || true
codesign --force --sign - --timestamp=none "$APP/Contents/MacOS/dMLX"
codesign --force --sign - --timestamp=none "$APP"
codesign --verify --verbose=1 "$APP" 2>&1 | tail -1

du -sh "$APP"

if [ "${1:-}" = "--zip" ]; then
  ZIP="$DIST/dMLX-$VERSION-macos-arm64.zip"
  rm -f "$ZIP"
  (cd "$DIST" && ditto -c -k --sequesterRsrc --keepParent dMLX.app "$ZIP")
  shasum -a 256 "$ZIP" | tee "$ZIP.sha256"
fi
echo "==> Done: $APP"
