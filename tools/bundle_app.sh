#!/usr/bin/env bash
# Build dist/dMLX.app: Swift app + a relocatable Python with the engine installed inside it.
#
#   tools/bundle_app.sh            # build the .app
#   tools/bundle_app.sh --dmg      # also produce dist/dMLX-<version>-macos-arm64.dmg (the website's download)
#   tools/bundle_app.sh --zip      # also produce dist/dMLX-<version>-macos-arm64.zip
#   tools/bundle_app.sh --dmg --zip
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
# uv's versioned folders are symlinks; resolve to the real install and copy its *contents*.
PY_HOME="$(cd -P "$(dirname "$PY_SRC")/.." && pwd -P)"
mkdir -p "$APP/Contents/Resources/engine/python"
cp -R "$PY_HOME/." "$APP/Contents/Resources/engine/python/"
PY="$APP/Contents/Resources/engine/python/bin/python3"
# Safety: everything below modifies the bundled copy only. Refuse if it is not a real directory
# inside dist/ (a symlink here would point back at the user's own Python).
BUNDLED_PY_HOME="$(cd -P "$APP/Contents/Resources/engine/python" && pwd -P)"
case "$BUNDLED_PY_HOME" in
  "$DIST"/*) ;;
  *) echo "refusing: bundled Python resolves outside dist/ ($BUNDLED_PY_HOME)" >&2; exit 1 ;;
esac
PY_REAL="$(cd -P "$(dirname "$PY")" && pwd -P)/$(basename "$(readlink "$PY" || basename "$PY")")"
case "$PY_REAL" in
  "$BUNDLED_PY_HOME"/*) ;;
  *) echo "refusing: bundled interpreter resolves outside the bundle ($PY_REAL)" >&2; exit 1 ;;
esac
# Drop pieces a GUI-launched interpreter never needs.
rm -rf "$APP/Contents/Resources/engine/python/lib/python$PY_VERSION/test" \
       "$APP/Contents/Resources/engine/python/lib/python$PY_VERSION/idlelib" \
       "$APP/Contents/Resources/engine/python/lib/python$PY_VERSION/tkinter" \
       "$APP/Contents/Resources/engine/python/share"

echo "==> Installing engine dependencies (locked)"
REQ="$(mktemp)"
(cd "$ROOT/engine" && uv export --frozen --no-dev --no-hashes --no-emit-project --extra local-llm --format requirements-txt) \
  | grep -v -E "$EXCLUDE" | grep -v '^\s*#' | grep -v '^-e' > "$REQ"
# local-llm brings mlx-vlm, the planning agent's runtime.
# mflux and mlx-vlm declare torch/opencv/matplotlib as hard deps; install exactly the lock without them
# (the export already lists every dependency), then mflux without deps.
grep -v -E '^mflux ?(==|@)' "$REQ" > "$REQ.rest"
uv pip install --python "$PY" --no-cache --break-system-packages --no-deps -r "$REQ.rest" >/dev/null
uv pip install --python "$PY" --no-cache --break-system-packages --no-deps "$(grep -E '^mflux ?(==|@)' "$REQ")" >/dev/null
uv pip install --python "$PY" --no-cache --break-system-packages --no-deps "$ROOT/engine" >/dev/null
rm -f "$REQ" "$REQ.rest"

echo "==> Smoke-testing the bundled engine"
"$PY" -I -c "import dmlx, dmlx.api, dmlx.adapters.mflux_adapter, mflux.models.flux2.variants, mflux.models.z_image.variants.z_image, mlx_lm, mlx_vlm; print('engine import ok', dmlx.__version__)"

echo "==> Cleaning bytecode caches"
find "$APP/Contents/Resources/engine" -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "==> Signing (ad-hoc)"
find "$APP/Contents/Resources/engine" \( -name "*.so" -o -name "*.dylib" \) -print0 | xargs -0 -n 50 codesign --force --sign - --timestamp=none 2>/dev/null
find "$APP/Contents/Resources/engine/python/bin" -type f -perm +111 -print0 | xargs -0 -n 50 codesign --force --sign - --timestamp=none 2>/dev/null || true
codesign --force --sign - --timestamp=none "$APP/Contents/MacOS/dMLX"
codesign --force --sign - --timestamp=none "$APP"
codesign --verify --verbose=1 "$APP" 2>&1 | tail -1

du -sh "$APP"

for arg in "$@"; do
  case "$arg" in
    --zip)
      ZIP="$DIST/dMLX-$VERSION-macos-arm64.zip"
      rm -f "$ZIP"
      (cd "$DIST" && ditto -c -k --sequesterRsrc --keepParent dMLX.app "$ZIP")
      (cd "$DIST" && shasum -a 256 "$(basename "$ZIP")" | tee "$(basename "$ZIP").sha256")
      ;;
    --dmg)
      "$ROOT/tools/make_dmg.sh" "$APP" "$DIST/dMLX-$VERSION-macos-arm64.dmg"
      ;;
  esac
done
echo "==> Done: $APP"
