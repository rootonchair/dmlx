#!/usr/bin/env bash
# Pack a built dMLX.app into a disk image with an Applications shortcut, the usual Mac install window.
#
#   tools/make_dmg.sh dist/dMLX.app dist/dMLX-<version>-macos-arm64.dmg
#
# Writes the .dmg and a .sha256 file next to it. Uses hdiutil, which comes with macOS.
set -euo pipefail

APP="${1:?usage: make_dmg.sh <dMLX.app> <out.dmg>}"
OUT="${2:?usage: make_dmg.sh <dMLX.app> <out.dmg>}"
[ -d "$APP" ] || { echo "not an app bundle: $APP" >&2; exit 1; }

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
ditto "$APP" "$STAGE/dMLX.app"
ln -s /Applications "$STAGE/Applications"

rm -f "$OUT"
# ULFO: LZFSE-compressed, read-only; opens on every macOS that dMLX supports (15 and later).
hdiutil create -volname "dMLX" -srcfolder "$STAGE" -fs HFS+ -format ULFO -ov "$OUT" >/dev/null
hdiutil verify "$OUT" >/dev/null
(cd "$(dirname "$OUT")" && shasum -a 256 "$(basename "$OUT")" | tee "$(basename "$OUT").sha256")
