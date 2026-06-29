#!/usr/bin/env bash
# =============================================================================
#  build_llama_xcframework.sh - build llama.cpp into Frameworks/llama.xcframework
#  for the llama.cpp / Bonsai (1-bit GGUF) engine path of this sample app.
#
#  Produces an iOS-only (device + simulator) xcframework with the Metal library
#  embedded (GGML_METAL_EMBED_LIBRARY=ON), so there's no loose .metallib to ship.
#
#  Builds from the PrismML fork (`prism` branch) by default — that's the source
#  the Bonsai model card points at. NOTE: Bonsai's Q1_0 kernels are ALSO in
#  mainline llama.cpp (PR #21273, merged), so `LLAMA_REPO`/`LLAMA_BRANCH` can be
#  pointed at ggml-org/llama.cpp:master with identical results.
#
#  Usage:
#    bash scripts/build_llama_xcframework.sh
#    LLAMA_REPO=https://github.com/ggml-org/llama.cpp.git LLAMA_BRANCH=master \
#      bash scripts/build_llama_xcframework.sh
#
#  Prereqs (macOS): Xcode 15+, cmake.
# =============================================================================
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: must run on macOS (Xcode + iOS SDK are macOS-only)."; exit 1
fi

LLAMA_REPO="${LLAMA_REPO:-https://github.com/PrismML-Eng/llama.cpp.git}"
LLAMA_BRANCH="${LLAMA_BRANCH:-prism}"

cd "$(dirname "$0")/.."
APP_ROOT="$(pwd)"
WORK="$APP_ROOT/.llama-build"
SRC="$WORK/llama.cpp"
DEST="$APP_ROOT/Frameworks/llama.xcframework"

mkdir -p "$WORK"

# ---- 1. Fetch source ------------------------------------------------------
if [[ ! -d "$SRC/.git" ]]; then
  echo "=== Cloning $LLAMA_REPO ($LLAMA_BRANCH)..."
  git clone --depth 1 -b "$LLAMA_BRANCH" "$LLAMA_REPO" "$SRC"
else
  echo "=== Reusing existing clone at $SRC"
fi

# Sanity: the Q1_0 type must be present (Bonsai needs it).
if ! grep -q "GGML_TYPE_Q1_0" "$SRC/ggml/include/ggml.h"; then
  echo "ERROR: this llama.cpp has no GGML_TYPE_Q1_0 — it cannot load Bonsai Q1_0."
  exit 1
fi

# ---- 2. Build the full (all-Apple-platform) xcframework -------------------
echo "=== Building llama.xcframework (this compiles several Apple platforms)..."
pushd "$SRC" >/dev/null
./build-xcframework.sh
popd >/dev/null

FULL="$SRC/build-apple/llama.xcframework"
[[ -d "$FULL" ]] || { echo "ERROR: $FULL not produced."; exit 1; }

# ---- 3. Trim to the iOS slices the app actually needs ---------------------
# The full xcframework also carries macOS/tvOS/visionOS slices + dSYMs (~750 MB).
# We only ship ios-arm64 (device) + ios-arm64_x86_64-simulator (~18 MB).
echo "=== Trimming to iOS-only -> $DEST"
mkdir -p "$APP_ROOT/Frameworks"
rm -rf "$DEST"
xcrun xcodebuild -create-xcframework \
  -framework "$FULL/ios-arm64/llama.framework" \
  -framework "$FULL/ios-arm64_x86_64-simulator/llama.framework" \
  -output "$DEST"

echo
echo "================================================================="
echo "Done. Frameworks/llama.xcframework ready ($(du -sh "$DEST" | cut -f1))."
echo "Next: drop a .gguf at Resources/model.gguf, then run scripts/bootstrap.sh"
echo "(or xcodegen generate) and build."
echo "================================================================="
