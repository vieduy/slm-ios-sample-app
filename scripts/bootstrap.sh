#!/usr/bin/env bash
# =============================================================================
#  bootstrap.sh - one-shot setup for the Gemma3Translator sample iOS app.
#
#  Lives at gemma-3/inference/sample-ios-app/scripts/bootstrap.sh.
#
#  1. Fetches LiteRT-LM prebuilt xcframework (CLiteRTLM.xcframework).
#  2. Builds gemma-3/inference into GemmaEngine.xcframework.
#  3. Copies both xcframeworks into sample-ios-app/Frameworks/.
#  4. Copies gemma-3/model/model.litertlm into Resources/ if present.
#  5. Runs xcodegen to produce Gemma3Translator.xcodeproj.
#  6. Pre-resolves SPM dependencies (there are none today, but harmless).
#
#  Re-run any time you change project.yml or rebuild the engine.
#
#  Prereqs (macOS):
#    - Xcode 15+
#    - brew install xcodegen
# =============================================================================
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: This script must run on macOS (Xcode + iOS SDK are macOS-only)."
  exit 1
fi

cd "$(dirname "$0")/.."
APP_ROOT="$(pwd)"
# We sit at gemma-3/inference/sample-ios-app, so the inference root is one up
# and the gemma-3 root is two up.
INFERENCE="$(cd .. && pwd)"
GEMMA3="$(cd ../.. && pwd)"

DIST_XCFW="$INFERENCE/dist/GemmaEngine.xcframework"
LITERT_XCFW="$INFERENCE/third_party/litert_lm/lib/ios/CLiteRTLM.xcframework"

APP_GEMMA="$APP_ROOT/Frameworks/GemmaEngine.xcframework"
APP_LITERT="$APP_ROOT/Frameworks/CLiteRTLM.xcframework"
APP_MODEL="$APP_ROOT/Resources/model.litertlm"
SRC_MODEL="$GEMMA3/model/model.litertlm"

if [[ ! -f "$INFERENCE/CMakeLists.txt" ]]; then
  echo "ERROR: $INFERENCE doesn't look like gemma-3/inference (no CMakeLists.txt)."
  exit 1
fi

# ---- 1. Fetch LiteRT-LM prebuilts ----------------------------------------
if [[ ! -d "$LITERT_XCFW" ]]; then
  echo "=== Fetching LiteRT-LM prebuilt xcframework..."
  pushd "$INFERENCE" >/dev/null
  ./scripts/fetch_prebuilt.sh
  popd >/dev/null
fi
if [[ ! -d "$LITERT_XCFW" ]]; then
  echo "ERROR: $LITERT_XCFW still missing after fetch_prebuilt.sh."
  exit 1
fi

# ---- 2. Build (or rebuild) GemmaEngine.xcframework -----------------------
# Always invoke build_ios.sh; CMake handles up-to-date checks itself.
echo "=== Building (or refreshing) $DIST_XCFW..."
pushd "$INFERENCE" >/dev/null
./scripts/build_ios.sh
popd >/dev/null

# ---- 3. Sync xcframeworks into the app -----------------------------------
mkdir -p "$APP_ROOT/Frameworks"
rm -rf "$APP_GEMMA" "$APP_LITERT"
cp -R "$DIST_XCFW"   "$APP_GEMMA"
cp -R "$LITERT_XCFW" "$APP_LITERT"
echo "=== Copied xcframeworks -> $APP_ROOT/Frameworks/"

# ---- 4. Copy model.litertlm into Resources/ ------------------------------
mkdir -p "$APP_ROOT/Resources"
if [[ -f "$SRC_MODEL" ]]; then
  if [[ ! -f "$APP_MODEL" ]] || ! cmp -s "$SRC_MODEL" "$APP_MODEL"; then
    echo "=== Copying model.litertlm into Resources/ (~285 MB)..."
    cp "$SRC_MODEL" "$APP_MODEL"
  else
    echo "=== Resources/model.litertlm already up to date."
  fi
else
  echo
  echo "WARNING: $SRC_MODEL not found."
  echo "         The app will fail to load the model on launch until you drop"
  echo "         a model.litertlm into $APP_ROOT/Resources/."
fi

# ---- 5. Generate Xcode project -------------------------------------------
if ! command -v xcodegen >/dev/null; then
  echo "ERROR: xcodegen not found. Install with: brew install xcodegen"
  exit 1
fi
echo "=== Running xcodegen..."
xcodegen generate

# ---- 6. Resolve SPM packages now -----------------------------------------
# No SPM deps today, but keeping this step in case we add Tokenizers later.
if [[ -d "Gemma3Translator.xcodeproj" ]]; then
  echo "=== Resolving SPM dependencies..."
  xcodebuild -resolvePackageDependencies \
    -project Gemma3Translator.xcodeproj \
    -scheme Gemma3Translator >/dev/null 2>&1 || true
fi

echo
echo "================================================================="
echo "Done."
echo "  Open:  open Gemma3Translator.xcodeproj"
echo "  Run in simulator (one-liner):  scripts/run_sim.sh"
echo "================================================================="
