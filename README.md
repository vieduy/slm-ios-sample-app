# SLM iOS Sample App

A minimal SwiftUI app that runs small language models **on-device** via two
swappable engines:

- **LiteRT-LM** (`GemmaEngine.xcframework` + `CLiteRTLM.xcframework`) — runs
  `.litertlm` models (e.g. Gemma-3 270M-IT).
- **llama.cpp** (`llama.xcframework`) — runs `.gguf` models, including the
  **1-bit Bonsai-1.7B (Q1_0, qwen3 arch)**.

The active engine is chosen automatically by which model file is in the bundle
(`.gguf` → llama.cpp, `.litertlm` → LiteRT-LM); see `InferenceEngine.swift`.
Both bridges expose the *same* selectors, so the SwiftUI/view-model layer is
engine-agnostic. Used as the reference integration target for the mobile team.

> **llama.cpp / Bonsai users: jump to [llama.cpp engine (Bonsai 1-bit GGUF)](#llamacpp-engine--bonsai-1-bit-gguf).**

```
[ direction picker  EN→VI | VI→EN ]
┌────────────────────────────────┐
│ Hello, how are you today?      │   ← input (cleared on direction switch)
└────────────────────────────────┘
            [ Translate ]
┌────────────────────────────────┐
│ Xin chào, hôm nay bạn thế nào? │   ← output
└────────────────────────────────┘
                850 ms · 13 ms/tok
```

---

## llama.cpp engine — Bonsai 1-bit (GGUF)

Runs `.gguf` models through **mainline-compatible llama.cpp** built into
`llama.xcframework` (Metal embedded). Validated with the **finetuned
Bonsai-1.7B Q1_0** (qwen3 architecture, ~231 MB on disk, ~1.125 bpw).

### Setup

```bash
cd gemma-3/inference/sample-ios-app

# 1. Build the llama.cpp xcframework (prism fork by default; mainline works too).
#    First run downloads + compiles llama.cpp for Apple platforms (~several min).
bash scripts/build_llama_xcframework.sh

# 2. Drop a GGUF as Resources/model.gguf  (gitignored — 100MB+).
cp /path/to/bonsai-1.7b-...-Q1_0.gguf Resources/model.gguf

# 3. Generate the project and build/run.
xcodegen generate
open Gemma3Translator.xcodeproj          # or: scripts/run_sim.sh
```

`bootstrap.sh` builds the LiteRT-LM frameworks; the llama.cpp framework is
built by the dedicated `build_llama_xcframework.sh` above. You only need the
engine matching the model you bundle.

### Key behavior (`LlamaBridge.mm`)

- **CPU is the recommended (and reliable) backend.** The bridge pins CPU mode
  to the `CPU` + `BLAS (Accelerate)` ggml devices and excludes Metal — this
  removes a one-time ~14 s Metal shader compile at launch and hundreds of
  per-token graph splits, while BLAS still accelerates prefill.
- **Thinking is disabled.** Bonsai/qwen3 are reasoning models whose template
  emits `<think>…</think>`. `llama_chat_apply_template` (no jinja) drops the
  template's empty-think block, so the bridge re-appends `<think>\n\n</think>`
  after the assistant header (the model's own no-think format).
- **Threads** = all cores (decode is compute-bound on the generic Q1_0 ARM
  kernel and scales with cores). `n_ctx` is capped (1024) to keep the KV cache
  small on memory-constrained devices.
- **Diagnostics**: at load it logs `[[CPUINFO]] DotProd=… I8MM=…` and
  `[[LLAMA]] reqBackend=… useGpu=…` to stderr (capture with
  `devicectl … --console`).

### ⚠️ GPU/Metal is unusable on older devices (e.g. A12 / iPhone XS)

On the iPhone XS the model **fully offloads to the A12 Metal GPU (29/29
layers) but produces incorrect output** — the A12 lacks `simdgroup
matmul`/`reduction`, which ggml-metal's matvec kernels need. Even setting that
aside, the A12 GPU wouldn't beat CPU for batch-1 decode (shared/unified memory
= no bandwidth edge). **Use CPU on the XS.** Newer GPUs (A17+/Apple Silicon)
run Metal correctly and fast. The CPU/GPU toggle remains for newer hardware.

### Performance reality (iPhone XS / A12)

| | CPU | notes |
|---|---|---|
| Decode | ~6 tok/s | hardware ceiling — see below |
| Prefill | ~21 tok/s | BLAS-accelerated |
| Peak RAM | ~0.5 GB | weights mmap'd + ~112 MiB KV @ n_ctx 1024 |

Decode is compute-bound on the generic Q1_0 kernel. The A12 lacks the `SDOT`
dot-product unit (added in the A13), so llama.cpp emulates it — but that
emulation costs only ~18% (measured), so the slowness is mostly the **old chip
overall**, not the missing instruction. **1-bit saves memory, not arithmetic.**
For interactive speed on an A12, use a smaller model; newer chips are far faster.

---

## File layout

```
sample-ios-app/
├── project.yml                          XcodeGen spec (source of truth for the project)
├── Gemma3Translator/
│   ├── Gemma3TranslatorApp.swift        @main entry point
│   ├── ContentView.swift                SwiftUI screen + direction picker
│   ├── TranslationViewModel.swift       @MainActor warm-up, generate on detached Task
│   ├── TranslationDirection.swift       .enToVi / .viToEn + short prompt template
│   ├── ResourceLookup.swift             Bundle.main path helper
│   ├── GemmaBridge.h / .mm              Obj-C++ shim over gemma::GemmaEngine (LiteRT-LM)
│   ├── LlamaBridge.h / .mm              Obj-C++ shim over llama.cpp (GGUF, same selectors)
│   ├── InferenceEngine.swift           protocol + runtime auto-select by model extension
│   ├── Gemma3Translator-Bridging-Header.h
│   └── Info.plist                       hardcoded (see "Implementation notes")
├── Resources/                           ← drop model.litertlm OR model.gguf here (gitignored)
├── Frameworks/                          ← populated by the build scripts
│   ├── GemmaEngine.xcframework          (LiteRT-LM path)
│   ├── CLiteRTLM.xcframework            (LiteRT-LM path)
│   └── llama.xcframework                (llama.cpp / GGUF path)
└── scripts/
    ├── bootstrap.sh                     fetch CLiteRTLM, build GemmaEngine, copy model, xcodegen
    ├── build_llama_xcframework.sh       build llama.xcframework (llama.cpp / Bonsai path)
    └── run_sim.sh                       boot sim, build, install, launch
```

---

## Prerequisites

- macOS with **Xcode 15+** (open it once after install to unpack the SDK)
- **XcodeGen** and optionally xcbeautify:
  ```bash
  brew install xcodegen xcbeautify
  ```
- `model.litertlm` available somewhere on disk — see below

---

## Resource files

This single file is gitignored and must be supplied before every build.
Drop it into `sample-ios-app/Resources/` (or let `bootstrap.sh` copy it
from `gemma-3/model/`):

| File | Source | Size |
|------|--------|------|
| `model.litertlm` | `gemma-3/model/model.litertlm` (produced by `export/export_gemma3_270m.py`) | ~285 MB |

**No tokenizer file, no chat-template file** — both are stored inside
`.litertlm` and applied by LiteRT-LM automatically.

The `bootstrap.sh` script auto-copies `model.litertlm` from
`gemma-3/model/` if it exists. If you have the file elsewhere, copy it in
manually:

```bash
cp /path/to/model.litertlm Resources/
```

---

## Quick start

```bash
cd gemma-3/inference/sample-ios-app

# 1. (Optional) drop model.litertlm if you don't have it at gemma-3/model/
cp /path/to/model.litertlm Resources/

# 2. Bootstrap (first time, or after engine source changes)
bash scripts/bootstrap.sh
#   first run:  ~3-5 min (downloads CLiteRTLM + builds GemmaEngine for both slices)
#   subsequent: ~30 s

# 3a. Launch on simulator from terminal
bash scripts/run_sim.sh                   # default: iPhone 15
SIM_NAME="iPhone 16 Pro" bash scripts/run_sim.sh

# 3b. — or — open in Xcode
open Gemma3Translator.xcodeproj
```

---

## How it works

### Startup

`TranslationViewModel.warmUpIfNeeded()` runs on first view appearance:

1. Locates `model.litertlm` in the bundle via `ResourceLookup`.
2. Resolves `~/Library/Caches/gemma3/` as the **writable cache dir** (the
   `.app` bundle is read-only — without a writable cache dir, LiteRT-LM
   silently fails to persist its XNNPACK kernel cache and you pay
   ~1–2 s of warm-up on every cold launch).
3. Calls `GemmaBridge.load(modelPath:backend:cacheDir:)` on a detached
   Task — mmaps the model, warms up the engine.

Total cold-start time: ~1.7 s on a recent Mac simulator. Subsequent
launches re-use the cache.

### Translation

1. View model builds a deliberately **short** prompt: `"Translate to
   Vietnamese: <input>"` (5 tokens of instruction, not 30). Every
   instruction token costs prefill compute per translate.
2. `GemmaBridge.generate(prompt:maxTokens:temperature:topK:topP:)`
   forwards to `gemma::GemmaEngine::Generate`, which recreates the
   LiteRT-LM conversation internally and runs the full prefill + decode
   loop on a background dispatch queue.
3. LiteRT-LM applies the **Gemma chat template** automatically, runs the
   sampler, and streams tokens back. The full reply text is returned at
   the end (no UI-side streaming in this sample — `GemmaEngine` exposes
   `GenerateStream` if you want token-by-token).
4. The view model trims whitespace and writes the result to `vm.output`.
   Status footer shows total ms.

### Direction switching

Switching EN↔VI clears both the input and output fields to prevent sending
text in the wrong source language. There is **no need to call
`engine.resetSession()`** between translations: `Generate()` already
recreates the conversation per call (`gemma_engine.cpp:161`).

---

## Implementation notes

### LiteRT-LM owns the hard parts

Unlike the matmoe sample, **nothing about tokenization, prompt templating,
KV cache, or sampling lives in app code**. All of that runs inside
`libLiteRtLmCApi.dylib`. The Swift / Obj-C surface is intentionally three
calls: `load`, `generate`, `lastError`. If you find yourself reaching for
HuggingFace tokenizers or chat-template libraries in Swift, you're working
against the integration.

### Two dynamic frameworks, both must Embed & Sign

`GemmaEngine.xcframework` contains `libgemma_engine.dylib` (our wrapper).
It is dynamically linked against `CLiteRTLM.framework` (LiteRT-LM C API).
**Both** xcframeworks must be embedded into the `.app/Frameworks/` and
signed; otherwise dyld fails to resolve `@rpath/CLiteRTLM.framework/...`
at launch.

The CMake build sets `INSTALL_NAME_DIR=@rpath` and adds
`@executable_path/Frameworks` + `@loader_path/Frameworks` rpaths so dyld
can find the sibling framework from inside the app bundle.

### The writable `cache_dir` argument matters

`Init(modelPath, backend, cacheDir)` accepts a third optional argument
that LiteRT-LM uses for its XNNPACK kernel cache (`model.litertlm.xnnpack_cache_*`
on the desktop reference). On iOS the `.app` bundle is read-only so the
default location (next to the model) fails. The sample uses
`FileManager.default.urls(for: .cachesDirectory).appendingPathComponent("gemma3")`.

Skip this and every cold launch rebuilds the kernel selection plan —
visible as a ~1–2 s tax on top of the ~0.5 s mmap.

### Short prompts beat verbose prompts

Gemma-3-IT follows free-form instructions, so a verbose preamble works —
but every preamble token gets prefilled before generation starts. The
sample uses `"Translate to Vietnamese: <text>"` (~5 tokens) rather than
`"You are a translation assistant. Translate the following English text
to Vietnamese. Reply with the translation only..."` (~30 tokens). Real
wall-clock difference per translate on simulator.

### iOS 26 simulator install footguns

The `project.yml` carries four non-obvious settings learned from the
matmoe sample app — keep them when forking:

| Setting | Why |
|---|---|
| `INFOPLIST_FILE` hardcoded, `GENERATE_INFOPLIST_FILE: NO` | Xcode 26.3 with both options on drops `CFBundleIdentifier` from the merged plist → installer rejects with IXErrorDomain 13 "Missing bundle ID". |
| `Resources/` as a Group (not `type: folder`) | Folder references ship as opaque blue folders that Xcode's signing step doesn't enumerate → `codesign` shows `Info.plist=not bound` → same "Missing bundle ID" error. |
| `ENABLE_DEBUG_DYLIB: NO` | Xcode 26's Debug host + `.debug.dylib` split confuses the iOS 26 sim installer. |
| `LD_RUNPATH_SEARCH_PATHS` includes `@executable_path/Frameworks` | Required for the two embedded dynamic xcframeworks (see above). |

### Build-script defensive bits

`scripts/build_ios.sh` (in `gemma-3/inference/scripts/`) is per-arch, not
multi-arch:

- One CMake configure per arch because `CMAKE_OSX_ARCHITECTURES="arm64;x86_64"`
  put a literal `;` in the build dir name on Xcode 26 (CMake parses it as
  a list and the file-write fails).
- `lipo` after the two sim builds when `SIM_X86_64=1`.
- Passes `CODE_SIGNING_ALLOWED=NO` because the intermediate `.dylib` is
  re-signed when `xcodebuild -create-xcframework` and again at app embed
  time. Xcode 26 otherwise fails the CMake build with "requires a
  development team."

`scripts/bootstrap.sh` always re-invokes `build_ios.sh` rather than
skipping when `dist/` exists — the previous shortcut shipped stale
headers when the engine sources changed.

---

## Porting to another app

Copy from `Gemma3Translator/`:

| File | Purpose |
|------|---------|
| `GemmaBridge.h` / `.mm` | C++ → Obj-C++ bridge (3 entry points + `lastError`) |
| `Gemma3Translator-Bridging-Header.h` | exposes bridge to Swift |
| `ResourceLookup.swift` | bundle-path helper that handles both flat and folder-grouped resources |
| `TranslationViewModel.swift` | minimal `@MainActor` async warm-up + generate pattern |

Xcode project wiring required:

1. **Embed & Sign** both `Frameworks/GemmaEngine.xcframework` and
   `Frameworks/CLiteRTLM.xcframework` (both are dynamic).
2. Set `SWIFT_OBJC_BRIDGING_HEADER` to the bridging header.
3. Add `Frameworks/GemmaEngine.xcframework/Headers/`,
   `.../ios-arm64/Headers/`,
   `.../ios-arm64-simulator/Headers/`, and
   `.../ios-arm64_x86_64-simulator/Headers/` to `HEADER_SEARCH_PATHS`.
4. Add `-lc++` to `OTHER_LDFLAGS`.
5. Add `@executable_path/Frameworks` and `@loader_path/Frameworks` to
   `LD_RUNPATH_SEARCH_PATHS`.
6. Bundle `model.litertlm` as a resource (group, not folder reference).
7. Apply the four iOS 26 footgun mitigations from the table above.

**No Swift Package needed** (unlike the matmoe sample, which pulls in
huggingface/swift-transformers for tokenization).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "model.litertlm not in bundle" | File missing from `Resources/` | Copy it in (or run `bootstrap.sh` with `gemma-3/model/model.litertlm` populated), rebuild |
| `Missing bundle ID` install error | `project.yml` regressed: `GENERATE_INFOPLIST_FILE: YES`, or `Resources/` became `type: folder`, or `ENABLE_DEBUG_DYLIB` left on | Reapply the four `project.yml` settings in the table above |
| `dyld: Library not loaded: @rpath/CLiteRTLM.framework/CLiteRTLM` | One of the two xcframeworks didn't embed | Set **both** to Embed & Sign in target settings; rerun `bootstrap.sh` |
| `Signing for "gemma_engine" requires a development team` during `bootstrap.sh` | Xcode 26 wants every iOS target signed | Already fixed in `scripts/build_ios.sh` via `CODE_SIGNING_ALLOWED=NO`. If you forked the script, restore those flags |
| `CMake Error: file failed to open for writing (Is a directory)` during iOS build | Build dir contained literal `;` from multi-arch | Already fixed: per-arch builds + optional lipo. Don't pass `arm64;x86_64` to a single CMake configure |
| `Missing package product 'Tokenizers'` | Stale SPM resolution from a different project | `bootstrap.sh` and `run_sim.sh` both run `xcodebuild -resolvePackageDependencies` — re-run them. (The gemma sample has no SPM deps today; this is leftover defensiveness.) |
| Slow cold launch (~1-2 s on top of the ~0.5 s mmap) | XNNPACK kernel cache being rebuilt | Confirm `cacheDir:` is being passed to `bridge.load(...)` and the resulting dir is writable |
| Output is gibberish / repetitive | The chat template was bypassed | Make sure you go through `Generate` / `GenerateStream`, not raw LiteRT-LM session APIs |
| VI→EN outputs Vietnamese (or vice versa) | Prompt prefix not flipped | Confirm `TranslationDirection.promptFor` is being called with the right direction |
| "Engine not ready" on Translate tap | Warm-up failed silently | Check `errorMessage` label at bottom of screen — usually a missing model file or read-only cache dir |
| Slow translation on simulator | Sim has no Neural Engine; XNNPACK paths slower than on device | Expected. ~3x slower per token than matmoe dim-256 is normal given gemma-3-270m is dense vs. matmoe's sparse-MoE. Test on real device for representative numbers. |

---

## See also

- Engine API + build instructions: [`../README.md`](../README.md) (`gemma-3/inference/`)
- Top-level integration doc for the mobile team: [`../../README.md`](../../README.md) (`gemma-3/`)
- Sibling matmoe sample app for comparison: [`../../../matmoe-inference-c/sample-ios-app/`](../../../matmoe-inference-c/sample-ios-app/)
- Python parity reference: [`../../export/verify_litertlm.py`](../../export/verify_litertlm.py)
- LiteRT-LM upstream: https://github.com/google-ai-edge/LiteRT-LM
