import Foundation
import SwiftUI

/// Selectable LoRA adapter for the llama.cpp engine. `.base` = no adapter (pure
/// base model); the others map to bundled `<resource>.gguf` files.
enum AdapterChoice: String, CaseIterable, Identifiable {
    case base, adapter1, adapter2

    var id: String { rawValue }

    var label: String {
        switch self {
        case .base:     return "Base"
        case .adapter1: return "Adapter 1"
        case .adapter2: return "Adapter 2"
        }
    }

    /// Bundle resource basename (.gguf), or nil for the base model. Doubles as
    /// the engine-side adapter identifier.
    var resourceName: String? {
        switch self {
        case .base:     return nil
        case .adapter1: return "adapter_1"
        case .adapter2: return "adapter_2"
        }
    }
}

@MainActor
final class TranslationViewModel: ObservableObject {

    @Published var direction: TranslationDirection = .enToVi
    // Both task adapters are download-on-demand (nothing pre-loaded). A "Translate
    // N" button is enabled only once its adapter has been downloaded + hot-loaded.
    @Published var adapter: AdapterChoice = .adapter1        // last-activated adapter
    // Only-active-resident policy: files persist on disk once downloaded, but at
    // most ONE adapter's weights are held in RAM at a time.
    @Published var downloadedAdapters: Set<AdapterChoice> = []  // present on disk (gates Translate)
    @Published var residentAdapters: Set<AdapterChoice> = []    // adapters currently loaded in RAM
    @Published var downloading: AdapterChoice? = nil            // in-flight download, if any
    @Published var output: String = ""
    @Published var isRunning: Bool = false
    @Published var isReady: Bool = false
    @Published var errorMessage: String?
    @Published var timing: String?

    // Benchmark state.
    // CPU is the faster backend for this 270m model on A12 (iPhone XS): Metal
    // there is memory-bound at batch-1 decode and incurs heavy graph-split +
    // kernel-dispatch overhead, measured slower than CPU+BLAS. GPU stays
    // available via the toggle for measurement, but CPU is the default.
    @Published var backend: GemmaBackend = .cpu
    @Published var benchmarkMode: BenchmarkMode = .short
    @Published var isBenchmarking: Bool = false
    @Published var benchmarkResult: BenchmarkResult?

    private var engine: (any InferenceEngine)?
    private var activeBackend: GemmaBackend = .cpu

    // Resolved once in warmUpIfNeeded(), reused on every backend reload.
    private var runtime: ModelRuntime = .liteRT
    private var modelPath: String?
    private var baseCacheDir: String?

    private let maxNewTokens: Int32 = 128
    // Deterministic decoding: temp 0 makes LlamaBridge use the greedy (argmax)
    // sampler, so the same prompt + adapter always yields the same output —
    // essential for comparing adapters. top_k=1 / top_p=1 keep it deterministic
    // even if temperature is ever raised (top_k=1 = always pick the top token).
    private let temperature: Float = 0.0
    private let topK: Int32 = 1
    private let topP: Float = 1.0

    // Benchmark uses a generation prompt (not translation) so decode actually
    // runs to the mode's token cap — a translation prompt emits end-of-turn
    // after ~30 tokens and never stresses long context. Token cap / iteration
    // count come from the selected BenchmarkMode.
    private let benchmarkPrompt =
        "Write a long, vivid, multi-paragraph description of a busy morning " +
        "market in a coastal town. Describe the sights, sounds, smells, the " +
        "vendors, the food, and the people. Keep going with rich detail."
    // Small discarded warm-up so the first prefill / kernel-cache build doesn't
    // skew the first measured iteration.
    private let benchmarkWarmupTokens: Int32 = 16

    func warmUpIfNeeded() async {
        if isReady { return }
        guard let resolved = ModelRuntime.resolve() else {
            errorMessage = "No model in bundle — drop model.gguf or model.litertlm into Resources/."
            return
        }
        self.runtime = resolved.runtime
        self.modelPath = resolved.path
        // Writable kernel-cache dir. Without this LiteRT-LM tries to write the
        // XNNPACK cache next to the model — which lives in the read-only .app
        // bundle on iOS, so every cold launch rebuilds it (~1-2 s wasted).
        let caches = FileManager.default.urls(for: .cachesDirectory,
                                              in: .userDomainMask).first
        self.baseCacheDir = caches?.appendingPathComponent("gemma3").path

        await reload(to: backend)
    }

    /// (Re)load the engine on `requested`. If GPU is requested but fails to
    /// initialise (common on older GPUs / unsupported models), fall back to CPU
    /// and surface that in `activeBackend` + `timing`. Safe to call repeatedly:
    /// the previous engine is dropped before the new one loads.
    func reload(to requested: GemmaBackend) async {
        guard let modelPath else { return }
        isReady = false
        engine = nil
        errorMessage = nil
        benchmarkResult = nil
        output = ""
        timing = "Loading model on \(requested.label)…"

        let baseCacheDir = self.baseCacheDir
        let runtime = self.runtime
        let t0 = Date()
        let outcome: (any InferenceEngine, GemmaBackend, Bool)? =
            await Task.detached(priority: .userInitiated) {
                if let b = Self.tryLoad(modelPath: modelPath,
                                        runtime: runtime,
                                        backend: requested,
                                        baseCacheDir: baseCacheDir) {
                    return (b, requested, false)
                }
                // GPU couldn't load — retry on CPU so the app stays usable.
                if requested == .gpu,
                   let b = Self.tryLoad(modelPath: modelPath,
                                        runtime: runtime,
                                        backend: .cpu,
                                        baseCacheDir: baseCacheDir) {
                    return (b, .cpu, true)
                }
                return nil
            }.value

        guard let (bridge, active, fellBack) = outcome else {
            errorMessage = "Failed to load model on \(requested.label)."
            timing = nil
            return
        }
        self.engine = bridge
        self.activeBackend = active
        // Keep the published toggle in sync with reality on a GPU→CPU fallback.
        if fellBack { self.backend = active }
        // Adapters are tied to the model instance, so (re)load them onto the
        // fresh engine and re-apply the current selection after every reload.
        self.installAdapters()
        self.isReady = true
        let ms = Int(Date().timeIntervalSince(t0) * 1000)
        self.timing = fellBack
            ? "GPU unavailable — loaded on CPU in \(ms) ms"
            : "Loaded on \(active.label) in \(ms) ms"
    }

    /// Whether the loaded runtime supports dynamic LoRA (llama.cpp only).
    var supportsAdapters: Bool { runtime == .llamaCpp }

    /// A Translate button is enabled once its adapter file is on disk
    /// (downloaded). The actual RAM load happens lazily, on first use.
    func isAdapterReady(_ choice: AdapterChoice) -> Bool { downloadedAdapters.contains(choice) }

    /// Writable Documents path the "downloaded" adapter lives at (the app bundle
    /// is read-only, so downloads must land here).
    private func adapterURL(for choice: AdapterChoice) -> URL? {
        guard let name = choice.resourceName else { return nil }
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            .first?.appendingPathComponent("\(name).gguf")
    }

    /// Re-scan state after every engine (re)load. Files persist on disk; RAM
    /// handles do NOT (they're tied to the model instance and die with it), so
    /// mark nothing resident — the next Translate re-loads on demand.
    private func installAdapters() {
        guard runtime == .llamaCpp else { downloadedAdapters = []; residentAdapters = []; return }
        var onDisk: Set<AdapterChoice> = []
        for choice in [AdapterChoice.adapter1, .adapter2] {
            if let url = adapterURL(for: choice),
               FileManager.default.fileExists(atPath: url.path) { onDisk.insert(choice) }
        }
        downloadedAdapters = onDisk
        residentAdapters = []
    }

    /// "Download" an adapter to disk (Documents) — does NOT load it into RAM.
    /// This is the point of the only-active-resident policy: downloaded ≠ in RAM.
    /// (Real apps URLSession-fetch the .gguf; here we copy the bundled file.)
    func downloadAdapter(_ choice: AdapterChoice) async {
        guard runtime == .llamaCpp,
              let name = choice.resourceName,
              let dest = adapterURL(for: choice) else { return }
        guard downloading == nil, !downloadedAdapters.contains(choice) else { return }
        downloading = choice
        errorMessage = nil
        timing = "Downloading \(choice.label)…"

        let copied: Bool = await Task.detached(priority: .userInitiated) {
            guard let src = ResourceLookup.path(name, ext: "gguf") else { return false }
            try? FileManager.default.removeItem(at: dest)
            do { try FileManager.default.copyItem(atPath: src, toPath: dest.path); return true }
            catch { return false }
        }.value
        downloading = nil
        if copied {
            downloadedAdapters.insert(choice)
            let fp = Double(MemoryFootprint.currentBytes() ?? 0) / 1_048_576
            timing = String(format: "%@ downloaded (on disk, 0 MB RAM) · footprint %.0f MB",
                            choice.label, fp)
        } else {
            errorMessage = "\(choice.label) download (copy) failed."
            timing = nil
        }
    }

    /// Make `choice` the SINGLE resident adapter: free every OTHER resident
    /// adapter, then load `choice`. Reports the freed / loaded MB so you can see
    /// only one adapter's weights (~30 MB) live at a time. Returns false on error.
    private func makeResident(_ choice: AdapterChoice) -> Bool {
        guard let engine, let name = choice.resourceName, let url = adapterURL(for: choice) else { return false }
        if residentAdapters == [choice] { return true }   // already the only resident one

        let before = MemoryFootprint.currentBytes() ?? 0
        for prev in residentAdapters where prev != choice {
            if let prevName = prev.resourceName { _ = engine.unloadAdapter(prevName) }  // free ~30 MB each
        }
        residentAdapters = residentAdapters.filter { $0 == choice }
        let afterFree = MemoryFootprint.currentBytes() ?? 0

        var loadMs = 0.0   // the switch latency: cost of re-loading from disk
        if !residentAdapters.contains(choice) {
            let t0 = Date()
            guard FileManager.default.fileExists(atPath: url.path),
                  engine.loadAdapter(path: url.path, identifier: name) else {
                errorMessage = "Load failed: \(engine.lastError)"
                return false
            }
            loadMs = Date().timeIntervalSince(t0) * 1000
            residentAdapters.insert(choice)
        }
        let afterLoad = MemoryFootprint.currentBytes() ?? 0

        let freedMB  = (Double(before)    - Double(afterFree)) / 1_048_576
        let loadedMB = (Double(afterLoad) - Double(afterFree)) / 1_048_576
        let totalMB  =  Double(afterLoad) / 1_048_576
        FileHandle.standardError.write(Data(String(format:
            "[[LORA-RAM]] resident→%@  freed=%.0fMB loaded=%.0fMB loadLatency=%.0fms  footprint=%.0fMB\n",
            name, max(0, freedMB), max(0, loadedMB), loadMs, totalMB).utf8))
        timing = String(format: "RAM: freed %.0f MB, loaded %@ %.0f MB in %.0f ms · footprint %.0f MB (1 resident)",
                        max(0, freedMB), choice.label, max(0, loadedMB), loadMs, totalMB)
        return true
    }

    /// Diagnostic: load BOTH downloaded adapters into RAM at once (keep-all-
    /// resident), so you can compare footprint against the only-active policy.
    /// Expect ~2× the single-adapter RAM. Adapters must be downloaded first.
    func loadBothAdapters() async {
        guard runtime == .llamaCpp, let engine else { return }
        let targets = [AdapterChoice.adapter1, .adapter2].filter { downloadedAdapters.contains($0) }
        guard !targets.isEmpty else {
            errorMessage = "Download the adapters first."
            return
        }
        let before = MemoryFootprint.currentBytes() ?? 0
        for choice in targets where !residentAdapters.contains(choice) {
            guard let name = choice.resourceName, let url = adapterURL(for: choice),
                  FileManager.default.fileExists(atPath: url.path),
                  engine.loadAdapter(path: url.path, identifier: name) else {
                errorMessage = "Load failed: \(engine.lastError)"
                continue
            }
            residentAdapters.insert(choice)
        }
        let after = MemoryFootprint.currentBytes() ?? 0
        let loadedMB = (Double(after) - Double(before)) / 1_048_576
        let totalMB  =  Double(after) / 1_048_576
        FileHandle.standardError.write(Data(String(format:
            "[[LORA-RAM]] loadBoth resident=%d  loaded=%.0fMB  footprint=%.0fMB\n",
            residentAdapters.count, max(0, loadedMB), totalMB).utf8))
        timing = String(format: "%d adapters resident · +%.0f MB · footprint %.0f MB",
                        residentAdapters.count, max(0, loadedMB), totalMB)
    }

    /// Per-adapter Translate: ensure `choice` is the resident adapter (loading it
    /// and freeing the other), activate it, then translate.
    func translate(using choice: AdapterChoice, _ text: String) async {
        guard runtime == .llamaCpp, let engine, downloadedAdapters.contains(choice) else { return }
        guard makeResident(choice) else { return }
        guard engine.setActiveAdapter(choice.resourceName, scale: 1.0) else {
            errorMessage = engine.lastError
            return
        }
        adapter = choice
        await translate(text)
    }

    /// Build a backend-specific GemmaBridge, or nil if Init fails. The kernel
    /// cache is namespaced per backend so a CPU cache never feeds a GPU load.
    private nonisolated static func tryLoad(modelPath: String,
                                            runtime: ModelRuntime,
                                            backend: GemmaBackend,
                                            baseCacheDir: String?) -> (any InferenceEngine)? {
        var cacheDir: String? = nil
        if let base = baseCacheDir {
            let dir = (base as NSString).appendingPathComponent(backend.rawValue)
            try? FileManager.default.createDirectory(atPath: dir,
                                                     withIntermediateDirectories: true)
            cacheDir = dir
        }
        // GemmaBridge uses cacheDir (XNNPACK kernel cache); LlamaBridge ignores it.
        let b = runtime.makeEngine()
        // Benchmark instrumentation on: lets the engine report prefill vs decode
        // tok/s separately (see runBenchmark). Harmless for normal use.
        return b.load(modelPath: modelPath,
                      backend: backend.rawValue,
                      cacheDir: cacheDir,
                      benchmark: true) ? b : nil
    }

    func translate(_ text: String) async {
        guard let engine else {
            errorMessage = "Engine not ready"
            return
        }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        isRunning = true
        errorMessage = nil
        output = ""

        let prompt = direction.promptFor(trimmed)
        let maxNew = maxNewTokens
        let temp   = temperature
        let topKv  = topK
        let topPv  = topP

        // No explicit resetSession() needed: GemmaEngine::Generate already
        // recreates the conversation per call (see gemma_engine.cpp:161), so
        // there is no leftover history to clear.

        let t0 = Date()
        let result: String? = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                let r = engine.generate(prompt: prompt,
                                        maxTokens: maxNew,
                                        temperature: temp,
                                        topK: topKv,
                                        topP: topPv)
                continuation.resume(returning: r)
                _ = self
            }
        }
        let elapsed = Date().timeIntervalSince(t0) * 1000

        if let result {
            self.output = result.trimmingCharacters(in: .whitespacesAndNewlines)
            let approxTokens = max(1, result.split(separator: " ").count)
            self.timing = String(format: "%.0f ms · %.1f ms/tok",
                                 elapsed, elapsed / Double(approxTokens))
        } else {
            self.errorMessage = engine.lastError
        }
        self.isRunning = false
    }

    // MARK: - Benchmark

    /// Run the throughput / latency / peak-RAM benchmark on the loaded engine.
    /// Streams generation so we can time the first token (prefill latency)
    /// separately from steady-state decode, and samples memory on a side timer
    /// to catch the transient peak.
    func runBenchmark() async {
        guard let engine, isReady, !isBenchmarking else { return }
        isBenchmarking = true
        errorMessage = nil
        benchmarkResult = nil

        let prompt = benchmarkPrompt
        let mode   = benchmarkMode
        let maxNew = mode.maxTokens
        let iters  = mode.iterations
        let warmup = benchmarkWarmupTokens
        let requested = backend
        let active = activeBackend

        let sampler = PeakMemorySampler()
        sampler.start()

        let result: BenchmarkResult? = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                // One short discarded warm-up so the first prefill / kernel-cache
                // build doesn't skew the average.
                _ = Self.streamOnce(engine: engine, prompt: prompt, maxTokens: warmup)

                var sumTtft = 0.0
                var sumTotal = 0.0
                var sumDecodeMs = 0.0
                var totalTokens = 0
                var decodeTokens = 0
                var lastText = ""

                for _ in 0..<iters {
                    let r = Self.streamOnce(engine: engine,
                                            prompt: prompt,
                                            maxTokens: maxNew)
                    sumTtft  += r.ttftMs
                    sumTotal += r.totalMs
                    sumDecodeMs += max(0, r.totalMs - r.ttftMs)
                    totalTokens  += r.tokens
                    // Throughput counts decode tokens only — the first token's
                    // cost is prefill, already captured in TTFT.
                    decodeTokens += max(0, r.tokens - 1)
                    lastText = r.text
                }

                // Engine's own prefill/decode timing from the last iteration —
                // the authoritative split (engine times prefill and decode
                // separately, independent of our wall clock).
                let eng = engine.lastBenchmark()
                let enginePrefillTps = eng?["prefillTps"]?.doubleValue ?? 0
                let engineDecodeTps  = eng?["decodeTps"]?.doubleValue ?? 0
                let enginePrefillTok = eng?["prefillTokens"]?.intValue ?? 0

                sampler.stop()
                let peakMB = Double(sampler.peakBytes()) / (1024 * 1024)
                let n = Double(iters)
                let throughput = sumDecodeMs > 0
                    ? Double(decodeTokens) / (sumDecodeMs / 1000.0) : 0
                let msPerTok = decodeTokens > 0
                    ? sumDecodeMs / Double(decodeTokens) : 0

                let res = BenchmarkResult(
                    requestedBackend: requested,
                    activeBackend: active,
                    fellBack: requested == .gpu && active == .cpu,
                    mode: mode,
                    iterations: iters,
                    maxTokens: Int(maxNew),
                    totalTokens: totalTokens,
                    avgTtftMs: sumTtft / n,
                    avgTotalMs: sumTotal / n,
                    decodeMsPerTok: msPerTok,
                    throughputTokPerSec: throughput,
                    peakRamMB: peakMB,
                    enginePrefillTps: enginePrefillTps,
                    engineDecodeTps: engineDecodeTps,
                    enginePrefillTokens: enginePrefillTok,
                    sampleOutput: lastText.trimmingCharacters(in: .whitespacesAndNewlines))
                continuation.resume(returning: res)
            }
        }

        benchmarkResult = result
        isBenchmarking = false
        if let result {
            timing = String(format: "%@ benchmark: %.1f tok/s · peak %.0f MB on %@",
                            result.mode.label,
                            result.throughputTokPerSec,
                            result.peakRamMB,
                            result.activeBackend.label)
            // Emit to stderr so a devicectl --console capture picks it up — lets
            // the prefill/decode experiment be read from logs, not the screen.
            FileHandle.standardError.write(Data(String(format:
                "[[BENCH]] backend=%@ mode=%@ wallDecodeTps=%.1f enginePrefillTps=%.1f engineDecodeTps=%.1f prefillTokens=%d peakRamMB=%.0f validOutput=%@\n",
                result.activeBackend.label, result.mode.label,
                result.throughputTokPerSec, result.enginePrefillTps,
                result.engineDecodeTps, result.enginePrefillTokens,
                result.peakRamMB, result.producedValidOutput ? "yes" : "no").utf8))
        }
    }

    // MARK: - Sweep experiment

    /// Prompt-length sweep with repeats, on the loaded engine/backend. For each
    /// prompt length we run R generations (short decode), reading the engine's
    /// own prefill/decode tok/s each time, and emit a `[[SWEEP]]` stderr line.
    /// Run it on CPU then GPU (and across models) and diff the logs — this is
    /// the reproducible dataset behind the prefill-vs-decode claims.
    func runSweep() async {
        guard let engine, isReady, !isBenchmarking else { return }
        isBenchmarking = true
        errorMessage = nil
        let active = activeBackend
        // Target prompt sizes (word counts → a spread of prefill token counts).
        let targets = [8, 32, 128, 320, 640]
        let repeats = 4
        let decodeCap: Int32 = 32

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                // Discarded warm-up.
                _ = Self.streamOnce(engine: engine, prompt: "Hello world.", maxTokens: 8)
                for words in targets {
                    let prompt = Self.makePrompt(words: words)
                    for run in 1...repeats {
                        _ = Self.streamOnce(engine: engine, prompt: prompt, maxTokens: decodeCap)
                        let b = engine.lastBenchmark()
                        let pTps = b?["prefillTps"]?.doubleValue ?? 0
                        let dTps = b?["decodeTps"]?.doubleValue ?? 0
                        let pTok = b?["prefillTokens"]?.intValue ?? 0
                        let dTok = b?["decodeTokens"]?.intValue ?? 0
                        FileHandle.standardError.write(Data(String(format:
                            "[[SWEEP]] backend=%@ targetWords=%d prefillTokens=%d run=%d prefillTps=%.1f decodeTps=%.1f decodeTokens=%d\n",
                            active.label, words, pTok, run, pTps, dTps, dTok).utf8))
                    }
                }
                FileHandle.standardError.write(Data("[[SWEEP]] DONE backend=\(active.label)\n".utf8))
                cont.resume()
            }
        }
        isBenchmarking = false
        timing = "Sweep done on \(active.label) — results in logs"
    }

    /// Build a prompt of roughly `words` filler words (content irrelevant — we
    /// only care about prefill length).
    private nonisolated static func makePrompt(words: Int) -> String {
        let base = ["the","market","was","busy","with","vendors","selling","fresh",
                    "fish","fruit","near","the","harbor","at","dawn","while",
                    "people","walked","past","stalls"]
        var w = [String]()
        var i = 0
        while w.count < words { w.append(base[i % base.count]); i += 1 }
        return "Summarize the following text: " + w.joined(separator: " ") + "."
    }

    /// One streamed generation. `stream(...)` runs the whole decode loop
    /// synchronously on this thread and fires `onChunk` per token, so the
    /// counters are fully settled by the time it returns.
    private nonisolated static func streamOnce(engine: any InferenceEngine,
                                               prompt: String,
                                               maxTokens: Int32)
        -> (ttftMs: Double, totalMs: Double, tokens: Int, text: String) {
        let counters = StreamCounters()
        let t0 = Date()
        engine.stream(prompt: prompt,
                      maxTokens: maxTokens,
                      temperature: 0.2,
                      topK: 40,
                      topP: 0.95) { chunk, _ in
            guard !chunk.isEmpty else { return }
            if counters.tokens == 0 {
                counters.ttftMs = Date().timeIntervalSince(t0) * 1000
            }
            counters.tokens += 1
            counters.text += chunk
        }
        let totalMs = Date().timeIntervalSince(t0) * 1000
        return (counters.ttftMs, totalMs, counters.tokens, counters.text)
    }
}
