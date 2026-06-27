import Foundation
import SwiftUI

@MainActor
final class TranslationViewModel: ObservableObject {

    @Published var direction: TranslationDirection = .enToVi
    @Published var output: String = ""
    @Published var isRunning: Bool = false
    @Published var isReady: Bool = false
    @Published var errorMessage: String?
    @Published var timing: String?

    // Benchmark state.
    @Published var backend: GemmaBackend = .cpu
    @Published var benchmarkMode: BenchmarkMode = .short
    @Published var isBenchmarking: Bool = false
    @Published var benchmarkResult: BenchmarkResult?

    private var engine: GemmaBridge?
    private var activeBackend: GemmaBackend = .cpu

    // Resolved once in warmUpIfNeeded(), reused on every backend reload.
    private var modelPath: String?
    private var baseCacheDir: String?

    private let maxNewTokens: Int32 = 128
    private let temperature: Float = 0.2     // translation prefers low temp
    private let topK: Int32 = 40
    private let topP: Float = 0.95

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
        guard let modelPath = ResourceLookup.path("model", ext: "litertlm") else {
            errorMessage = "model.litertlm not in bundle — drop into Resources/."
            return
        }
        self.modelPath = modelPath
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
        let t0 = Date()
        let outcome: (GemmaBridge, GemmaBackend, Bool)? =
            await Task.detached(priority: .userInitiated) {
                if let b = Self.tryLoad(modelPath: modelPath,
                                        backend: requested,
                                        baseCacheDir: baseCacheDir) {
                    return (b, requested, false)
                }
                // GPU couldn't load — retry on CPU so the app stays usable.
                if requested == .gpu,
                   let b = Self.tryLoad(modelPath: modelPath,
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
        self.isReady = true
        let ms = Int(Date().timeIntervalSince(t0) * 1000)
        self.timing = fellBack
            ? "GPU unavailable — loaded on CPU in \(ms) ms"
            : "Loaded on \(active.label) in \(ms) ms"
    }

    /// Build a backend-specific GemmaBridge, or nil if Init fails. The kernel
    /// cache is namespaced per backend so a CPU cache never feeds a GPU load.
    private nonisolated static func tryLoad(modelPath: String,
                                            backend: GemmaBackend,
                                            baseCacheDir: String?) -> GemmaBridge? {
        var cacheDir: String? = nil
        if let base = baseCacheDir {
            let dir = (base as NSString).appendingPathComponent(backend.rawValue)
            try? FileManager.default.createDirectory(atPath: dir,
                                                     withIntermediateDirectories: true)
            cacheDir = dir
        }
        let b = GemmaBridge()
        // Benchmark instrumentation on: lets the engine report prefill vs decode
        // tok/s separately (see runBenchmark). Harmless for normal translation.
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
    private nonisolated static func streamOnce(engine: GemmaBridge,
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
