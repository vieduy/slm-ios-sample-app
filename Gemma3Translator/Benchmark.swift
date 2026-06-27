import Foundation

/// Which compute backend we ask LiteRT-LM to load. The engine only *accepts*
/// the request and either initialises or fails — there is no API to read back
/// the backend actually in use — so "active" here means "the request that
/// succeeded".
enum GemmaBackend: String, CaseIterable, Identifiable {
    case cpu
    case gpu

    var id: String { rawValue }
    var label: String { self == .cpu ? "CPU" : "GPU" }

    /// What LiteRT-LM runs under each request on iOS, for display.
    var detail: String { self == .cpu ? "CPU (XNNPACK)" : "GPU (Metal)" }
}

/// Benchmark workload size. The model's configured context (KV cache) is 1024
/// tokens, so Long generates up to ~full context. Larger modes run fewer
/// iterations to keep total wall-clock reasonable.
enum BenchmarkMode: String, CaseIterable, Identifiable {
    case short
    case medium
    case long

    var id: String { rawValue }
    var label: String {
        switch self {
        case .short:  return "Short"
        case .medium: return "Medium"
        case .long:   return "Long"
        }
    }
    /// Max output tokens to decode.
    var maxTokens: Int32 {
        switch self {
        case .short:  return 128
        case .medium: return 512
        case .long:   return 1024   // ≈ full context for this .litertlm
        }
    }
    /// Measured iterations (averaged). Long runs are ~30 s each, so just one.
    var iterations: Int {
        switch self {
        case .short:  return 3
        case .medium: return 2
        case .long:   return 1
        }
    }
    var detail: String { "\(label) · up to \(maxTokens) tok × \(iterations)" }
}

/// Aggregated results of a benchmark run. Latency is split into prefill
/// (time-to-first-token) and decode (per-token) because they scale very
/// differently — prefill grows with prompt length, decode is steady-state.
struct BenchmarkResult {
    let requestedBackend: GemmaBackend
    let activeBackend: GemmaBackend
    let fellBack: Bool            // requested GPU but loaded on CPU

    let mode: BenchmarkMode
    let iterations: Int
    let maxTokens: Int
    let totalTokens: Int          // summed across measured iterations (≈ stream chunks)

    let avgTtftMs: Double         // prefill latency: prompt → first token
    let avgTotalMs: Double        // wall-clock per generation
    let decodeMsPerTok: Double    // decode latency
    let throughputTokPerSec: Double // decode throughput (tokens after the first)
    let peakRamMB: Double

    /// Engine-measured (LiteRT-LM benchmark instrumentation) throughput from the
    /// last iteration — prefill and decode timed *separately by the engine*, not
    /// by our wall clock. 0 if unavailable. This is the authoritative proof of
    /// the prefill-vs-decode split between CPU and GPU.
    let enginePrefillTps: Double
    let engineDecodeTps: Double
    let enginePrefillTokens: Int

    /// Decoded text from the final measured iteration. Surfaced so the actual
    /// output quality is visible — a backend that is fast but emits gibberish
    /// (e.g. GPU mishandling CPU-quantized weights) is otherwise invisible in
    /// pure timing numbers.
    let sampleOutput: String

    /// One-line backend verdict for the header.
    var backendLine: String {
        fellBack ? "GPU unavailable → ran on \(activeBackend.detail)"
                 : activeBackend.detail
    }

    /// True if the decode actually produced real text. A backend can emit the
    /// configured number of tokens at a healthy tok/s yet have every step
    /// collapse to a special token (e.g. the GPU path on this model returns
    /// only `<pad>` on A12) — in which case the throughput figure is
    /// meaningless. Strip the known special tokens and see if anything remains.
    var producedValidOutput: Bool {
        var s = sampleOutput
        for tok in ["<pad>", "<eos>", "<bos>", "<unk>", "<end_of_turn>", "<start_of_turn>"] {
            s = s.replacingOccurrences(of: tok, with: "")
        }
        return !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

/// Mutable counters for a single streamed generation. A reference type so the
/// `@escaping` token callback can mutate it without capture-by-value surprises.
final class StreamCounters {
    var ttftMs: Double = 0
    var tokens: Int = 0
    var text: String = ""
}
