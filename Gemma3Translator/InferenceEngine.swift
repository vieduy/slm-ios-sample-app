import Foundation

/// Common surface implemented by both on-device engines so the view model is
/// engine-agnostic:
///   • GemmaBridge  → LiteRT-LM (.litertlm)
///   • LlamaBridge  → llama.cpp (.gguf, e.g. Bonsai-1.7B Q1_0)
/// Both Obj-C classes already expose these exact selectors (see their headers),
/// so the conformances below are empty — the protocol just unifies the type.
protocol InferenceEngine: AnyObject {
    func load(modelPath: String, backend: String, cacheDir: String?, benchmark: Bool) -> Bool
    func lastBenchmark() -> [String: NSNumber]?
    func generate(prompt: String, maxTokens: Int32, temperature: Float, topK: Int32, topP: Float) -> String?
    func stream(prompt: String, maxTokens: Int32, temperature: Float, topK: Int32, topP: Float,
                onChunk: @escaping (String, Bool) -> Void)
    func resetSession()
    var lastError: String { get }

    // Dynamic LoRA. Only the llama.cpp engine implements these; the LiteRT path
    // falls through to the no-op defaults below (its text-LoRA call is stubbed).
    func loadAdapter(path: String, identifier: String) -> Bool
    func setActiveAdapter(_ identifier: String?, scale: Float) -> Bool
}

extension InferenceEngine {
    func loadAdapter(path: String, identifier: String) -> Bool { false }
    func setActiveAdapter(_ identifier: String?, scale: Float) -> Bool { false }
}

extension GemmaBridge: InferenceEngine {}
extension LlamaBridge: InferenceEngine {}

/// Which on-device runtime backs the currently bundled model, chosen by the
/// model file present in the app bundle.
enum ModelRuntime {
    case liteRT   // model.litertlm  → GemmaBridge
    case llamaCpp // model.gguf      → LlamaBridge

    /// Resolve the bundled model. Prefers a .gguf (llama.cpp) if both exist.
    static func resolve() -> (runtime: ModelRuntime, path: String)? {
        if let p = ResourceLookup.path("model", ext: "gguf")     { return (.llamaCpp, p) }
        if let p = ResourceLookup.path("model", ext: "litertlm") { return (.liteRT, p) }
        return nil
    }

    func makeEngine() -> InferenceEngine {
        switch self {
        case .liteRT:   return GemmaBridge()
        case .llamaCpp: return LlamaBridge()
        }
    }
}
