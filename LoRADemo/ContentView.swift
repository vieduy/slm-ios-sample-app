import SwiftUI

/// Verifies dynamic LoRA swap: same prompt under adapter A, adapter B, and the
/// base model — outputs should differ.
@MainActor
final class SwapModel: ObservableObject {
    @Published var status = "Idle"
    @Published var prompt = "Translate to Vietnamese: Good morning, how are you?"
    @Published var outA = ""
    @Published var outB = ""
    @Published var outBase = ""
    @Published var running = false
    @Published var backend = "cpu"   // try "gpu" (Metal) too

    private let bridge = LoraBridge()
    private var loaded = false

    private func bundlePath(_ name: String, _ ext: String) -> String? {
        Bundle.main.path(forResource: name, ofType: ext)
    }

    func run() {
        guard !running else { return }
        running = true
        outA = ""; outB = ""; outBase = ""
        let prompt = self.prompt
        let backend = self.backend
        Task.detached(priority: .userInitiated) {
            await self.runImpl(prompt: prompt, backend: backend)
        }
    }

    private func runImpl(prompt: String, backend: String) async {
        guard let base = bundlePath("gemma3-270m-lora64", "litertlm"),
              let a = bundlePath("v4", "tflite"),
              let b = bundlePath("v4.1", "tflite") else {
            await set { $0.status = "Model files missing from bundle" }
            await set { $0.running = false }
            return
        }
        let cache = NSSearchPathForDirectoriesInDomains(.cachesDirectory, .userDomainMask, true).first

        if !loaded {
            await set { $0.status = "Loading base model (\(backend))…" }
            let ok = bridge.loadBase(modelPath: base, backend: backend, cacheDir: cache)
            if !ok {
                await set { $0.status = "Load failed: \(self.bridge.lastError)"; $0.running = false }
                return
            }
            loaded = true
        }

        await set { $0.status = "Adapter A…" }
        let ra = bridge.generate(prompt: prompt, loraPath: a, maxTokens: 64) ?? "‹error: \(bridge.lastError)›"
        await set { $0.outA = ra }

        await set { $0.status = "Adapter B…" }
        let rb = bridge.generate(prompt: prompt, loraPath: b, maxTokens: 64) ?? "‹error: \(bridge.lastError)›"
        await set { $0.outB = rb }

        await set { $0.status = "Base (no adapter)…" }
        let rbase = bridge.generate(prompt: prompt, loraPath: nil, maxTokens: 64) ?? "‹error: \(bridge.lastError)›"
        await set { $0.outBase = rbase }

        let differ = (ra != rb)
        await set {
            $0.status = differ ? "✅ Adapters produce DIFFERENT output" : "⚠️ Outputs identical — check setup"
            $0.running = false
        }
    }

    private func set(_ mutate: @escaping (SwapModel) -> Void) async {
        await MainActor.run { mutate(self) }
    }
}

struct ContentView: View {
    @StateObject private var m = SwapModel()

    var body: some View {
        NavigationStack {
            Form {
                Section("Prompt") {
                    TextField("Prompt", text: $m.prompt, axis: .vertical)
                    Picker("Backend", selection: $m.backend) {
                        Text("CPU").tag("cpu"); Text("GPU (Metal)").tag("gpu")
                    }.pickerStyle(.segmented)
                    Button(m.running ? "Running…" : "Run swap test") { m.run() }
                        .disabled(m.running)
                }
                Section("Status") { Text(m.status).font(.callout) }
                resultSection("Adapter A (v4)", m.outA)
                resultSection("Adapter B (v4.1)", m.outB)
                resultSection("Base (no LoRA)", m.outBase)
            }
            .navigationTitle("LoRA Swap")
        }
    }

    @ViewBuilder private func resultSection(_ title: String, _ text: String) -> some View {
        Section(title) {
            Text(text.isEmpty ? "—" : text)
                .font(.system(.footnote, design: .monospaced))
                .textSelection(.enabled)
        }
    }
}
