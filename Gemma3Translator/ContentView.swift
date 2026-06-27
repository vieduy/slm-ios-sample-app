import SwiftUI

struct ContentView: View {
    @StateObject private var vm = TranslationViewModel()
    @State private var input: String = "Hello, how are you today?"
    @State private var showBenchmark = false

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                backendPicker
                directionPicker
                inputCard
                translateButton
                benchmarkModePicker
                benchmarkButton
                sweepButton
                outputCard
                Spacer()
                statusFooter
            }
            .padding()
            .navigationTitle("Gemma3 Translator")
            .navigationBarTitleDisplayMode(.inline)
        }
        .task { await vm.warmUpIfNeeded() }
        .sheet(isPresented: $showBenchmark) {
            BenchmarkSheet(vm: vm)
        }
    }

    private var backendPicker: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Compute backend")
                .font(.caption).foregroundStyle(.secondary)
            Picker("Backend", selection: $vm.backend) {
                ForEach(GemmaBackend.allCases) { b in
                    Text(b.label).tag(b)
                }
            }
            .pickerStyle(.segmented)
            .disabled(vm.isRunning || vm.isBenchmarking)
            .onChange(of: vm.backend) { newValue in
                Task { await vm.reload(to: newValue) }
            }
        }
    }

    private var directionPicker: some View {
        Picker("Direction", selection: $vm.direction) {
            ForEach(TranslationDirection.allCases) { dir in
                Text(dir.label).tag(dir)
            }
        }
        .pickerStyle(.segmented)
        .onChange(of: vm.direction) { _ in
            input = ""
            vm.output = ""
        }
    }

    private var inputCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(vm.direction.sourceLabel)
                .font(.caption).foregroundStyle(.secondary)
            TextEditor(text: $input)
                .frame(minHeight: 100)
                .padding(8)
                .background(Color(.secondarySystemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    private var translateButton: some View {
        Button {
            Task { await vm.translate(input) }
        } label: {
            HStack {
                if vm.isRunning { ProgressView().controlSize(.small) }
                Text(vm.isRunning ? "Translating…" : "Translate")
                    .frame(maxWidth: .infinity)
            }
            .padding(.vertical, 10)
        }
        .buttonStyle(.borderedProminent)
        .disabled(vm.isRunning
                  || vm.isBenchmarking
                  || !vm.isReady
                  || input.trimmingCharacters(in: .whitespaces).isEmpty)
    }

    private var benchmarkModePicker: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Benchmark size")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text(vm.benchmarkMode.detail)
                    .font(.caption2).foregroundStyle(.tertiary)
            }
            Picker("Benchmark size", selection: $vm.benchmarkMode) {
                ForEach(BenchmarkMode.allCases) { m in
                    Text(m.label).tag(m)
                }
            }
            .pickerStyle(.segmented)
            .disabled(vm.isRunning || vm.isBenchmarking)
        }
    }

    private var benchmarkButton: some View {
        Button {
            Task {
                await vm.runBenchmark()
                showBenchmark = true
            }
        } label: {
            HStack {
                if vm.isBenchmarking { ProgressView().controlSize(.small) }
                Image(systemName: "gauge.with.dots.needle.67percent")
                Text(vm.isBenchmarking ? "Benchmarking…" : "Benchmark (\(vm.benchmarkMode.label))")
                    .frame(maxWidth: .infinity)
            }
            .padding(.vertical, 8)
        }
        .buttonStyle(.bordered)
        .disabled(vm.isRunning || vm.isBenchmarking || !vm.isReady)
    }

    private var sweepButton: some View {
        Button {
            Task { await vm.runSweep() }
        } label: {
            Text(vm.isBenchmarking ? "Running sweep…" : "Run Sweep (CPU/GPU experiment)")
                .frame(maxWidth: .infinity)
                .padding(.vertical, 6)
        }
        .buttonStyle(.bordered)
        .disabled(vm.isRunning || vm.isBenchmarking || !vm.isReady)
    }

    private var outputCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(vm.direction.targetLabel)
                .font(.caption).foregroundStyle(.secondary)
            ScrollView {
                Text(vm.output.isEmpty ? " " : vm.output)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .textSelection(.enabled)
            }
            .frame(minHeight: 120)
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    private var statusFooter: some View {
        Group {
            if let err = vm.errorMessage {
                Text(err).font(.caption).foregroundStyle(.red)
            } else if let s = vm.timing {
                Text(s).font(.caption).foregroundStyle(.secondary)
            } else if !vm.isReady {
                Text("Loading model… (gemma-3-270m, ~285 MB)")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

/// Results panel for the on-device benchmark: backend verdict (CPU vs GPU),
/// throughput, latency split into prefill/decode, and peak RAM.
struct BenchmarkSheet: View {
    @ObservedObject var vm: TranslationViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if let r = vm.benchmarkResult {
                    resultList(r)
                } else {
                    ContentUnavailableViewCompat()
                }
            }
            .navigationTitle("Benchmark")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func resultList(_ r: BenchmarkResult) -> some View {
        List {
            if !r.producedValidOutput {
                Section {
                    Label {
                        Text("\(r.activeBackend.label) produced no valid tokens (only padding/special tokens). The numbers below are not meaningful — this backend can't run this model on your device. Use CPU.")
                            .font(.callout)
                    } icon: {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                    }
                }
            }
            Section("Backend") {
                row("Running on", r.backendLine,
                    highlight: r.fellBack ? .orange : .green)
                row("Requested", r.requestedBackend.label)
                row("Output", r.producedValidOutput ? "valid ✓" : "invalid ✗",
                    highlight: r.producedValidOutput ? .green : .red)
            }
            Section {
                row("Decode (wall clock)", String(format: "%.1f tok/s", r.throughputTokPerSec))
                if r.enginePrefillTps > 0 {
                    row("Prefill (engine)", String(format: "%.1f tok/s", r.enginePrefillTps))
                }
                if r.engineDecodeTps > 0 {
                    row("Decode (engine)", String(format: "%.1f tok/s", r.engineDecodeTps))
                }
            } header: {
                Text("Throughput")
            } footer: {
                if r.enginePrefillTps > 0 {
                    Text("Prefill vs decode are timed separately by LiteRT-LM. Compare these between CPU and GPU: the GPU's gap is in decode (memory-bound, batch-1), while prefill (batched, compute-bound) is far more GPU-friendly.")
                }
            }
            Section("Latency") {
                row("Prefill (TTFT)", String(format: "%.0f ms", r.avgTtftMs))
                row("Decode", String(format: "%.1f ms/tok", r.decodeMsPerTok))
                row("End-to-end", String(format: "%.0f ms / gen", r.avgTotalMs))
            }
            Section("Memory") {
                row("Peak RAM", String(format: "%.0f MB", r.peakRamMB))
            }
            Section {
                Text(r.sampleOutput.isEmpty ? "(no output)" : r.sampleOutput)
                    .font(.callout)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } header: {
                Text("Sample output (\(r.activeBackend.label))")
            } footer: {
                Text("Decoded text from the last run. If this is gibberish on GPU but coherent on CPU, the GPU backend is mishandling this model — use CPU.")
            }
            Section("Run") {
                row("Mode", "\(r.mode.label) (cap \(r.maxTokens))")
                row("Iterations", "\(r.iterations) (+1 warm-up)")
                row("Tokens decoded", "≈ \(r.totalTokens)")
            }
        }
    }

    private func row(_ name: String, _ value: String,
                     highlight: Color? = nil) -> some View {
        HStack {
            Text(name).foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.medium)
                .foregroundStyle(highlight ?? .primary)
                .multilineTextAlignment(.trailing)
        }
    }
}

/// Minimal placeholder so the sheet builds on iOS 16 (ContentUnavailableView
/// is iOS 17+).
private struct ContentUnavailableViewCompat: View {
    var body: some View {
        VStack(spacing: 8) {
            ProgressView()
            Text("No benchmark results yet.")
                .font(.callout).foregroundStyle(.secondary)
        }
    }
}

#Preview {
    ContentView()
}
