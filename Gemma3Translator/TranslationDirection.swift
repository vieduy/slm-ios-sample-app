import Foundation

enum TranslationDirection: String, CaseIterable, Identifiable {
    case enToVi
    case viToEn

    var id: String { rawValue }

    var label: String {
        switch self {
        case .enToVi: return "EN → VI"
        case .viToEn: return "VI → EN"
        }
    }

    var sourceLabel: String { self == .enToVi ? "English" : "Tiếng Việt" }
    var targetLabel: String { self == .enToVi ? "Tiếng Việt" : "English" }

    /// Prompt template handed to Gemma-3-IT. The instruction-tuned model
    /// follows free-form English instructions, so we just describe the
    /// task — no special token (unlike the matmoe encoder which expects
    /// <translate-en-vi>/<translate-vi-en>).
    ///
    /// Kept short on purpose: every instruction token costs prefill compute
    /// per translate. A 30-token preamble vs. 5-token preamble is a real
    /// wall-clock difference on the simulator.
    func promptFor(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        switch self {
        case .enToVi:
            return "Translate to Vietnamese: \(trimmed)"
        case .viToEn:
            return "Translate to English: \(trimmed)"
        }
    }
}
