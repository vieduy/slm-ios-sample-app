import Foundation

/// Reads the process' physical memory footprint — the same `phys_footprint`
/// metric iOS jetsam and Xcode's memory gauge report. Resident size alone
/// undercounts compressed/dirty pages, so this is the number to quote for
/// "peak RAM" of an on-device model.
enum MemoryFootprint {
    static func currentBytes() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        return info.phys_footprint
    }
}

/// Polls `phys_footprint` on a background queue and remembers the peak seen
/// between `start()` and `stop()`. Generation runs synchronously on a worker
/// thread, so sampling on a separate timer is the only way to catch the
/// transient peak during decode.
final class PeakMemorySampler {
    private let queue = DispatchQueue(label: "benchmark.mem.sampler")
    private var timer: DispatchSourceTimer?
    private var peak: UInt64 = 0

    func start(interval: TimeInterval = 0.05) {
        peak = MemoryFootprint.currentBytes() ?? 0
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now(), repeating: interval)
        t.setEventHandler { [weak self] in
            guard let self, let b = MemoryFootprint.currentBytes() else { return }
            if b > self.peak { self.peak = b }
        }
        t.resume()
        timer = t
    }

    func stop() {
        timer?.cancel()
        timer = nil
    }

    /// Peak bytes observed. Read after `stop()`; `queue.sync` flushes any
    /// in-flight sample so the value is settled.
    func peakBytes() -> UInt64 { queue.sync { peak } }
}
