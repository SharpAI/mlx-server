import Foundation
import Darwin
import MLX

public struct MemoryUtils {
    /// OS-level physical memory footprint (what Activity Monitor shows).
    /// Capped by available physical RAM — when this hits the ceiling,
    /// macOS begins swapping to SSD, which degrades performance.
    public static func getOSPhysFootprintGB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size) / 4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            return Double(info.phys_footprint) / (1024.0 * 1024.0 * 1024.0)
        } else {
            return 0.0
        }
    }
    
    /// Total memory the model has requested (physical + swapped to SSD).
    /// This is the TRUE memory demand — it exceeds phys_footprint when macOS
    /// is swapping. Comparing this between Vanilla and TurboQuant shows the
    /// actual compression benefit even at extreme context lengths.
    public static func getTotalMemoryDemandGB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size) / 4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            // internal = private pages (includes compressed + swapped)
            // phys_footprint is capped by RAM. internal is not.
            let _ = Int(info.internal)
            let compressedBytes = Int(info.compressed)
            // Total demand = what's physically wired + what got compressed/swapped
            let total = Int(info.phys_footprint) + compressedBytes
            return Double(total) / (1024.0 * 1024.0 * 1024.0)
        } else {
            return 0.0
        }
    }
    
    /// MLX GPU buffer allocation (what MLX's allocator explicitly tracks).
    public static func getGPUActiveMemoryGB() -> Double {
        return Double(Memory.activeMemory) / (1024.0 * 1024.0 * 1024.0)
    }
    
    /// Combined snapshot: (OS physical GB, total demand GB, GPU active GB)
    public static func snapshot() -> (os: Double, demand: Double, gpu: Double) {
        (getOSPhysFootprintGB(), getTotalMemoryDemandGB(), getGPUActiveMemoryGB())
    }
}
