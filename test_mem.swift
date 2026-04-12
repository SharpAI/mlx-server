import Foundation

struct SystemMemory {
    static func getFreeMemoryBytes() -> UInt64 {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        if result == KERN_SUCCESS {
            let pageSize = UInt64(getpagesize())
            let freeMemory = UInt64(stats.free_count) * pageSize
            let inactiveMemory = UInt64(stats.inactive_count) * pageSize
            return freeMemory + inactiveMemory
        }
        return 0
    }
}

print(SystemMemory.getFreeMemoryBytes())
