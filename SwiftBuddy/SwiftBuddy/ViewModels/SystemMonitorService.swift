import Foundation
import SwiftData
import Metal
import MachO

@MainActor
public final class SystemMonitorService: ObservableObject {
    public static let shared = SystemMonitorService()
    
    @Published public var cpuLoad: Double = 0.0
    @Published public var memoryUsedBytes: UInt64 = 0
    @Published public var vramUsedBytes: UInt64 = 0
    
    private var timer: Timer?
    private var previousCpuInfo: host_cpu_load_info?
    
    private let mtlDevice = MTLCreateSystemDefaultDevice()
    private let hostPort = mach_host_self()
    
    private init() {
        startMonitoring()
    }
    
    public func startMonitoring() {
        timer?.invalidate()
        // Refresh 2 times per second for smooth monitoring UI updates
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.updateMetrics()
            }
        }
    }
    
    public func stopMonitoring() {
        timer?.invalidate()
        timer = nil
    }
    
    private func updateMetrics() {
        updateCPU()
        updateMemory()
        updateGPU()
    }
    
    private func updateCPU() {
        var size = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info_data_t>.size / MemoryLayout<integer_t>.size)
        var cpuLoadInfo = host_cpu_load_info()
        
        let result = withUnsafeMutablePointer(to: &cpuLoadInfo) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(size)) {
                host_statistics64(hostPort, Int32(HOST_CPU_LOAD_INFO), $0, &size)
            }
        }
        
        if result == KERN_SUCCESS {
            if let prev = previousCpuInfo {
                let userDiff = Double(cpuLoadInfo.cpu_ticks.0 - prev.cpu_ticks.0)
                let sysDiff = Double(cpuLoadInfo.cpu_ticks.1 - prev.cpu_ticks.1)
                let idleDiff = Double(cpuLoadInfo.cpu_ticks.2 - prev.cpu_ticks.2)
                let niceDiff = Double(cpuLoadInfo.cpu_ticks.3 - prev.cpu_ticks.3) // nice usually 0
                
                let totalDiff = userDiff + sysDiff + idleDiff + niceDiff
                if totalDiff > 0 {
                    let activeDiff = userDiff + sysDiff + niceDiff
                    self.cpuLoad = activeDiff / totalDiff
                }
            }
            previousCpuInfo = cpuLoadInfo
        }
    }
    
    private func updateMemory() {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        
        let result = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        
        if result == KERN_SUCCESS {
            self.memoryUsedBytes = UInt64(info.phys_footprint)
        }
    }
    
    private func updateGPU() {
        if let device = mtlDevice {
            self.vramUsedBytes = UInt64(device.currentAllocatedSize)
        }
    }
}
