// ResourceDashboardView.swift — Real-time telemetry overlay 
import SwiftUI

public struct ResourceDashboardView: View {
    @StateObject private var monitor = SystemMonitorService.shared
    
    // Dynamic scaling for gauges based on total physical memory (approximate logic depending on machine specs)
    // Here we default the visual scale to 32GB for Apple Silicon
    private let maxMemoryBytes: UInt64 = 32 * 1024 * 1024 * 1024 // 32 GB
    
    public init() {}
    
    public var body: some View {
        VStack(spacing: 16) {
            HStack {
                Text("SYSTEM RESOURCES")
                    .font(.system(size: 11, weight: .black, design: .rounded))
                    .foregroundStyle(.secondary)
                    .tracking(1.5)
                Spacer()
                Image(systemName: "cpu")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            
            // CPU Block
            ResourceGauge(
                title: "CPU UTILIZATION",
                value: monitor.cpuLoad,
                valueText: String(format: "%.0f%%", monitor.cpuLoad * 100),
                subLabel: "Global Threads",
                color: Color.green
            )
            
            // GPU Block
            ResourceGauge(
                title: "GPU UNIFIED ALLOCATION",
                value: Double(monitor.vramUsedBytes) / Double(maxMemoryBytes),
                valueText: formatBytes(monitor.vramUsedBytes),
                subLabel: "Metal API Reserved",
                color: Color.purple
            )
            
            // MEMORY Block
            ResourceGauge(
                title: "PROCESS FOOTPRINT",
                value: Double(monitor.memoryUsedBytes) / Double(maxMemoryBytes),
                valueText: formatBytes(monitor.memoryUsedBytes),
                subLabel: "Active Memory",
                color: Color.blue
            )
        }
        .padding(20)
        .background(
            ZStack {
                RoundedRectangle(cornerRadius: 16)
                    .fill(.ultraThinMaterial)
                
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color(nsColor: .windowBackgroundColor).opacity(0.3))
                
                // Ambient Glow logic
                LinearGradient(colors: [.green.opacity(0.1), .purple.opacity(0.05), .blue.opacity(0.05)], startPoint: .topLeading, endPoint: .bottomTrailing)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
            }
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(Color.white.opacity(0.15), lineWidth: 0.5)
        )
        .shadow(color: Color.black.opacity(0.3), radius: 15, x: 0, y: 10)
    }
    
    private func formatBytes(_ bytes: UInt64) -> String {
        let gb = Double(bytes) / (1024 * 1024 * 1024)
        if gb >= 1.0 {
            return String(format: "%.1f GB", gb)
        }
        let mb = Double(bytes) / (1024 * 1024)
        return String(format: "%.0f MB", mb)
    }
}

struct ResourceGauge: View {
    let title: String
    let value: Double
    let valueText: String
    let subLabel: String
    let color: Color
    
    var body: some View {
        HStack(spacing: 16) {
            ZStack {
                Circle()
                    .stroke(color.opacity(0.15), lineWidth: 7)
                
                Circle()
                    .trim(from: 0, to: CGFloat(min(max(value, 0.0), 1.0)))
                    .stroke(color, style: StrokeStyle(lineWidth: 7, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .shadow(color: color.opacity(0.6), radius: 6)
                    .animation(.easeOut(duration: 0.5), value: value)
                
                Text(valueText)
                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                    .foregroundColor(.white)
            }
            .frame(width: 55, height: 55)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(.white)
                    .tracking(0.5)
                
                Text(subLabel)
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
                
                // Sleek graph-like ambient line representing full capacity
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Color.white.opacity(0.1))
                            .frame(height: 3)
                        
                        RoundedRectangle(cornerRadius: 2)
                            .fill(
                                LinearGradient(colors: [color.opacity(0.8), color], startPoint: .leading, endPoint: .trailing)
                            )
                            .frame(width: max(0, geometry.size.width * CGFloat(min(max(value, 0.0), 1.0))), height: 3)
                            .shadow(color: color.opacity(0.5), radius: 3)
                            .animation(.easeOut(duration: 0.5), value: value)
                    }
                }
                .frame(height: 3)
                .padding(.top, 2)
            }
            Spacer()
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.black.opacity(0.2))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(color.opacity(0.2), lineWidth: 1)
                )
        )
    }
}
