import Foundation

func getSystemDownloadBytes() -> UInt64 {
    var ifaddr: UnsafeMutablePointer<ifaddrs>?
    guard getifaddrs(&ifaddr) == 0 else { return 0 }
    defer { freeifaddrs(ifaddr) }
    
    var total: UInt64 = 0
    var ptr = ifaddr
    while let p = ptr {
        if p.pointee.ifa_addr != nil, p.pointee.ifa_addr.pointee.sa_family == UInt8(AF_LINK) {
            let data = p.pointee.ifa_data?.bindMemory(to: if_data.self, capacity: 1)
            total += UInt64(data?.pointee.ifi_ibytes ?? 0)
        }
        ptr = p.pointee.ifa_next
    }
    return total
}

let start = getSystemDownloadBytes()
Thread.sleep(forTimeInterval: 1.0)
let end = getSystemDownloadBytes()
print("Speed: \((Double(end - start) / 1048576.0)) MB/s")
