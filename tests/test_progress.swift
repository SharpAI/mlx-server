import Foundation

let p = Progress()
print("throughput:", p.throughput ?? "nil")
print("userInfo:", p.userInfo)
