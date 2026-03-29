import Foundation

func sumBytes(_ p: Progress) -> Int64 {
    if p.children.isEmpty {
        return p.completedUnitCount
    }
    return p.children.reduce(0) { $0 + sumBytes($1) }
}
