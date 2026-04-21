// DFlashIntermediateDumper.swift
//
// Utility to dump DFlash intermediate values to .npy files for comparison
// with the Python reference implementation.
//
// Usage: Set DFLASH_DUMP_DIR env var before running SwiftLM.
//        All intermediate arrays are saved as .npy files.
//        Only the first cycle's dumps are saved to avoid huge files.

import Foundation
import MLX

public enum DFlashDumper {
    
    private static var dumpDir: String? = ProcessInfo.processInfo.environment["DFLASH_DUMP_DIR"]
    private static var cycleCount = 0
    private static var saved = Set<String>()
    
    public static var isEnabled: Bool { dumpDir != nil }
    
    public static func setup() {
        if let dir = dumpDir {
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            print("[DFlashDumper] Dumping intermediates to: \(dir)")
        }
        cycleCount = 0
        saved.removeAll()
    }
    
    public static func markCycle() {
        cycleCount += 1
    }
    
    /// Save an MLXArray as a .npy file (float32 format)
    /// Only saves on the first cycle to avoid huge files.
    public static func save(_ name: String, _ arr: MLXArray) {
        guard let dir = dumpDir else { return }
        guard !saved.contains(name) else { return }  // only save first occurrence
        saved.insert(name)
        
        let floatArr = arr.asType(.float32)
        eval(floatArr)
        
        let shape = (0..<floatArr.ndim).map { floatArr.dim($0) }
        let totalElements = shape.reduce(1, *)
        
        // Build numpy .npy header
        let shapeStr = shape.map(String.init).joined(separator: ", ")
        let header = "{'descr': '<f4', 'shape': (\(shapeStr)), 'fortran_order': False}"
        
        var headerBytes = Array(header.utf8)
        headerBytes.append(0x0A)  // newline
        // Pad to 64-byte boundary
        while (headerBytes.count + 10) % 64 != 0 {
            headerBytes.append(0x20)  // space
        }
        
        var fileData = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59])  // \x93NUMPY
        fileData.append(0x01)  // major version 1
        fileData.append(0x00)  // minor version 0
        let headerLen = UInt16(headerBytes.count)
        fileData.append(UInt8(headerLen & 0xFF))
        fileData.append(UInt8((headerLen >> 8) & 0xFF))
        fileData.append(Data(headerBytes))
        
        // Convert to [Float] and write
        let floatData = floatArr.asArray(Float.self)
        floatData.withUnsafeBufferPointer { ptr in
            fileData.append(Data(buffer: ptr))
        }
        
        let url = URL(fileURLWithPath: dir).appendingPathComponent("\(name).npy")
        try? fileData.write(to: url)
    }
    
    /// Save an MLXArray as .npy (int32 format)
    public static func saveInt(_ name: String, _ arr: MLXArray) {
        guard let dir = dumpDir else { return }
        guard !saved.contains(name) else { return }
        saved.insert(name)
        
        let intArr = arr.asType(.int32)
        eval(intArr)
        
        let shape = (0..<intArr.ndim).map { intArr.dim($0) }
        
        let shapeStr = shape.map(String.init).joined(separator: ", ")
        let header = "{'descr': '<i4', 'shape': (\(shapeStr)), 'fortran_order': False}"
        
        var headerBytes = Array(header.utf8)
        headerBytes.append(0x0A)
        while (headerBytes.count + 10) % 64 != 0 {
            headerBytes.append(0x20)
        }
        
        var fileData = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59])
        fileData.append(0x01)
        fileData.append(0x00)
        let headerLen = UInt16(headerBytes.count)
        fileData.append(UInt8(headerLen & 0xFF))
        fileData.append(UInt8((headerLen >> 8) & 0xFF))
        fileData.append(Data(headerBytes))
        
        let intData = intArr.asArray(Int32.self)
        intData.withUnsafeBufferPointer { ptr in
            fileData.append(Data(buffer: ptr))
        }
        
        let url = URL(fileURLWithPath: dir).appendingPathComponent("\(name).npy")
        try? fileData.write(to: url)
    }
}
