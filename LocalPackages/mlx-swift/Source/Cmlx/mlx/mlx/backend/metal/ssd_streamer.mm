// Copyright © 2026 SharpAI
// ssd_streamer.mm

#include "mlx/backend/metal/ssd_streamer.h"
#include "mlx/backend/metal/device.h"
#include <unistd.h>
#include <fcntl.h>
#include <iostream>

namespace mlx::core::fast {

SSDStreamer::SSDStreamer(const std::string& file_path, size_t buffer_size_bytes, int num_buffers)
    : file_path_(file_path), buffer_size_bytes_(buffer_size_bytes), event_counter_(0) {
    
    // POSIX open file descriptor
    fd_ = open(file_path_.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("[SSDStreamer] Failed to open safetensors file: " + file_path_);
    }

    // Grand Central Dispatch IO Queue
    // We use a SERIAL queue to enforce strictly monotonic MTLSharedEvent signaling.
    // If it were concurrent, pread completions could signal out-of-order, bypassing GPU wait operations
    // and causing kIOGPUCommandBufferCallbackErrorTimeout.
    io_queue_ = dispatch_queue_create("com.mlx.ssd_streamer", DISPATCH_QUEUE_SERIAL);
    
    // We do NOT use dispatch_io_create because dispatch_io_read copies data 
    // into a dispatch_data_t. We want zero-copy direct pread into mapped Metal memory.
    
    auto& d = metal::device(mlx::core::Device::gpu);
    
    // Create shared event for GPU-CPU synchronization without blocking evaluation threads
    shared_event_ = d.mtl_device()->newSharedEvent();
}

SSDStreamer::~SSDStreamer() {
    if (fd_ >= 0) close(fd_);
    if (shared_event_) shared_event_->release();
}

void SSDStreamer::load_sync(off_t byte_offset, size_t length, void* dst_ptr) {
    if (length > buffer_size_bytes_) {
        throw std::invalid_argument("[SSDStreamer] Load length exceeds Pinned Buffer capacity.");
    }

    // Synchronously read exactly byte_offset into the MLX allocator CPU mapped pointer.
    // By blocking the MLX graph evaluator thread, we implement absolute backpressure
    // against the fast CPU evaluation loop, preventing the Apple GPU Watchdog from
    // timing out due to excessive queued disk accesses.
    ssize_t result = pread(fd_, dst_ptr, length, byte_offset);
    if (result < 0) {
        std::cerr << "[SSDStreamer] load_sync pread failed with error " << errno << std::endl;
        throw std::runtime_error("[SSDStreamer] pread failed");
    } else if ((size_t)result != length) {
        std::cerr << "[SSDStreamer] load_sync partial read: " << result << " of " << length << std::endl;
        throw std::runtime_error("[SSDStreamer] pread partial");
    }
}

} // namespace mlx::core::fast
