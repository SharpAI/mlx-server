// Copyright © 2026 SharpAI
// ssd_streamer.h
// Metal-backed zero-copy asynchronous SSD stream architecture for MoE Experts.

#pragma once

#include <string>
#include <memory>
#include <vector>
#include <dispatch/dispatch.h>
#include <Metal/Metal.hpp>

// Forward declaration from MLX allocator
namespace mlx::core::metal {
    class MetalAllocator;
}

namespace mlx::core::fast {

// The StreamBuffer pool is removed in favor of caller-provided pinned memory
// managed natively by mlx::core::allocator.

class SSDStreamer {
public:
    SSDStreamer(const std::string& file_path, size_t buffer_size_bytes, int num_buffers = 2);
    ~SSDStreamer();

    // Prevent copy/move to maintain descriptor safety
    SSDStreamer(const SSDStreamer&) = delete;
    SSDStreamer& operator=(const SSDStreamer&) = delete;

    /**
    /**
     * Dispatch an asynchronous GCD POSIX read from the safetensors file descriptor.
     * @param byte_offset The offset into the .safetensors file.
     * @param length The number of bytes to read
     * @param dst_ptr The destination pointer (must be GPU accessible, e.g. via MLX allocator)
     * @return The synchronization token that the GPU must wait on before executing.
     */
    void load_sync(off_t byte_offset, size_t length, void* dst_ptr);

private:
    std::string file_path_;
    int fd_;
    dispatch_io_t channel_;
    dispatch_queue_t io_queue_;

    size_t buffer_size_bytes_;

    
    // Hardware synchronization primitives
    MTL::SharedEvent* shared_event_;
    uint64_t event_counter_;
    
public:
    MTL::SharedEvent* get_shared_event() const { return shared_event_; }
    const std::string& get_file_path() const { return file_path_; }
};

} // namespace mlx::core::fast
