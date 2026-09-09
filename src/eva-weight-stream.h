#pragma once

// eva::WeightStream - fixed-address weight streaming for models larger than VRAM.
//
// The problem. When the weights do not fit, some bytes cross PCIe every token. That byte
// count is a physical floor and no amount of software removes it. What software can remove
// is the serialisation: `transfer + compute` becomes `max(transfer, compute)` if the copy
// runs on a queue that is not the compute queue. Weight access order is fully deterministic
// (layer 0, 1, ... N, every token), so layer i+k can be fetched while layer i computes.
//
// Why sparse is in here at all. Three designs reach the same data path but differ in what
// the kernel has to know:
//
//   A  read host memory through a sparse mapping   - measured 14.5x VRAM and cannot overlap,
//                                                    because the shader itself stalls
//   B  copy into a VRAM window, plumb the offset   - overlaps, full speed, but the address
//                                                    moves, so descriptors or push constants
//                                                    or the command buffer must follow it
//   C  copy into a VRAM slab, then rebind the VA   - overlaps, full speed, and the address
//                                                    never moves                    <- this
//
// So sparse is not the data path here. The transfer queue moves the bytes; sparse puts the
// result at the same virtual address every time, which is what lets one recorded command
// buffer run for the whole model.
//
// Measured (RTX 3080, NVK, PCIe 16 GT/s, docs/gpu_masterplan/tools/vkslide):
//   a full pass lands within 1.040-1.045x of the host->VRAM copy floor - 96% of the
//   hardware ceiling - at 8x to 32x oversubscription and across bind granules from 1 MiB
//   to 64 MiB, with zero descriptor rewrites and zero command buffer re-records.
//
// Why this class owns the semaphore chain. Getting the order wrong is not a crash. A copy
// that outruns its bind lands on unbacked address space, is silently discarded by strict
// residency, and compute then reads zeros - no VkResult, no validation warning. That was
// observed while validating this design: relying on vkQueueWaitIdle between the bind and
// the copy produced intermittent wrong answers (millions of mismatched words, 1 to 5 runs
// out of 5), and an explicit semaphore fixed it deterministically. The ordering is
// therefore not a caller responsibility.

#include "eva-memory.h"
#include "eva-memory-topology.h"
#include "eva-sparse-arena.h"

#include <memory>
#include <mutex>
#include <vector>

namespace eva
{

class WeightStream
{
public:
    struct Config
    {
        VkDeviceSize layerBytes    = 0;   // bytes of weights per layer. required
        uint32_t     layers        = 0;   // required
        uint32_t     windowSlabs   = 4;   // VRAM slabs held at once. windowSlabs*layerBytes must fit
        uint32_t     prefetchDepth = 2;   // how far ahead to fetch. clamped to windowSlabs-1
        VkDeviceSize bindGranule   = 0;   // 0 = one bind per layer. only affects bind count
        MemoryTier   tier          = MemoryTier::Device;
        VkBufferUsageFlags usage   = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                   | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        uint32_t     logLevel      = 0;
    };

    // What a compute submission must wait on and signal to read one layer. Hand these
    // straight to VkTimelineSemaphoreSubmitInfo. Do not invent your own values: `signal`
    // is what releases the slab back to the window, so skipping it deadlocks the ring, and
    // skipping `wait` is the silent-zeros failure described above.
    struct ReadTicket
    {
        VkSemaphore wait        = VK_NULL_HANDLE;
        uint64_t    waitValue   = 0;
        VkSemaphore signal      = VK_NULL_HANDLE;
        uint64_t    signalValue = 0;
        bool valid() const { return wait != VK_NULL_HANDLE; }
    };

    struct Stats
    {
        VkDeviceSize virtualSize   = 0;   // whole model's address space
        VkDeviceSize resident      = 0;   // window bytes actually backed
        uint32_t     shards        = 0;   // VkBuffers, one per maxStorageBufferRange chunk
        uint32_t     layersPerShard = 0;
        uint32_t     slabs         = 0;
        uint64_t     bindsSubmitted = 0, copiesSubmitted = 0, layersPrefetched = 0;
        VkDeviceSize bindGranule   = 0;
        uint64_t     bindsPerLayer = 0;
        // A caller that never signals `done` shows up here as prefetch stalling, not as a
        // wrong answer, which is the point of keeping the counter.
        uint64_t     stallsOnSlab  = 0;
    };

    WeightStream() = default;
    // `transferQueue` must come from a family with TRANSFER and SPARSE_BINDING but without
    // COMPUTE where one exists. Sharing the compute family is legal but then the copy does
    // not overlap, which removes the only reason to do any of this.
    WeightStream(VkDevice, const MemoryTopology&, DeviceAllocator&,
                 std::unique_ptr<IBindBackend>, VkQueue transferQueue, Config);
    ~WeightStream();
    WeightStream(const WeightStream&)            = delete;
    WeightStream& operator=(const WeightStream&) = delete;

    bool valid() const { return device_ != VK_NULL_HANDLE; }
    // False when the device or the backend cannot change residency. Then every layer is
    // backed up front and `prefetch` is bookkeeping, so the window must hold the model.
    bool sliding() const { return sliding_; }

    // The permanent address of a layer. Constant for the life of the stream - that is the
    // whole contract. Record descriptors and command buffers against these once.
    ArenaRange layer(uint32_t index) const;

    // Bind layer `index` onto its slab and submit `copyCb` to fill it, ordered so the copy
    // cannot outrun the bind, and so the slab's previous tenant has been read first.
    // `copyCb` must be a command buffer on the transfer queue's family that copies this
    // layer's bytes into `layer(index)`.
    VkResult prefetch(uint32_t index, VkCommandBuffer copyCb);

    // Keep the pipeline `Config::prefetchDepth` layers ahead of `index`. Call once before
    // submitting the read for `index`. `copyCbs` is indexed by layer.
    VkResult advance(uint32_t index, const std::vector<VkCommandBuffer>& copyCbs);

    // What compute must wait on and signal to read layer `index`.
    ReadTicket ticket(uint32_t index) const;

    // Drop every binding and rewind to layer 0 so a second pass can run. Waits first.
    VkResult reset();

    VkResult waitIdle();
    Stats    stats() const;

private:
    struct Shard { VkBuffer buffer = VK_NULL_HANDLE; uint32_t firstLayer = 0, layers = 0; };

    VkDevice              device_ = VK_NULL_HANDLE;
    const MemoryTopology* topo_   = nullptr;
    DeviceAllocator*      alloc_  = nullptr;
    VkQueue               xfer_   = VK_NULL_HANDLE;
    Config                cfg_{};
    std::unique_ptr<IBindBackend> backend_;

    mutable std::mutex mutex_;
    bool         sliding_        = false;
    uint32_t     bindTypeIndex_  = ~0u;
    uint32_t     layersPerShard_ = 0;
    VkDeviceSize granule_        = 0;
    uint32_t     nextPrefetch_   = 0;

    std::vector<Shard>                          shards_;
    std::vector<DeviceAllocator::Suballocation> slabs_;
    // Timeline chain. bound -> ready -> done, one tick per layer.
    //   bound[i+1]  the bind for layer i has landed; the copy may run
    //   ready[i+1]  layer i's bytes are in place; compute may read
    //   done[i+1]   compute finished with layer i; its slab may be reused
    VkSemaphore semBound_ = VK_NULL_HANDLE;
    VkSemaphore semReady_ = VK_NULL_HANDLE;
    VkSemaphore semDone_  = VK_NULL_HANDLE;
    Stats stats_{};

    VkResult bindLocked(uint32_t index, VkDeviceMemory memory, VkDeviceSize memoryOffset,
                        VkSemaphore wait, uint64_t waitValue,
                        VkSemaphore signal, uint64_t signalValue);
};

} // namespace eva
