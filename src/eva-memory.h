#ifndef EVA_MEMORY_H
#define EVA_MEMORY_H

#include "eva-memory-topology.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace eva {

/////////////////////////////////////////////////////////////////////////////////////////
// DeviceAllocator - cuts suballocations out of large slabs.
//
// Replaces one vkAllocateMemory per buffer. The count was never the problem on the drivers
// we measured (the RTX 3080 reports maxMemoryAllocationCount = 4294967295); what a dedicated
// allocation per buffer actually costs is per-allocation driver overhead, an inability to
// coalesce neighbouring frees, and an eviction granularity welded to one buffer.
//
// Three properties the old path did not have:
//   - it consults the live heap budget before committing, instead of finding out by failure;
//   - it demotes to a fallback tier rather than asserting when a heap is full;
//   - it can be told to make slabs exportable, which has to be decided at allocation time
//     because VkExportMemoryAllocateInfo cannot be applied to memory after the fact.
//
// Thread-safe: every public entry point takes an internal lock.
/////////////////////////////////////////////////////////////////////////////////////////
class DeviceAllocator
{
public:
    struct Config
    {
        VkDeviceSize slabSize       = 256ull << 20;  // nominal slab; larger requests get their own
        VkDeviceSize minSlabSize    =  16ull << 20;  // floor when the budget is tight
        float        budgetHeadroom = 0.90f;         // never commit past this fraction of budget
        bool         exportable     = false;         // stamp VkExportMemoryAllocateInfo on slabs
        uint32_t     logLevel       = 0;             // 0 off, 1 slabs, 2 every suballocation
    };

    // One handed-out range inside a slab. `memory` is shared with every other suballocation
    // from the same slab, so callers must always pass `offset` on to vkBindBufferMemory.
    struct Suballocation
    {
        VkDeviceMemory memory          = VK_NULL_HANDLE;
        VkDeviceSize   offset          = 0;
        VkDeviceSize   size            = 0;   // rounded up to the request's alignment
        uint32_t       memoryTypeIndex = ~0u;
        MemoryTier     tier            = MemoryTier::_count;   // where it actually landed
        uint8_t*       mapped          = nullptr;              // slab base + offset, host-visible only
        uint64_t       id              = 0;                    // stable, for DRM-level correlation

        bool valid() const { return memory != VK_NULL_HANDLE; }
        explicit operator bool() const { return valid(); }
    };

    DeviceAllocator() = default;
    DeviceAllocator(VkDevice, const MemoryTopology&, Config = {});
    ~DeviceAllocator();

    DeviceAllocator(const DeviceAllocator&)            = delete;
    DeviceAllocator& operator=(const DeviceAllocator&) = delete;

    // Try `preferred`, then `fallback`, honouring req.memoryTypeBits and the live budget.
    // Returns VK_ERROR_OUT_OF_DEVICE_MEMORY only when both tiers are genuinely exhausted;
    // check `out->tier` to see whether a demotion happened.
    VkResult allocate(const VkMemoryRequirements& req,
                      MemoryTier                  preferred,
                      MemoryTier                  fallback,
                      Suballocation*              out);

    // Convenience: no demotion.
    VkResult allocate(const VkMemoryRequirements& req, MemoryTier tier, Suballocation* out)
    {
        return allocate(req, tier, tier, out);
    }

    // Restrict placement to a specific memory type index - the sparse arena needs this,
    // because a sparse buffer accepts a far narrower set of types than an ordinary one.
    VkResult allocateInType(const VkMemoryRequirements& req, uint32_t memoryTypeIndex,
                            Suballocation* out);

    void free(const Suballocation&);

    // Slabs are kept for reuse. Returns bytes handed back to the driver.
    VkDeviceSize trim();

    struct Stats
    {
        VkDeviceSize reserved = 0;   // committed to the driver
        VkDeviceSize live     = 0;   // handed out and not yet freed
        VkDeviceSize largestFreeBlock = 0;
        uint32_t     slabs = 0, liveAllocations = 0;
        // reserved/live; 1.0 is perfect. The old pool's `byteSize * 2.5f` upper bound made
        // this as bad as 2.5 by construction.
        double fragmentation() const { return live ? double(reserved) / double(live) : 1.0; }
    };
    Stats stats(MemoryTier) const;
    Stats statsTotal() const;

    // One line per live slab, machine-readable, for joining against a DRM-level trace.
    // Format: "slab <id> type=<n> tier=<name> size=<bytes> used=<bytes> exportable=<0|1>"
    std::vector<std::string> traceLines() const;

    const MemoryTopology& topology() const { return *topo_; }

private:
    struct Slab;
    struct TypePool;

    VkDevice              device_ = VK_NULL_HANDLE;
    const MemoryTopology* topo_   = nullptr;
    Config                cfg_{};

    mutable std::mutex                     mutex_;
    std::vector<std::unique_ptr<TypePool>> pools_;      // indexed by memory type
    uint64_t                               nextId_ = 1;

    VkResult allocateLocked(const VkMemoryRequirements&, const std::vector<uint32_t>& types,
                            Suballocation*);
    TypePool& poolFor(uint32_t memoryTypeIndex);
    bool      budgetAllows(uint32_t memoryTypeIndex, VkDeviceSize bytes) const;
};

} // namespace eva

#endif // EVA_MEMORY_H
