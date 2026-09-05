#ifndef EVA_MEMORY_H
#define EVA_MEMORY_H

#include "eva-memory-topology.h"

#include <cstdint>
#include <cstdio>
#include <chrono>
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
/////////////////////////////////////////////////////////////////////////////////////////
// AllocTag - names whatever the current scope allocates, in the timeline.
//
// The allocation trace is meant to be joined against the node schedule, and a few hundred
// anonymous slab cuts cannot be. One RAII object around the interesting call turns the whole
// scope's allocations into named rows. Thread-local so a tag set on one thread cannot mislabel
// another thread's work; nested tags restore the outer one.
/////////////////////////////////////////////////////////////////////////////////////////
class AllocTag
{
public:
    explicit AllocTag(const char* name) : prev_(current()) { current() = name; }
    ~AllocTag() { current() = prev_; }

    AllocTag(const AllocTag&)            = delete;
    AllocTag& operator=(const AllocTag&) = delete;

    // The tag in effect right now, or nullptr. Read by DeviceAllocator when it records an
    // event; safe to call with no tag active.
    static const char*& current()
    {
        static thread_local const char* tag = nullptr;
        return tag;
    }

private:
    const char* prev_ = nullptr;
};


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
        bool         traceEvents    = false;         // record an allocation timeline
        // Sampling the heap on every event costs a driver round trip, so it is separate from
        // traceEvents: you usually want the timeline, and only sometimes want it annotated
        // with how full the heap was at that instant.
        bool         traceHeapUsage = false;
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
    // Two overloads rather than `Config = {}`. A default argument is not a complete-class
    // context, so the NSDMIs in Config are not yet available there and g++ rejects it; an
    // inline constructor body IS a complete-class context, so this compiles. Same shape is
    // required anywhere else a nested type with NSDMIs wants a defaulted parameter.
    DeviceAllocator(VkDevice, const MemoryTopology&, Config);
    DeviceAllocator(VkDevice device, const MemoryTopology& topo)
    : DeviceAllocator(device, topo, Config{}) {}
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

    // ---- allocation timeline ----
    // `requested` vs `actual` is the whole point: the allocator demotes rather than failing,
    // and a demotion is otherwise completely silent. Reading those two columns is the only
    // way a caller can find out that its device-local request landed in host memory.
    struct TraceEvent
    {
        uint64_t     serial      = 0;   // matches Suballocation::id; the join key
        uint64_t     timeNs      = 0;   // since the allocator was constructed
        bool         isFree      = false;
        VkDeviceSize size        = 0;
        VkDeviceSize offset      = 0;   // within the slab
        uint64_t     slabId      = 0;
        uint32_t     memoryTypeIndex = ~0u;
        MemoryTier   requested   = MemoryTier::_count;
        MemoryTier   actual      = MemoryTier::_count;
        VkDeviceSize heapBudget  = 0;   // 0 when traceHeapUsage is off
        VkDeviceSize heapUsed    = 0;
        const char*  tag         = nullptr;
    };

    // Tracing is a run-time switch, not just a construction option: the interesting window is
    // usually one forward pass in the middle of a process that has already allocated plenty.
    void                    setTracing(bool on);
    bool                    tracing() const;
    std::vector<TraceEvent> events() const;
    void                    clearEvents();
    // Tab-separated with a header row, machine readable on purpose - the timeline exists to
    // be joined against the node schedule and against a DRM-level trace, not to be read.
    void                    writeTrace(std::FILE*) const;

    // Frees the allocator refused because they did not describe a live suballocation. It is a
    // counter and not an assert on purpose: an assert compiles out under NDEBUG, which is
    // exactly the build where a double free would go on to corrupt a neighbour.
    uint64_t refusedFrees() const;

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
    uint64_t                               refusedFrees_ = 0;

    bool                    tracing_ = false;
    std::vector<TraceEvent> events_;
    std::chrono::steady_clock::time_point  epoch_ = std::chrono::steady_clock::now();

    void recordEventLocked(const Suballocation&, MemoryTier requested, bool isFree);

    VkResult allocateLocked(const VkMemoryRequirements&, const std::vector<uint32_t>& types,
                            Suballocation*);
    TypePool& poolFor(uint32_t memoryTypeIndex);
    bool      budgetAllows(uint32_t memoryTypeIndex, VkDeviceSize bytes) const;
};

} // namespace eva

#endif // EVA_MEMORY_H
