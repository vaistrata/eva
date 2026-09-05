#ifndef EVA_SPARSE_ARENA_H
#define EVA_SPARSE_ARENA_H

#include "eva-memory.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace eva {

/////////////////////////////////////////////////////////////////////////////////////////
// SparseArena - a fixed virtual address space whose backing can be swapped at runtime.
//
// This is the piece that makes a statically recorded command buffer compatible with memory
// that moves. A descriptor points at (VkBuffer, offset, range) and never at VkDeviceMemory,
// so rebinding the pages underneath a sparse buffer leaves every descriptor and every
// recorded command untouched. vkBindBufferMemory is one-shot, whole-buffer and permanent;
// vkQueueBindSparse is repeatable, page-granular and ordered on a queue. That difference is
// the whole reason this class exists.
//
// Two limits shape it, both measured rather than assumed:
//   - a storage buffer binding cannot exceed maxStorageBufferRange (4 GiB on the RTX 3080),
//     so the address space is cut into shards and no region may straddle a shard boundary;
//   - a sparse buffer accepts a much narrower set of memory types than an ordinary one
//     (type 1 only on the NVIDIA proprietary driver), so pages go in and out of existence
//     rather than migrating to a cheaper tier.
/////////////////////////////////////////////////////////////////////////////////////////

using VAddr = uint64_t;   // byte offset inside the arena's virtual address space

struct Region
{
    VAddr        base = 0;
    VkDeviceSize size = 0;
    bool empty() const { return size == 0; }

    // Sub-range in bytes, for pinning only the part of a reservation that is live.
    Region slice(VkDeviceSize offset, VkDeviceSize count) const
    {
        return Region{ base + offset, count };
    }
};

// What a descriptor needs. Guaranteed not to straddle a shard.
struct ArenaRange
{
    VkBuffer     buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size   = 0;
};


/////////////////////////////////////////////////////////////////////////////////////////
// IBindBackend - the seam where a vendor-specific path would enter.
//
// Everything above this interface (page table, residency bookkeeping, batching, eviction
// policy) is backend-agnostic. Only the act of committing a batch of mappings differs:
// VkSparseMemoryBind{resourceOffset, size, memory, memoryOffset} and amdgpu's
// AMDGPU_VA_OP_MAP carry very nearly the same payload. Keeping the split here is what makes
// a later DRM-level backend an addition rather than a rewrite.
/////////////////////////////////////////////////////////////////////////////////////////
class IBindBackend
{
public:
    virtual ~IBindBackend() = default;

    struct Bind
    {
        VkBuffer       buffer;
        VkDeviceSize   resourceOffset;   // offset within that shard
        VkDeviceSize   size;
        VkDeviceMemory memory;           // VK_NULL_HANDLE means unbind
        VkDeviceSize   memoryOffset;
    };

    virtual const char* name() const = 0;
    // Whether this backend can actually change residency. The arena must ask rather than
    // infer it from the device: a sparse-capable device fitted with the null backend would
    // otherwise report pages as resident that were never bound, which is a lie that only
    // shows up as garbage reads. When this is false the arena commits its whole address
    // space up front and pin/unpin become bookkeeping.
    virtual bool        canBind() const = 0;
    // Commit `binds` and order them against the compute timeline. A backend that cannot
    // actually unbind (the null backend) may ignore unbind entries.
    virtual VkResult    submit(const std::vector<Bind>& binds,
                               VkSemaphore wait,   uint64_t waitValue,
                               VkSemaphore signal, uint64_t signalValue) = 0;
    virtual VkResult    waitIdle() = 0;
};

// vkQueueBindSparse. The default.
std::unique_ptr<IBindBackend> makeVulkanBindBackend(VkDevice, VkQueue sparseBindQueue);

// For devices without sparseResidencyBuffer (Mali, Adreno, older Intel). The arena commits
// its whole address space up front and pin/unpin become bookkeeping only, so behaviour is
// preserved and only the memory saving is lost. Having this from the start is what keeps
// the vendor-neutral promise honest.
std::unique_ptr<IBindBackend> makeNullBindBackend();


/////////////////////////////////////////////////////////////////////////////////////////
class SparseArena
{
public:
    struct Config
    {
        VkDeviceSize virtualSize = 4ull << 30;   // total address space to reserve
        VkDeviceSize shardSize   = 0;            // 0 = min(1 GiB, maxStorageBufferRange)
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                 | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                 | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        // Page-backing memory comes from here. Must be a type the sparse buffer accepts;
        // the arena intersects this against sparseMemoryTypeBits() and reports a demotion.
        MemoryTier   preferredTier = MemoryTier::Device;
        uint32_t     logLevel      = 0;
    };

    SparseArena() = default;
    // Two overloads, not `Config = {}` - see the same note on DeviceAllocator. A default
    // argument is not a complete-class context, so Config's NSDMIs are not visible there.
    SparseArena(VkDevice, const MemoryTopology&, DeviceAllocator&,
                std::unique_ptr<IBindBackend>, Config);
    SparseArena(VkDevice d, const MemoryTopology& t, DeviceAllocator& a,
                std::unique_ptr<IBindBackend> b)
    : SparseArena(d, t, a, std::move(b), Config{}) {}
    ~SparseArena();

    SparseArena(const SparseArena&)            = delete;
    SparseArena& operator=(const SparseArena&) = delete;

    bool valid() const { return device_ != VK_NULL_HANDLE; }

    // ---- address space ----
    // Reserves virtual space only; zero physical bytes are committed. The size is rounded up
    // to whole pages and the base is page-aligned, because two regions sharing a page could
    // never be bound or unbound independently. A reservation that would straddle a shard
    // boundary is pushed to the start of the next shard.
    Region reserve(VkDeviceSize bytes);

    // ---- residency ----
    // Queue a bind / unbind. Nothing reaches the queue until flush(). Already-resident pages
    // in a pin range are skipped, so calling this every step with a growing range is cheap.
    // Returns the tier the new pages will land in (may differ from Config::preferredTier).
    MemoryTier stagePin(Region, MemoryTier preferred);
    MemoryTier stagePin(Region r) { return stagePin(r, cfg_.preferredTier); }
    void       stageUnpin(Region);

    // Commit everything staged. vkQueueBindSparse is a queue submission with real
    // synchronisation cost, so this must be called on a batch, never per token.
    VkResult flush(VkSemaphore wait = VK_NULL_HANDLE, uint64_t waitValue = 0,
                   VkSemaphore signal = VK_NULL_HANDLE, uint64_t signalValue = 0);

    bool     hasPendingBinds() const;
    VkResult waitIdle();

    // ---- access ----
    // Throws nothing; returns an empty range if the region straddles a shard, which reserve()
    // guarantees cannot happen for regions it produced.
    ArenaRange range(Region) const;
    VkBuffer   shardAt(VAddr) const;

    bool isResident(Region) const;   // every page of it

    // ---- accounting ----
    struct Stats
    {
        VkDeviceSize virtualSize = 0;
        VkDeviceSize reserved    = 0;   // virtual space handed out by reserve()
        VkDeviceSize resident    = 0;   // physically backed right now
        uint64_t     pageCount = 0, residentPages = 0;
        uint64_t     bindSubmissions = 0, pagesBound = 0, pagesUnbound = 0;
        // stagePin ran out of memory part way through a range. Silence here would look like
        // a successful pin whose pages happen not to be resident.
        uint64_t     pinFailures = 0;
    };
    Stats stats() const;

    VkDeviceSize pageSize()   const { return pageSize_; }
    // Out of line on purpose. Inline, `shards_.size()` odr-uses std::vector<Shard> with Shard
    // still incomplete; that compiles in the one TU where Shard is defined and fails in every
    // other consumer, which is a confusing way to find out.
    uint32_t     shardCount() const;
    const char*  backendName() const;

    // True when the backend cannot change residency, so the arena committed everything up
    // front. Callers that reason about memory saved have to branch on this; callers that only
    // need correctness do not.
    bool         committedWhole() const { return committedWhole_; }

private:
    struct Shard;
    struct Page;

    VkDevice              device_ = VK_NULL_HANDLE;
    const MemoryTopology* topo_   = nullptr;
    DeviceAllocator*      alloc_  = nullptr;
    Config                cfg_{};

    std::unique_ptr<IBindBackend> backend_;

    mutable std::mutex mutex_;
    VkDeviceSize       pageSize_  = 0;
    VkDeviceSize       shardSize_ = 0;
    VkDeviceSize       cursor_    = 0;   // next free virtual address
    uint32_t           bindTypeIndex_ = ~0u;
    bool               committedWhole_ = false;

    std::vector<Shard>              shards_;
    std::vector<Page>               pages_;    // one entry per page of the whole arena
    std::vector<IBindBackend::Bind> staged_;
    Stats                           stats_{};
};

} // namespace eva

#endif // EVA_SPARSE_ARENA_H
