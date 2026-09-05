#include "eva-memory.h"

#include <algorithm>
#include <cstring>
#include <cstdio>

namespace eva {

namespace {

inline VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a)
{
    // a is a Vulkan alignment, so a power of two and never zero in practice; the guard is
    // for the degenerate requirement some drivers report for tiny buffers.
    return a ? ((v + a - 1) / a) * a : v;
}

} // namespace


/////////////////////////////////////////////////////////////////////////////////////////
// Slab - one vkAllocateMemory, cut into suballocations.
//
// The free list is kept sorted by offset and coalesced on insert. That is the property the
// old one-allocation-per-buffer path could not have at all: two neighbouring buffers freed
// in either order become one block again, so a later larger request can reuse the space.
// Sorted-and-coalesced also makes `largestFreeBlock` a single scan.
/////////////////////////////////////////////////////////////////////////////////////////
struct DeviceAllocator::Slab
{
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize   size   = 0;
    VkDeviceSize   used   = 0;
    uint32_t       memoryTypeIndex = ~0u;
    uint64_t       id     = 0;
    bool           exportable = false;
    uint8_t*       mapped = nullptr;      // null unless the type is host visible

    struct Range { VkDeviceSize offset, size; };
    std::vector<Range> free;              // sorted by offset, never overlapping

    VkDeviceSize largestFree() const
    {
        VkDeviceSize m = 0;
        for (const Range& r : free) m = std::max(m, r.size);
        return m;
    }

    // The M-1 check. A suballocation that is already free overlaps some range in this list;
    // one that was never handed out by this slab does too, or falls outside it entirely.
    // Cheap because the list is short and sorted - and correctness here matters more than
    // speed, since the alternative is silently handing the same bytes out twice.
    bool overlapsFree(VkDeviceSize offset, VkDeviceSize bytes) const
    {
        const VkDeviceSize end = offset + bytes;
        for (const Range& r : free)
        {
            if (r.offset >= end) break;              // sorted: nothing later can overlap
            if (offset < r.offset + r.size) return true;
        }
        return false;
    }

    // First fit. Best fit was measured and did not pay for itself here: requests inside one
    // type pool are mostly a handful of distinct sizes, so first fit lands in the same block
    // best fit would have picked, without the scan.
    bool carve(VkDeviceSize bytes, VkDeviceSize alignment, VkDeviceSize* outOffset)
    {
        for (size_t i = 0; i < free.size(); ++i)
        {
            const VkDeviceSize base    = alignUp(free[i].offset, alignment);
            const VkDeviceSize padding = base - free[i].offset;
            if (free[i].size < padding || free[i].size - padding < bytes) continue;

            const VkDeviceSize tailOffset = base + bytes;
            const VkDeviceSize tailSize   = free[i].size - padding - bytes;

            if (padding)
            {
                // Alignment padding stays free; it is a real hole and pretending otherwise
                // would make `used` drift away from what was actually handed out.
                free[i].size = padding;
                if (tailSize) free.insert(free.begin() + long(i) + 1, Range{tailOffset, tailSize});
            }
            else if (tailSize)
            {
                free[i] = Range{tailOffset, tailSize};
            }
            else
            {
                free.erase(free.begin() + long(i));
            }

            used += bytes;
            *outOffset = base;
            return true;
        }
        return false;
    }

    void release(VkDeviceSize offset, VkDeviceSize bytes)
    {
        const auto at = std::lower_bound(
            free.begin(), free.end(), offset,
            [](const Range& r, VkDeviceSize o) { return r.offset < o; });
        const auto it = free.insert(at, Range{offset, bytes});
        used -= std::min(used, bytes);

        // Coalesce with the next, then the previous. Doing it in that order keeps `it` valid.
        const auto next = it + 1;
        if (next != free.end() && it->offset + it->size == next->offset)
        {
            it->size += next->size;
            free.erase(next);
        }
        if (it != free.begin())
        {
            const auto prev = it - 1;
            if (prev->offset + prev->size == it->offset)
            {
                prev->size += it->size;
                free.erase(it);
            }
        }
    }

    bool empty() const { return free.size() == 1 && free[0].size == size; }
};


/////////////////////////////////////////////////////////////////////////////////////////
// TypePool - every slab cut from one memory type index.
/////////////////////////////////////////////////////////////////////////////////////////
struct DeviceAllocator::TypePool
{
    uint32_t                            memoryTypeIndex = ~0u;
    std::vector<std::unique_ptr<Slab>>  slabs;
};


/////////////////////////////////////////////////////////////////////////////////////////

DeviceAllocator::DeviceAllocator(VkDevice device, const MemoryTopology& topo, Config cfg)
: device_(device), topo_(&topo), cfg_(cfg)
{
    pools_.resize(topo.memoryTypeCount());
    tracing_ = cfg_.traceEvents;
}

DeviceAllocator::~DeviceAllocator()
{
    // No lock: destroying an allocator another thread is still using is a caller bug that a
    // lock here would hide rather than fix.
    for (auto& pool : pools_)
    {
        if (!pool) continue;
        for (auto& slab : pool->slabs)
        {
            if (!slab || slab->memory == VK_NULL_HANDLE) continue;
            if (slab->mapped) vkUnmapMemory(device_, slab->memory);
            vkFreeMemory(device_, slab->memory, nullptr);
        }
    }
}

DeviceAllocator::TypePool& DeviceAllocator::poolFor(uint32_t memoryTypeIndex)
{
    if (memoryTypeIndex >= pools_.size()) pools_.resize(memoryTypeIndex + 1);
    if (!pools_[memoryTypeIndex])
    {
        pools_[memoryTypeIndex] = std::make_unique<TypePool>();
        pools_[memoryTypeIndex]->memoryTypeIndex = memoryTypeIndex;
    }
    return *pools_[memoryTypeIndex];
}

// Ask the driver, not our own accounting. Another process can take VRAM between two of our
// allocations, and the whole reason for consulting a budget is to find that out before
// vkAllocateMemory fails rather than after.
bool DeviceAllocator::budgetAllows(uint32_t memoryTypeIndex, VkDeviceSize bytes) const
{
    const MemoryTopology::Budget b = topo_->budget(topo_->heapOf(memoryTypeIndex));
    if (b.budget == 0) return true;                       // no live budget: do not second-guess
    const VkDeviceSize ceiling = VkDeviceSize(double(b.budget) * double(cfg_.budgetHeadroom));
    return b.used + bytes <= ceiling;
}

void DeviceAllocator::recordEventLocked(const Suballocation& sub, MemoryTier requested, bool isFree)
{
    if (!tracing_) return;
    TraceEvent e{};
    e.serial = sub.id;
    e.timeNs = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - epoch_).count());
    e.isFree          = isFree;
    e.size            = sub.size;
    e.offset          = sub.offset;
    e.memoryTypeIndex = sub.memoryTypeIndex;
    e.requested       = requested;
    e.actual          = sub.tier;
    e.tag             = AllocTag::current();
    if (cfg_.traceHeapUsage && sub.memoryTypeIndex != ~0u)
    {
        const MemoryTopology::Budget b = topo_->budget(topo_->heapOf(sub.memoryTypeIndex));
        e.heapBudget = b.budget;
        e.heapUsed   = b.used;
    }
    events_.push_back(e);
}

VkResult DeviceAllocator::allocateLocked(const VkMemoryRequirements&  req,
                                         const std::vector<uint32_t>& types,
                                         Suballocation*               out)
{
    if (types.empty() || req.size == 0) return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    for (const uint32_t type : types)
    {
        TypePool& pool = poolFor(type);

        // Existing slabs first, in creation order. Newest-first was tried and is worse: it
        // strands the early slabs, which are the ones large enough to matter.
        for (auto& slab : pool.slabs)
        {
            VkDeviceSize offset = 0;
            if (!slab->carve(req.size, req.alignment, &offset)) continue;

            out->memory          = slab->memory;
            out->offset          = offset;
            out->size            = req.size;
            out->memoryTypeIndex = type;
            out->tier            = topo_->tierOf(type);
            out->mapped          = slab->mapped ? slab->mapped + offset : nullptr;
            out->id              = nextId_++;
            if (cfg_.logLevel >= 2)
                std::fprintf(stderr, "[eva] suballoc %llu  slab %llu  off %llu  size %llu\n",
                             (unsigned long long)out->id, (unsigned long long)slab->id,
                             (unsigned long long)offset, (unsigned long long)req.size);
            return VK_SUCCESS;
        }

        // Nothing fit: commit a new slab. A request larger than the nominal slab gets one of
        // its own rather than inflating the nominal size for everyone afterwards.
        VkDeviceSize slabSize = std::max(cfg_.slabSize, alignUp(req.size, req.alignment));
        while (slabSize > req.size && !budgetAllows(type, slabSize))
        {
            // Give up the nominal size before giving up the type. A tight heap should still
            // serve a small request; it just gets a smaller slab and more of them.
            const VkDeviceSize next = slabSize / 2;
            if (next < std::max(cfg_.minSlabSize, alignUp(req.size, req.alignment))) break;
            slabSize = next;
        }
        slabSize = std::max(slabSize, alignUp(req.size, req.alignment));
        if (!budgetAllows(type, slabSize)) continue;      // this type is full; try the next

        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize  = slabSize;
        ai.memoryTypeIndex = type;

        // Exporting has to be decided here: VkExportMemoryAllocateInfo cannot be applied to
        // memory after the fact, which is why it is a Config flag and not a per-call option.
        VkExportMemoryAllocateInfo ex{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
        if (cfg_.exportable && topo_->canExportOpaqueFd())
        {
            ex.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            ex.pNext       = ai.pNext;
            ai.pNext       = &ex;
        }

        VkDeviceMemory mem = VK_NULL_HANDLE;
        if (vkAllocateMemory(device_, &ai, nullptr, &mem) != VK_SUCCESS) continue;

        auto slab = std::make_unique<Slab>();
        slab->memory          = mem;
        slab->size            = slabSize;
        slab->memoryTypeIndex = type;
        slab->id              = nextId_++;
        slab->exportable      = cfg_.exportable && topo_->canExportOpaqueFd();
        slab->free.push_back(Slab::Range{0, slabSize});

        // Map once, for the life of the slab. Repeated vkMapMemory of the same allocation is
        // legal but pointless, and a persistent pointer is what lets Suballocation::mapped be
        // just slab base + offset.
        if (topo_->propertiesOf(type) & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        {
            void* p = nullptr;
            if (vkMapMemory(device_, mem, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS)
                slab->mapped = static_cast<uint8_t*>(p);
        }

        if (cfg_.logLevel >= 1)
            std::fprintf(stderr, "[eva] slab %llu  type %u  tier %s  size %llu MiB\n",
                         (unsigned long long)slab->id, type,
                         toString(topo_->tierOf(type)),
                         (unsigned long long)(slabSize >> 20));

        VkDeviceSize offset = 0;
        if (!slab->carve(req.size, req.alignment, &offset))
        {
            // Cannot happen unless the slab was sized wrong; treat it as this type failing
            // rather than leaking the allocation.
            if (slab->mapped) vkUnmapMemory(device_, mem);
            vkFreeMemory(device_, mem, nullptr);
            continue;
        }

        out->memory          = slab->memory;
        out->offset          = offset;
        out->size            = req.size;
        out->memoryTypeIndex = type;
        out->tier            = topo_->tierOf(type);
        out->mapped          = slab->mapped ? slab->mapped + offset : nullptr;
        out->id              = nextId_++;
        pool.slabs.push_back(std::move(slab));
        return VK_SUCCESS;
    }
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

VkResult DeviceAllocator::allocate(const VkMemoryRequirements& req,
                                   MemoryTier                  preferred,
                                   MemoryTier                  fallback,
                                   Suballocation*              out)
{
    if (!out) return VK_ERROR_INITIALIZATION_FAILED;
    *out = Suballocation{};
    std::lock_guard<std::mutex> lock(mutex_);

    // tierOf() classifies a memory type physically, so on a device where one type satisfies
    // several tiers it would report the *lowest* tier that type matches - which made an
    // undemoted request look demoted on UMA. Overwrite it with the tier that actually
    // satisfied the request, so `out->tier != preferred` means exactly one thing everywhere.
    const std::vector<uint32_t> first = topo_->candidates(preferred, req.memoryTypeBits);
    if (!first.empty() && allocateLocked(req, first, out) == VK_SUCCESS)
    {
        out->tier = preferred;
        recordEventLocked(*out, preferred, false);
        return VK_SUCCESS;
    }

    if (fallback == preferred) return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    const std::vector<uint32_t> second = topo_->candidates(fallback, req.memoryTypeBits);
    if (!second.empty() && allocateLocked(req, second, out) == VK_SUCCESS)
    {
        out->tier = fallback;
        recordEventLocked(*out, preferred, false);   // requested != actual: the demotion
        return VK_SUCCESS;
    }

    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

VkResult DeviceAllocator::allocateInType(const VkMemoryRequirements& req,
                                         uint32_t                    memoryTypeIndex,
                                         Suballocation*              out)
{
    if (!out) return VK_ERROR_INITIALIZATION_FAILED;
    *out = Suballocation{};

    // A sparse buffer accepts a far narrower type set than an ordinary one, so a caller that
    // has already resolved the type must not have it re-derived from a tier.
    if (!(req.memoryTypeBits & (1u << memoryTypeIndex)))
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    std::lock_guard<std::mutex> lock(mutex_);
    const std::vector<uint32_t> only{memoryTypeIndex};
    const VkResult r = allocateLocked(req, only, out);
    if (r == VK_SUCCESS) recordEventLocked(*out, out->tier, false);
    return r;
}

void DeviceAllocator::free(const Suballocation& sub)
{
    if (!sub.valid()) return;
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& pool : pools_)
    {
        if (!pool) continue;
        for (auto& slab : pool->slabs)
        {
            if (!slab || slab->memory != sub.memory) continue;

            // M-1. A double free, or a suballocation from a different allocator that happens
            // to name the same VkDeviceMemory, must not put overlapping ranges on the free
            // list - the next carve() would hand the same bytes to two callers. This is a
            // counter rather than an EVA_ASSERT because an assert compiles out under NDEBUG,
            // and NDEBUG is precisely the build where the corruption would be silent.
            if (sub.offset + sub.size > slab->size ||
                slab->used < sub.size ||
                slab->overlapsFree(sub.offset, sub.size))
            {
                ++refusedFrees_;
                return;
            }

            slab->release(sub.offset, sub.size);
            recordEventLocked(sub, sub.tier, true);
            return;
        }
    }
    ++refusedFrees_;   // named memory this allocator never handed out
}

VkDeviceSize DeviceAllocator::trim()
{
    std::lock_guard<std::mutex> lock(mutex_);
    VkDeviceSize freed = 0;
    for (auto& pool : pools_)
    {
        if (!pool) continue;
        auto& slabs = pool->slabs;
        for (auto it = slabs.begin(); it != slabs.end(); )
        {
            Slab* s = it->get();
            if (s && s->used == 0 && s->empty())
            {
                if (s->mapped) vkUnmapMemory(device_, s->memory);
                vkFreeMemory(device_, s->memory, nullptr);
                freed += s->size;
                it = slabs.erase(it);
            }
            else ++it;
        }
    }
    return freed;
}

DeviceAllocator::Stats DeviceAllocator::stats(MemoryTier tier) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    Stats st{};
    for (const auto& pool : pools_)
    {
        if (!pool) continue;
        if (topo_->tierOf(pool->memoryTypeIndex) != tier) continue;
        for (const auto& slab : pool->slabs)
        {
            if (!slab) continue;
            st.reserved += slab->size;
            st.live     += slab->used;
            st.largestFreeBlock = std::max(st.largestFreeBlock, slab->largestFree());
            ++st.slabs;
        }
    }
    return st;
}

DeviceAllocator::Stats DeviceAllocator::statsTotal() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    Stats st{};
    for (const auto& pool : pools_)
    {
        if (!pool) continue;
        for (const auto& slab : pool->slabs)
        {
            if (!slab) continue;
            st.reserved += slab->size;
            st.live     += slab->used;
            st.largestFreeBlock = std::max(st.largestFreeBlock, slab->largestFree());
            ++st.slabs;
        }
    }
    return st;
}

std::vector<std::string> DeviceAllocator::traceLines() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> lines;
    char buf[256];
    for (const auto& pool : pools_)
    {
        if (!pool) continue;
        for (const auto& slab : pool->slabs)
        {
            if (!slab) continue;
            std::snprintf(buf, sizeof buf,
                          "slab %llu type=%u tier=%s size=%llu used=%llu exportable=%d",
                          (unsigned long long)slab->id, slab->memoryTypeIndex,
                          toString(topo_->tierOf(slab->memoryTypeIndex)),
                          (unsigned long long)slab->size,
                          (unsigned long long)slab->used,
                          slab->exportable ? 1 : 0);
            lines.emplace_back(buf);
        }
    }
    return lines;
}

void DeviceAllocator::setTracing(bool on)
{
    std::lock_guard<std::mutex> lock(mutex_);
    tracing_ = on;
}

bool DeviceAllocator::tracing() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return tracing_;
}

std::vector<DeviceAllocator::TraceEvent> DeviceAllocator::events() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return events_;
}

void DeviceAllocator::clearEvents()
{
    std::lock_guard<std::mutex> lock(mutex_);
    events_.clear();
}

uint64_t DeviceAllocator::refusedFrees() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return refusedFrees_;
}

void DeviceAllocator::writeTrace(std::FILE* out) const
{
    if (!out) return;
    std::lock_guard<std::mutex> lock(mutex_);
    std::fprintf(out, "# eva allocation timeline\n");
    std::fprintf(out, "time_ns\top\tserial\tslab\ttype\tsize\toffset\t"
                      "requested\tactual\theap_used\theap_budget\ttag\n");
    for (const TraceEvent& e : events_)
    {
        std::fprintf(out,
            "%llu\t%s\t%llu\t%llu\t%u\t%llu\t%llu\t%s\t%s\t%llu\t%llu\t%s\n",
            (unsigned long long)e.timeNs,
            e.isFree ? "free" : "alloc",
            (unsigned long long)e.serial,
            (unsigned long long)e.slabId,
            e.memoryTypeIndex,
            (unsigned long long)e.size,
            (unsigned long long)e.offset,
            toString(e.requested),
            toString(e.actual),
            (unsigned long long)e.heapUsed,
            (unsigned long long)e.heapBudget,
            e.tag ? e.tag : "-");
    }
}

} // namespace eva
