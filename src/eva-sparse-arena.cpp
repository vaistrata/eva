#include "eva-sparse-arena.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace eva {

/////////////////////////////////////////////////////////////////////////////////////////
// Backends
/////////////////////////////////////////////////////////////////////////////////////////
namespace {

class VulkanBindBackend final : public IBindBackend
{
public:
    VulkanBindBackend(VkDevice d, VkQueue q) : device_(d), queue_(q) {}

    const char* name() const override { return "vkQueueBindSparse"; }
    bool        canBind() const override { return queue_ != VK_NULL_HANDLE; }

    VkResult submit(const std::vector<Bind>& binds,
                    VkSemaphore wait,   uint64_t waitValue,
                    VkSemaphore signal, uint64_t signalValue) override
    {
        if (queue_ == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;

        // One VkSparseBufferMemoryBindInfo per buffer, so group by buffer first. The arena
        // stages in address order, which means the groups are already contiguous, but the
        // grouping is written not to depend on that.
        std::vector<VkBuffer>                            buffers;
        std::vector<std::vector<VkSparseMemoryBind>>     perBuffer;
        for (const Bind& b : binds)
        {
            size_t idx = buffers.size();
            for (size_t i = 0; i < buffers.size(); ++i)
                if (buffers[i] == b.buffer) { idx = i; break; }
            if (idx == buffers.size()) { buffers.push_back(b.buffer); perBuffer.emplace_back(); }

            VkSparseMemoryBind m{};
            m.resourceOffset = b.resourceOffset;
            m.size           = b.size;
            m.memory         = b.memory;          // VK_NULL_HANDLE unbinds the range
            m.memoryOffset   = b.memoryOffset;
            perBuffer[idx].push_back(m);
        }

        std::vector<VkSparseBufferMemoryBindInfo> infos(buffers.size());
        for (size_t i = 0; i < buffers.size(); ++i)
        {
            infos[i].buffer    = buffers[i];
            infos[i].bindCount = (uint32_t)perBuffer[i].size();
            infos[i].pBinds    = perBuffer[i].data();
        }

        VkTimelineSemaphoreSubmitInfo tl{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        tl.waitSemaphoreValueCount   = wait   != VK_NULL_HANDLE ? 1u : 0u;
        tl.pWaitSemaphoreValues      = wait   != VK_NULL_HANDLE ? &waitValue : nullptr;
        tl.signalSemaphoreValueCount = signal != VK_NULL_HANDLE ? 1u : 0u;
        tl.pSignalSemaphoreValues    = signal != VK_NULL_HANDLE ? &signalValue : nullptr;

        VkBindSparseInfo bi{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
        bi.pNext                = &tl;
        bi.bufferBindCount      = (uint32_t)infos.size();
        bi.pBufferBinds         = infos.empty() ? nullptr : infos.data();
        bi.waitSemaphoreCount   = wait   != VK_NULL_HANDLE ? 1u : 0u;
        bi.pWaitSemaphores      = wait   != VK_NULL_HANDLE ? &wait : nullptr;
        bi.signalSemaphoreCount = signal != VK_NULL_HANDLE ? 1u : 0u;
        bi.pSignalSemaphores    = signal != VK_NULL_HANDLE ? &signal : nullptr;

        return vkQueueBindSparse(queue_, 1, &bi, VK_NULL_HANDLE);
    }

    VkResult waitIdle() override
    {
        return queue_ == VK_NULL_HANDLE ? VK_SUCCESS : vkQueueWaitIdle(queue_);
    }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue  queue_  = VK_NULL_HANDLE;
};

class NullBindBackend final : public IBindBackend
{
public:
    const char* name() const override { return "null"; }
    bool        canBind() const override { return false; }
    VkResult submit(const std::vector<Bind>&, VkSemaphore, uint64_t,
                    VkSemaphore, uint64_t) override { return VK_SUCCESS; }
    VkResult waitIdle() override { return VK_SUCCESS; }
};

} // namespace

std::unique_ptr<IBindBackend> makeVulkanBindBackend(VkDevice d, VkQueue q)
{
    return std::make_unique<VulkanBindBackend>(d, q);
}

std::unique_ptr<IBindBackend> makeNullBindBackend()
{
    return std::make_unique<NullBindBackend>();
}


/////////////////////////////////////////////////////////////////////////////////////////
// Arena internals
/////////////////////////////////////////////////////////////////////////////////////////
struct SparseArena::Shard
{
    VkBuffer     buffer = VK_NULL_HANDLE;
    VAddr        base   = 0;
    VkDeviceSize size   = 0;
    // Only used when the backend cannot bind: the shard is backed whole, once.
    DeviceAllocator::Suballocation whole{};
};

// Four booleans rather than one enum because the states genuinely overlap: a page can be
// bound and have an unbind staged, and the distinction between "staged" and "submitted"
// is what stops waitIdle() from reclaiming memory the queue never saw (A-2).
struct SparseArena::Page
{
    DeviceAllocator::Suballocation mem{};
    bool bound           = false;   // committed to the queue and readable
    bool pinStaged       = false;   // bind is in staged_, not yet submitted
    bool unbindStaged    = false;   // unbind is in staged_, not yet submitted
    bool unbindSubmitted = false;   // unbind reached the queue; memory reclaimable on idle
};


SparseArena::SparseArena(VkDevice device, const MemoryTopology& topo, DeviceAllocator& alloc,
                         std::unique_ptr<IBindBackend> backend, Config cfg)
: device_(device), topo_(&topo), alloc_(&alloc), cfg_(cfg), backend_(std::move(backend))
{
    if (!backend_) backend_ = makeNullBindBackend();

    // Sparse only when the device says so AND the backend can actually act on it. Inferring
    // it from the device alone is the A-3 defect: a sparse-capable device with the null
    // backend would report pages resident that were never bound.
    const bool sparse = topo.sparseBinding() && topo.sparseResidency() &&
                        topo.sparsePageSize() != 0 && backend_->canBind();
    committedWhole_ = !sparse;

    // A synthetic page when we are not really paging: the bookkeeping still wants a
    // granularity, and 64 KiB matches what the drivers that do support sparse report.
    pageSize_ = sparse ? topo.sparsePageSize() : 64ull << 10;

    const VkDeviceSize maxRange = topo.maxStorageBufferRange();
    shardSize_ = cfg_.shardSize ? cfg_.shardSize : std::min<VkDeviceSize>(1ull << 30, maxRange);
    shardSize_ = std::min(shardSize_, maxRange);
    shardSize_ = (shardSize_ / pageSize_) * pageSize_;
    if (shardSize_ == 0) { device_ = VK_NULL_HANDLE; return; }

    const VkDeviceSize virtualSize =
        ((cfg_.virtualSize + shardSize_ - 1) / shardSize_) * shardSize_;

    // The memory type pages will come from. A sparse buffer accepts a far narrower set than
    // an ordinary one, so intersect rather than trusting the tier's own candidate list.
    if (sparse)
    {
        const uint32_t bits = topo.sparseMemoryTypeBits();
        for (MemoryTier t : {cfg_.preferredTier, MemoryTier::Device,
                             MemoryTier::DeviceHost, MemoryTier::HostPinned})
        {
            const std::vector<uint32_t> c = topo.candidates(t, bits);
            if (!c.empty()) { bindTypeIndex_ = c[0]; break; }
        }
        if (bindTypeIndex_ == ~0u) { committedWhole_ = true; pageSize_ = 64ull << 10; }
    }

    for (VAddr base = 0; base < virtualSize; base += shardSize_)
    {
        Shard sh{};
        sh.base = base;
        sh.size = std::min(shardSize_, virtualSize - base);

        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        ci.size        = sh.size;
        ci.usage       = cfg_.usage;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (!committedWhole_)
        {
            ci.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT;
            if (topo.sparseResidency()) ci.flags |= VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
        }
        if (vkCreateBuffer(device_, &ci, nullptr, &sh.buffer) != VK_SUCCESS)
        {
            device_ = VK_NULL_HANDLE;
            return;
        }

        if (committedWhole_)
        {
            // No paging available: back the shard whole, once, so every reservation is
            // readable and pin/unpin degrade to bookkeeping. Behaviour is preserved; only
            // the memory saving is lost, which is the honest vendor-neutral fallback.
            VkMemoryRequirements mr{};
            vkGetBufferMemoryRequirements(device_, sh.buffer, &mr);
            if (alloc_->allocate(mr, cfg_.preferredTier, MemoryTier::HostPinned, &sh.whole)
                    != VK_SUCCESS ||
                vkBindBufferMemory(device_, sh.buffer, sh.whole.memory, sh.whole.offset)
                    != VK_SUCCESS)
            {
                vkDestroyBuffer(device_, sh.buffer, nullptr);
                device_ = VK_NULL_HANDLE;
                return;
            }
        }
        shards_.push_back(sh);
    }

    pages_.resize(size_t(virtualSize / pageSize_));
    if (committedWhole_)
        for (Page& p : pages_) p.bound = true;

    stats_.virtualSize = virtualSize;
    stats_.pageCount   = pages_.size();
    if (committedWhole_)
    {
        stats_.residentPages = pages_.size();
        stats_.resident      = virtualSize;
    }

    if (cfg_.logLevel >= 1)
        std::fprintf(stderr,
            "[eva] arena %llu MiB  shard %llu MiB x%zu  page %llu KiB  backend %s%s\n",
            (unsigned long long)(virtualSize >> 20), (unsigned long long)(shardSize_ >> 20),
            shards_.size(), (unsigned long long)(pageSize_ >> 10), backend_->name(),
            committedWhole_ ? "  (committed whole)" : "");
}

SparseArena::~SparseArena()
{
    if (device_ == VK_NULL_HANDLE) return;
    if (backend_) backend_->waitIdle();

    for (Page& p : pages_)
        if (p.mem.valid()) alloc_->free(p.mem);

    for (Shard& sh : shards_)
    {
        if (sh.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device_, sh.buffer, nullptr);
        if (sh.whole.valid()) alloc_->free(sh.whole);
    }
}

uint32_t SparseArena::shardCount() const { return (uint32_t)shards_.size(); }

const char* SparseArena::backendName() const
{
    return backend_ ? backend_->name() : "none";
}

Region SparseArena::reserve(VkDeviceSize bytes)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (bytes == 0 || shards_.empty()) return Region{};

    // Page granularity is not an optimisation here. Two regions sharing a page could never
    // be bound or unbound independently, so a sub-page reservation would silently couple
    // two callers' residency.
    const VkDeviceSize size = ((bytes + pageSize_ - 1) / pageSize_) * pageSize_;
    if (size > shardSize_) return Region{};   // cannot be named by one descriptor

    VAddr base = ((cursor_ + pageSize_ - 1) / pageSize_) * pageSize_;

    // A descriptor cannot straddle two shards, so push to the next shard rather than
    // handing back a region range() would refuse.
    const VAddr shardIndex = base / shardSize_;
    if ((base % shardSize_) + size > shardSize_)
        base = (shardIndex + 1) * shardSize_;

    if (base + size > stats_.virtualSize) return Region{};

    cursor_ = base + size;
    stats_.reserved += size;
    return Region{base, size};
}

MemoryTier SparseArena::stagePin(Region r, MemoryTier preferred)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (r.empty() || pages_.empty()) return MemoryTier::_count;
    if (committedWhole_) return topo_->tierOf(shards_[0].whole.memoryTypeIndex);

    const size_t first = size_t(r.base / pageSize_);
    const size_t last  = size_t((r.base + r.size + pageSize_ - 1) / pageSize_);
    MemoryTier landed = preferred;

    for (size_t i = first; i < last && i < pages_.size(); ++i)
    {
        Page& p = pages_[i];

        // A pin over a range with an unbind still staged cancels the unbind. Without this a
        // grow-then-shrink-then-grow inside one batch would leave the page unbound.
        if (p.unbindStaged)
        {
            for (size_t k = 0; k < staged_.size(); ++k)
            {
                const VAddr addr = i * pageSize_;
                const Shard& sh  = shards_[size_t(addr / shardSize_)];
                if (staged_[k].buffer == sh.buffer &&
                    staged_[k].resourceOffset == addr % shardSize_ &&
                    staged_[k].memory == VK_NULL_HANDLE)
                {
                    staged_.erase(staged_.begin() + long(k));
                    break;
                }
            }
            // The page now carries neither flag, so it must leave the staged index too or
            // flush() would commit a pin that was never staged.
            for (size_t k = 0; k < stagedPages_.size(); ++k)
                if (stagedPages_[k] == i)
                {
                    stagedPages_.erase(stagedPages_.begin() + long(k));
                    break;
                }
            p.unbindStaged = false;
            continue;                      // still bound; nothing new to allocate
        }
        if (p.bound || p.pinStaged) continue;

        VkMemoryRequirements mr{};
        mr.size           = pageSize_;
        mr.alignment      = pageSize_;
        mr.memoryTypeBits = 1u << bindTypeIndex_;
        if (alloc_->allocateInType(mr, bindTypeIndex_, &p.mem) != VK_SUCCESS)
        {
            // Out of memory part way through. Everything staged so far stays staged and
            // valid; the caller is told by the tier, and stats records how far it got.
            ++stats_.pinFailures;
            return MemoryTier::_count;
        }
        landed = p.mem.tier;

        const VAddr addr = i * pageSize_;
        const Shard& sh  = shards_[size_t(addr / shardSize_)];
        staged_.push_back(IBindBackend::Bind{
            sh.buffer, addr % shardSize_, pageSize_, p.mem.memory, p.mem.offset});
        p.pinStaged = true;
        stagedPages_.push_back(i);
    }
    return landed;
}

void SparseArena::stageUnpin(Region r)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (r.empty() || committedWhole_) return;

    const size_t first = size_t(r.base / pageSize_);
    const size_t last  = size_t((r.base + r.size + pageSize_ - 1) / pageSize_);

    for (size_t i = first; i < last && i < pages_.size(); ++i)
    {
        Page& p = pages_[i];
        if (!p.bound || p.unbindStaged || p.unbindSubmitted) continue;

        const VAddr addr = i * pageSize_;
        const Shard& sh  = shards_[size_t(addr / shardSize_)];
        staged_.push_back(IBindBackend::Bind{
            sh.buffer, addr % shardSize_, pageSize_, VK_NULL_HANDLE, 0});
        p.unbindStaged = true;
        stagedPages_.push_back(i);
    }
}

VkResult SparseArena::flush(VkSemaphore wait, uint64_t waitValue,
                            VkSemaphore signal, uint64_t signalValue)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // A-1. An empty batch still has to signal. Returning early here means a caller that
    // waits on the semaphore hangs whenever a step happened to stage nothing - which is the
    // common case once the working set stops growing, so it would look intermittent.
    if (staged_.empty() && signal == VK_NULL_HANDLE) return VK_SUCCESS;

    const VkResult r = backend_->submit(staged_, wait, waitValue, signal, signalValue);
    if (r != VK_SUCCESS)
    {
        // A-5. Keep the batch. Dropping it loses binds the caller believes are queued, and
        // the page states below would then claim a residency that never happened.
        return r;
    }

    // Only the staged pages changed, so only they are visited. Scanning all of pages_ here
    // was O(reservation) and independent of how much was staged - 12.5 ms on a 14 GiB arena
    // against 30 us for the bind it was recording. See bench_flush.cpp.
    //
    // residentPages is maintained incrementally rather than recounted. The predicate is
    // `bound && !unbindSubmitted`, and both transitions below cross it exactly once:
    //
    //   pinStaged    -> bound.           stagePin skips pages already bound, and
    //                                    unbindSubmitted implies bound, so the page was
    //                                    non-resident. +1.
    //   unbindStaged -> unbindSubmitted. stageUnpin requires bound and skips pages already
    //                                    unbindSubmitted, so the page was resident. -1.
    //
    // waitIdle() clears bound and unbindSubmitted together, leaving the predicate false
    // either way, so it does not participate.
    for (const size_t i : stagedPages_)
    {
        Page& p = pages_[i];
        if (p.pinStaged)
        {
            p.bound = true;  p.pinStaged = false;
            ++stats_.pagesBound;
            ++stats_.residentPages;
        }
        if (p.unbindStaged)
        {
            p.unbindStaged = false;  p.unbindSubmitted = true;
            ++stats_.pagesUnbound;
            if (stats_.residentPages) --stats_.residentPages;
        }
    }
    staged_.clear();
    stagedPages_.clear();
    ++stats_.bindSubmissions;

    stats_.resident = stats_.residentPages * pageSize_;
    return VK_SUCCESS;
}

bool SparseArena::hasPendingBinds() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return !staged_.empty();
}

VkResult SparseArena::waitIdle()
{
    VkResult r = VK_SUCCESS;
    if (backend_) r = backend_->waitIdle();

    std::lock_guard<std::mutex> lock(mutex_);
    // A-2. Only pages whose unbind actually reached the queue may have their memory
    // returned. Reclaiming on unbindStaged instead would hand the allocator memory the GPU
    // can still reach through a binding that was never removed.
    for (Page& p : pages_)
    {
        if (!p.unbindSubmitted) continue;
        if (p.mem.valid()) alloc_->free(p.mem);
        p.mem             = DeviceAllocator::Suballocation{};
        p.bound           = false;
        p.unbindSubmitted = false;
    }
    return r;
}

ArenaRange SparseArena::range(Region r) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (r.empty() || shards_.empty()) return ArenaRange{};

    const size_t idx = size_t(r.base / shardSize_);
    if (idx >= shards_.size()) return ArenaRange{};
    const Shard& sh = shards_[idx];
    const VkDeviceSize off = r.base - sh.base;
    if (off + r.size > sh.size) return ArenaRange{};   // straddles; reserve() prevents this
    return ArenaRange{sh.buffer, off, r.size};
}

VkBuffer SparseArena::shardAt(VAddr a) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t idx = size_t(a / shardSize_);
    return idx < shards_.size() ? shards_[idx].buffer : VK_NULL_HANDLE;
}

bool SparseArena::isResident(Region r) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (r.empty()) return true;
    if (committedWhole_) return true;

    const size_t first = size_t(r.base / pageSize_);
    const size_t last  = size_t((r.base + r.size + pageSize_ - 1) / pageSize_);
    for (size_t i = first; i < last && i < pages_.size(); ++i)
    {
        const Page& p = pages_[i];
        // pinStaged is deliberately not enough. Until flush() succeeds the bind has not
        // reached the queue, and a shader reading the page would see undefined data on any
        // device where nonResidentStrict is false.
        if (!p.bound || p.unbindStaged || p.unbindSubmitted) return false;
    }
    return true;
}

SparseArena::Stats SparseArena::stats() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

} // namespace eva
