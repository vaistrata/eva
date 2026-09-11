#include "eva-sparse-arena.h"

#include <algorithm>
#include <cstdio>
#include <chrono>
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

    // 계측용. flush() 가 읽어 간다 — 그룹핑(호스트 CPU)과 제출 호출을 가른다.
    double lastGroupUs = 0.0, lastCallUs = 0.0;

    const char* name() const override { return "vkQueueBindSparse"; }
    bool        canBind() const override { return queue_ != VK_NULL_HANDLE; }

    VkResult submit(const std::vector<Bind>& binds,
                    VkSemaphore wait,   uint64_t waitValue,
                    VkSemaphore signal, uint64_t signalValue) override
    {
        if (queue_ == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;

        const auto _tg = std::chrono::steady_clock::now();

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

        const auto _tc = std::chrono::steady_clock::now();
        const VkResult _r = vkQueueBindSparse(queue_, 1, &bi, VK_NULL_HANDLE);
        const auto _te = std::chrono::steady_clock::now();
        lastGroupUs = std::chrono::duration<double, std::micro>(_tc - _tg).count();
        lastCallUs  = std::chrono::duration<double, std::micro>(_te - _tc).count();
        return _r;
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

void SparseArena::removePageFromRuns(std::vector<PageRun>& runs, size_t page)
{
    for (size_t k = 0; k < runs.size(); ++k)
    {
        PageRun& r = runs[k];
        if (page < r.first || page >= r.first + r.count) continue;

        if (r.count == 1)                  { runs.erase(runs.begin() + long(k)); return; }
        if (page == r.first)               { ++r.first; --r.count;               return; }
        if (page == r.first + r.count - 1) { --r.count;                          return; }

        // 중간이다. 앞쪽을 줄이고 뒤쪽을 새 런으로 끼운다. insert 가 r 을 무효화할 수
        // 있으므로 필요한 값을 먼저 다 뽑아 둔다.
        const size_t tailFirst = page + 1;
        const size_t tailCount = r.first + r.count - tailFirst;
        const bool   unbind    = r.unbind;
        r.count = page - r.first;
        runs.insert(runs.begin() + long(k) + 1, PageRun{tailFirst, tailCount, unbind});
        return;
    }
}

MemoryTier SparseArena::stagePin(Region r, MemoryTier preferred)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (r.empty() || pages_.empty()) return MemoryTier::_count;
    if (committedWhole_) return topo_->tierOf(shards_[0].whole.memoryTypeIndex);

    const size_t first = size_t(r.base / pageSize_);
    const size_t last  = size_t((r.base + r.size + pageSize_ - 1) / pageSize_);
    MemoryTier landed = preferred;

    // 페이지마다 할당하지 않는다. 붙일 페이지가 연속이면 **한 번** 할당해서 페이지별로
    // 나눠 준다. 이유가 둘이다.
    //
    //  1. 비용. 4 KiB 마다 allocateInType 을 부르면 granule 256 토큰에서 7,168 번이다
    //     (실측 321 us). 런으로 묶으면 448 번이 된다.
    //  2. 결정성. flush() 의 인접 합치기는 자원 오프셋과 **메모리 오프셋**이 둘 다
    //     이어져야 동작한다. 페이지마다 따로 할당하면 그것이 할당기 운에 달리는데,
    //     런으로 잡으면 정의상 이어진다.
    //
    // free() 는 범위 기반이다(슬랩을 memory 로 찾아 release(offset, size)). 그래서 런을
    // 페이지별 Suballocation 으로 쪼개 두고 하나씩 반납해도 맞는다 — waitIdle() 의
    // 페이지 단위 회수 경로를 바꾸지 않아도 된다.
    size_t i = first;
    while (i < last && i < pages_.size())
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
            // flush() would commit a pin that was never staged. 런 중간이면 쪼개진다.
            removePageFromRuns(stagedRuns_, i);
            p.unbindStaged = false;
            ++i;
            continue;                      // still bound; nothing new to allocate
        }
        if (p.bound || p.pinStaged) { ++i; continue; }

        // 할당이 필요한 연속 구간의 끝을 찾는다. Bind 는 버퍼 하나를 가리키므로
        // shard 경계에서 끊는다.
        const size_t shardIdx = size_t((VAddr(i) * pageSize_) / shardSize_);
        size_t j = i;
        while (j < last && j < pages_.size())
        {
            const Page& q = pages_[j];
            if (q.unbindStaged || q.bound || q.pinStaged) break;
            if (size_t((VAddr(j) * pageSize_) / shardSize_) != shardIdx) break;
            ++j;
        }
        const size_t runLen = j - i;

        const VAddr  addr = VAddr(i) * pageSize_;
        const Shard& sh   = shards_[shardIdx];

        // 런을 한 번에 잡아 본다. 조각화 때문에 큰 요청이 실패할 수 있으므로, 실패하면
        // 아래 페이지 단위 경로로 떨어진다 — 느려질 뿐 실패하지는 않는다.
        DeviceAllocator::Suballocation run{};
        bool runOk = false;
        if (runLen > 1)
        {
            VkMemoryRequirements mr{};
            mr.size           = VkDeviceSize(runLen) * pageSize_;
            mr.alignment      = pageSize_;
            mr.memoryTypeBits = 1u << bindTypeIndex_;
            runOk = alloc_->allocateInType(mr, bindTypeIndex_, &run) == VK_SUCCESS;
        }

        if (runOk)
        {
            for (size_t k = 0; k < runLen; ++k)
            {
                Page& q = pages_[i + k];
                q.mem        = run;
                q.mem.offset = run.offset + VkDeviceSize(k) * pageSize_;
                // 마지막 페이지가 남은 전부를 가진다. 할당기가 요청보다 크게 줬을 때
                // 그 나머지가 어느 페이지에도 속하지 않으면 반납되지 않고 샌다.
                q.mem.size   = (k + 1 == runLen)
                             ? run.size - VkDeviceSize(k) * pageSize_
                             : pageSize_;
                if (run.mapped) q.mem.mapped = run.mapped + VkDeviceSize(k) * pageSize_;
                q.pinStaged = true;
            }
            stagedRuns_.push_back(PageRun{i, runLen, false});
            landed = run.tier;
            staged_.push_back(IBindBackend::Bind{
                sh.buffer, addr % shardSize_,
                VkDeviceSize(runLen) * pageSize_, run.memory, run.offset});
            i = j;
            continue;
        }

        for (size_t k = i; k < j; ++k)
        {
            Page& q = pages_[k];
            VkMemoryRequirements mr{};
            mr.size           = pageSize_;
            mr.alignment      = pageSize_;
            mr.memoryTypeBits = 1u << bindTypeIndex_;
            if (alloc_->allocateInType(mr, bindTypeIndex_, &q.mem) != VK_SUCCESS)
            {
                // Out of memory part way through. Everything staged so far stays staged and
                // valid; the caller is told by the tier, and stats records how far it got.
                ++stats_.pinFailures;
                return MemoryTier::_count;
            }
            landed = q.mem.tier;

            const VAddr  a2  = VAddr(k) * pageSize_;
            const Shard& sh2 = shards_[size_t(a2 / shardSize_)];
            staged_.push_back(IBindBackend::Bind{
                sh2.buffer, a2 % shardSize_, pageSize_, q.mem.memory, q.mem.offset});
            q.pinStaged = true;
            stagedRuns_.push_back(PageRun{k, 1, false});
        }
        i = j;
    }
    return landed;
}

void SparseArena::stageUnpin(Region r)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (r.empty() || committedWhole_) return;

    const size_t first = size_t(r.base / pageSize_);
    const size_t last  = size_t((r.base + r.size + pageSize_ - 1) / pageSize_);

    // 자격 있는 페이지가 이어지는 구간을 하나의 런으로 모은다. 자격 없는 페이지
    // (이미 떼였거나 안 붙어 있는)를 만나면 런을 끊는다.
    size_t runStart = 0, runLen = 0;
    auto closeRun = [&]() {
        if (runLen) { stagedRuns_.push_back(PageRun{runStart, runLen, true}); runLen = 0; }
    };

    for (size_t i = first; i < last && i < pages_.size(); ++i)
    {
        Page& p = pages_[i];
        if (!p.bound || p.unbindStaged || p.unbindSubmitted) { closeRun(); continue; }

        const VAddr addr = i * pageSize_;
        const Shard& sh  = shards_[size_t(addr / shardSize_)];
        staged_.push_back(IBindBackend::Bind{
            sh.buffer, addr % shardSize_, pageSize_, VK_NULL_HANDLE, 0});
        p.unbindStaged = true;
        if (!runLen) runStart = i;
        ++runLen;
    }
    closeRun();
}

VkResult SparseArena::flush(VkSemaphore wait, uint64_t waitValue,
                            VkSemaphore signal, uint64_t signalValue)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // A-1. An empty batch still has to signal. Returning early here means a caller that
    // waits on the semaphore hangs whenever a step happened to stage nothing - which is the
    // common case once the working set stops growing, so it would look intermittent.
    if (staged_.empty() && signal == VK_NULL_HANDLE) return VK_SUCCESS;

    // 인접한 것을 하나로 합친다. 드라이버가 무는 값은 매핑 **개수**이고, 그 개수는
    // 페이지 수가 아니라 **연속 구간 수**여야 한다. 흩어진 4 KiB 항목 448 개는
    // vkQueueBindSparse 호출에서 약 204 us 인데, 같은 개수가 연속이면 6~32 us 다
    // (vkbindwait --stride 실측: 연속 5.6 us vs 간격 2 페이지 208 us).
    //
    // stagePin 이 주소 순서로 넣으므로 인접 여부는 바로 앞 항목만 보면 된다.
    // 자원 오프셋과 메모리 오프셋이 **둘 다** 이어져야 합칠 수 있다 - 메모리가 끊기면
    // 한 항목으로 쓸 수 없다.
    const size_t _staged = staged_.size();
    size_t _pages = 0;
    for (const PageRun& _r : stagedRuns_) _pages += _r.count;
    size_t _merged = 0;
    if (staged_.size() > 1)
    {
        size_t w = 0;
        for (size_t i = 1; i < staged_.size(); ++i)
        {
            IBindBackend::Bind &a = staged_[w];
            const IBindBackend::Bind &b = staged_[i];
            const bool joins = a.buffer == b.buffer &&
                               a.memory == b.memory &&
                               a.resourceOffset + a.size == b.resourceOffset &&
                               a.memoryOffset   + a.size == b.memoryOffset;
            if (joins) { a.size += b.size; ++_merged; }
            else       { staged_[++w] = b; }
        }
        staged_.resize(w + 1);
    }

    const auto _t0 = std::chrono::steady_clock::now();
    const VkResult r = backend_->submit(staged_, wait, waitValue, signal, signalValue);
    const auto _t1 = std::chrono::steady_clock::now();
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
    // 집계는 런당 한 번으로 뺀다. 페이지당 남는 것은 플래그 두 개를 보고 쓰는 것뿐이다.
    uint64_t nBound = 0, nUnbound = 0;
    for (const PageRun& run : stagedRuns_)
    {
        const size_t end = run.first + run.count;
        for (size_t i = run.first; i < end && i < pages_.size(); ++i)
        {
            Page& p = pages_[i];
            if (p.pinStaged)    { p.bound = true; p.pinStaged = false;              ++nBound;   }
            if (p.unbindStaged) { p.unbindStaged = false; p.unbindSubmitted = true; ++nUnbound; }
        }
        // 런 전체를 회수 목록에 넣는다. 전이하지 않은 페이지가 섞여 있어도 아래 회수
        // 루프의 `!p.unbindSubmitted` 가드가 건너뛴다 - 기존 방어를 그대로 남긴다.
        if (run.unbind) unbindSubmittedRuns_.push_back(run);
    }
    stats_.pagesBound    += nBound;
    stats_.residentPages += nBound;
    stats_.pagesUnbound  += nUnbound;
    stats_.residentPages -= std::min<uint64_t>(stats_.residentPages, nUnbound);

    staged_.clear();
    stagedRuns_.clear();
    ++stats_.bindSubmissions;

    if (cfg_.logLevel >= 2)
    {
        const auto _t2 = std::chrono::steady_clock::now();
        // submit 안쪽은 백엔드가 재 둔다 — Vulkan 것은 vkQueueBindSparse 호출뿐이고
        // 나머지는 전부 호스트 CPU 다. 이 셋을 갈라야 무엇을 고칠지가 정해진다.
        double g = 0.0, c = 0.0;
        if (auto *vb = dynamic_cast<VulkanBindBackend *>(backend_.get()))
        { g = vb->lastGroupUs; c = vb->lastCallUs; }
        std::fprintf(stderr,
            // "항목" 은 staged_ 의 크기다. stagePin 이 이미 연속 런을 하나로 묶어
            // 넣으므로 보통 여기서 더 합칠 것이 없다(합침 0) — 합침이 0 이 아니면
            // 별개 stagePin 호출들이 서로 인접했다는 뜻이다.
            "[eva] arena flush: 항목 %zu -> %zu (%zu 합침) · 상주페이지 +%llu · "
            "submit %.1f us (그룹핑 %.1f + 호출 %.1f) · 페이지부기 %.1f us\n",
            _staged, _staged - _merged, _merged,
            (unsigned long long)_pages,
            std::chrono::duration<double, std::micro>(_t1 - _t0).count(), g, c,
            std::chrono::duration<double, std::micro>(_t2 - _t1).count());
    }

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
    // can still reach through a binding that was never removed. unbindSubmittedRuns_ is
    // exactly that set, so this no longer walks the whole reservation to find it.
    //
    // The backend wait above happens before the lock and before any free, which is what makes
    // reclaiming safe: the queue has drained, so no binding still points at this memory.
    for (const PageRun& run : unbindSubmittedRuns_)
    {
        const size_t end = run.first + run.count;
        for (size_t i = run.first; i < end && i < pages_.size(); ++i)
        {
            Page& p = pages_[i];
            if (!p.unbindSubmitted) continue;  // defensive; the set should not contain others
            if (p.mem.valid()) alloc_->free(p.mem);
            p.mem             = DeviceAllocator::Suballocation{};
            p.bound           = false;
            p.unbindSubmitted = false;
        }
    }
    unbindSubmittedRuns_.clear();
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
