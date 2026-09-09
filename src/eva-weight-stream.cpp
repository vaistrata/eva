#include "eva-weight-stream.h"

#include <algorithm>
#include <cstdio>

namespace eva
{

namespace
{
VkResult
makeTimeline(VkDevice dev, VkSemaphore* out)
{
    VkSemaphoreTypeCreateInfo t{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    t.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    t.initialValue  = 0;
    VkSemaphoreCreateInfo s{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    s.pNext = &t;
    return vkCreateSemaphore(dev, &s, nullptr, out);
}
} // namespace

WeightStream::WeightStream(VkDevice device, const MemoryTopology& topo, DeviceAllocator& alloc,
                           std::unique_ptr<IBindBackend> backend, VkQueue transferQueue,
                           Config cfg)
: device_(device), topo_(&topo), alloc_(&alloc), xfer_(transferQueue), cfg_(cfg),
  backend_(std::move(backend))
{
    if (!backend_) backend_ = makeNullBindBackend();
    if (!cfg_.layers || !cfg_.layerBytes || xfer_ == VK_NULL_HANDLE) {
        device_ = VK_NULL_HANDLE; return;
    }
    if (!cfg_.windowSlabs) cfg_.windowSlabs = 1;
    if (cfg_.windowSlabs > cfg_.layers) cfg_.windowSlabs = cfg_.layers;
    if (cfg_.prefetchDepth >= cfg_.windowSlabs) cfg_.prefetchDepth = cfg_.windowSlabs - 1;
    if (!cfg_.prefetchDepth) cfg_.prefetchDepth = 1;

    // Sparse only when the device says so AND the backend can act on it. Same reasoning as
    // SparseArena: a sparse-capable device with the null backend would otherwise report a
    // residency that never happened.
    sliding_ = topo.sparseBinding() && topo.sparseResidency() && backend_->canBind();

    const VkDeviceSize pageSize = sliding_ ? std::max<VkDeviceSize>(topo.sparsePageSize(), 1)
                                           : 1;
    // A layer must start on a page or its first bind cannot name it.
    if (sliding_ && (cfg_.layerBytes % pageSize)) {
        if (cfg_.logLevel >= 1)
            std::fprintf(stderr, "[eva] WeightStream: layerBytes %llu is not a multiple of the "
                                 "%llu B sparse page; rounding up\n",
                         (unsigned long long)cfg_.layerBytes, (unsigned long long)pageSize);
        cfg_.layerBytes = ((cfg_.layerBytes + pageSize - 1) / pageSize) * pageSize;
    }

    granule_ = cfg_.bindGranule ? cfg_.bindGranule : cfg_.layerBytes;
    if (sliding_ && (granule_ % pageSize))
        granule_ = ((granule_ + pageSize - 1) / pageSize) * pageSize;
    granule_ = std::min(granule_, cfg_.layerBytes);
    if (!granule_) granule_ = cfg_.layerBytes;

    // One VkBuffer can only be named by a descriptor up to maxStorageBufferRange, so the
    // model is cut into shards. A layer never straddles a shard - that keeps `layer()` a
    // single ArenaRange, which is what callers record against.
    const VkDeviceSize maxRange = std::max<VkDeviceSize>(topo.maxStorageBufferRange(),
                                                         cfg_.layerBytes);
    layersPerShard_ = (uint32_t)std::max<VkDeviceSize>(1, maxRange / cfg_.layerBytes);
    const uint32_t nShards = (cfg_.layers + layersPerShard_ - 1) / layersPerShard_;

    for (uint32_t s = 0; s < nShards; s++) {
        Shard sh{};
        sh.firstLayer = s * layersPerShard_;
        sh.layers     = std::min(layersPerShard_, cfg_.layers - sh.firstLayer);
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size  = (VkDeviceSize)sh.layers * cfg_.layerBytes;
        bci.usage = cfg_.usage;
        if (sliding_)
            bci.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
        if (vkCreateBuffer(device_, &bci, nullptr, &sh.buffer) != VK_SUCCESS) {
            if (cfg_.logLevel >= 1)
                std::fprintf(stderr, "[eva] WeightStream: shard %u (%.2f MiB) creation failed\n",
                             s, bci.size / 1048576.0);
            device_ = VK_NULL_HANDLE; return;
        }
        shards_.push_back(sh);
    }

    // The memory type slabs come from. A sparse buffer accepts a narrower set than an
    // ordinary one, so intersect rather than trusting the tier's list.
    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(device_, shards_[0].buffer, &mr);
    const uint32_t bits = sliding_ ? (topo.sparseMemoryTypeBits() & mr.memoryTypeBits)
                                   : mr.memoryTypeBits;
    for (MemoryTier t : {cfg_.tier, MemoryTier::Device, MemoryTier::DeviceHost,
                         MemoryTier::HostPinned}) {
        const std::vector<uint32_t>& c = topo.candidates(t, bits);
        if (!c.empty()) { bindTypeIndex_ = c[0]; break; }
    }
    if (bindTypeIndex_ == ~0u) { device_ = VK_NULL_HANDLE; return; }

    // Slabs. In sliding mode there are windowSlabs of them; otherwise the model has to be
    // fully backed, so there is one per layer and prefetch degenerates to bookkeeping.
    const uint32_t nSlabs = sliding_ ? cfg_.windowSlabs : cfg_.layers;
    slabs_.resize(nSlabs);
    for (uint32_t i = 0; i < nSlabs; i++) {
        VkMemoryRequirements req{};
        req.size           = cfg_.layerBytes;
        req.alignment      = std::max<VkDeviceSize>(mr.alignment, 1);
        req.memoryTypeBits = 1u << bindTypeIndex_;
        if (alloc_->allocateInType(req, bindTypeIndex_, &slabs_[i]) != VK_SUCCESS) {
            if (cfg_.logLevel >= 1)
                std::fprintf(stderr, "[eva] WeightStream: slab %u of %u (%.2f MiB) failed - "
                                     "lower windowSlabs\n",
                             i, nSlabs, cfg_.layerBytes / 1048576.0);
            slabs_.resize(i);
            device_ = VK_NULL_HANDLE; return;
        }
    }

    if (makeTimeline(device_, &semBound_) != VK_SUCCESS ||
        makeTimeline(device_, &semReady_) != VK_SUCCESS ||
        makeTimeline(device_, &semDone_)  != VK_SUCCESS) {
        if (cfg_.logLevel >= 1)
            std::fprintf(stderr, "[eva] WeightStream: timeline semaphores unavailable - "
                                 "the ordering this class exists to guarantee cannot be built\n");
        device_ = VK_NULL_HANDLE; return;
    }

    stats_.virtualSize    = (VkDeviceSize)cfg_.layers * cfg_.layerBytes;
    stats_.shards         = (uint32_t)shards_.size();
    stats_.layersPerShard = layersPerShard_;
    stats_.slabs          = (uint32_t)slabs_.size();
    stats_.bindGranule    = granule_;
    stats_.bindsPerLayer  = (cfg_.layerBytes + granule_ - 1) / granule_;

    // Not sliding means every layer is permanently backed, so bind once here and let
    // prefetch be a no-op. Behaviour is preserved; only the memory saving is lost.
    if (!sliding_) {
        for (uint32_t i = 0; i < cfg_.layers; i++) {
            const ArenaRange r = layer(i);
            if (vkBindBufferMemory(device_, r.buffer, slabs_[i].memory, slabs_[i].offset)
                != VK_SUCCESS) { device_ = VK_NULL_HANDLE; return; }
        }
        stats_.resident = stats_.virtualSize;
    } else {
        stats_.resident = (VkDeviceSize)slabs_.size() * cfg_.layerBytes;
    }

    if (cfg_.logLevel >= 1)
        std::fprintf(stderr,
            "[eva] WeightStream %.2f GiB model / %.2f GiB window (%.1fx) - %u shards, "
            "%llu binds/layer, depth %u, backend %s%s\n",
            stats_.virtualSize / 1073741824.0, stats_.resident / 1073741824.0,
            (double)stats_.virtualSize / (double)std::max<VkDeviceSize>(stats_.resident, 1),
            stats_.shards, (unsigned long long)stats_.bindsPerLayer, cfg_.prefetchDepth,
            backend_->name(), sliding_ ? "" : "  (not sliding: fully backed)");
}

WeightStream::~WeightStream()
{
    if (device_ == VK_NULL_HANDLE && shards_.empty() && slabs_.empty()) return;
    if (backend_) backend_->waitIdle();
    for (auto& s : slabs_) if (s.valid()) alloc_->free(s);
    for (auto& sh : shards_)
        if (sh.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device_, sh.buffer, nullptr);
    if (semBound_) vkDestroySemaphore(device_, semBound_, nullptr);
    if (semReady_) vkDestroySemaphore(device_, semReady_, nullptr);
    if (semDone_)  vkDestroySemaphore(device_, semDone_, nullptr);
}

ArenaRange
WeightStream::layer(uint32_t index) const
{
    if (index >= cfg_.layers || shards_.empty()) return ArenaRange{};
    const uint32_t s = index / layersPerShard_;
    const Shard& sh  = shards_[s];
    return ArenaRange{ sh.buffer, (VkDeviceSize)(index - sh.firstLayer) * cfg_.layerBytes,
                       cfg_.layerBytes };
}

VkResult
WeightStream::bindLocked(uint32_t index, VkDeviceMemory memory, VkDeviceSize memoryOffset,
                         VkSemaphore wait, uint64_t waitValue,
                         VkSemaphore signal, uint64_t signalValue)
{
    const ArenaRange r = layer(index);
    if (r.buffer == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;

    std::vector<IBindBackend::Bind> binds;
    binds.reserve((size_t)stats_.bindsPerLayer);
    for (VkDeviceSize o = 0; o < cfg_.layerBytes; o += granule_) {
        const VkDeviceSize sz = std::min(granule_, cfg_.layerBytes - o);
        binds.push_back(IBindBackend::Bind{ r.buffer, r.offset + o, sz,
                                            memory, memory ? memoryOffset + o : 0 });
    }
    const VkResult res = backend_->submit(binds, wait, waitValue, signal, signalValue);
    if (res == VK_SUCCESS) stats_.bindsSubmitted += binds.size();
    return res;
}

VkResult
WeightStream::prefetch(uint32_t index, VkCommandBuffer copyCb)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (device_ == VK_NULL_HANDLE || index >= cfg_.layers)
        return VK_ERROR_INITIALIZATION_FAILED;

    const uint64_t tick = (uint64_t)index + 1;

    if (sliding_) {
        // The slab is layer % windowSlabs, so it is still holding layer index-windowSlabs
        // until that layer's read has signalled done. Waiting on the bind rather than on the
        // host is what keeps the transfer queue running ahead.
        const bool reuse = index >= cfg_.windowSlabs;
        if (reuse) stats_.stallsOnSlab++;
        const DeviceAllocator::Suballocation& sl = slabs_[index % cfg_.windowSlabs];
        const VkResult r = bindLocked(index, sl.memory, sl.offset,
                                      reuse ? semDone_ : VK_NULL_HANDLE,
                                      reuse ? (uint64_t)(index - cfg_.windowSlabs) + 1 : 0,
                                      semBound_, tick);
        if (r != VK_SUCCESS) return r;
    } else {
        // Fully backed already. Still tick `bound` so the copy below has something to wait
        // on and the caller's chain is identical in both modes.
        VkSemaphoreSignalInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        si.semaphore = semBound_; si.value = tick;
        const VkResult r = vkSignalSemaphore(device_, &si);
        if (r != VK_SUCCESS) return r;
    }

    // The copy must not outrun the bind. This is the ordering the class exists to own: a
    // copy onto unbacked address space is discarded in silence, not reported.
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    uint64_t waitValue = tick, signalValue = tick;
    VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ts.waitSemaphoreValueCount   = 1; ts.pWaitSemaphoreValues   = &waitValue;
    ts.signalSemaphoreValueCount = 1; ts.pSignalSemaphoreValues = &signalValue;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext                = &ts;
    si.waitSemaphoreCount   = 1;  si.pWaitSemaphores   = &semBound_;
    si.pWaitDstStageMask    = &stage;
    si.commandBufferCount   = copyCb ? 1u : 0u;
    si.pCommandBuffers      = copyCb ? &copyCb : nullptr;
    si.signalSemaphoreCount = 1;  si.pSignalSemaphores = &semReady_;
    const VkResult r = vkQueueSubmit(xfer_, 1, &si, VK_NULL_HANDLE);
    if (r != VK_SUCCESS) return r;

    stats_.copiesSubmitted += copyCb ? 1 : 0;
    stats_.layersPrefetched++;
    if (index + 1 > nextPrefetch_) nextPrefetch_ = index + 1;
    return VK_SUCCESS;
}

VkResult
WeightStream::advance(uint32_t index, const std::vector<VkCommandBuffer>& copyCbs)
{
    if (device_ == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;
    const uint32_t target = std::min<uint32_t>(cfg_.layers, index + cfg_.prefetchDepth + 1);
    uint32_t start;
    { std::lock_guard<std::mutex> lock(mutex_); start = nextPrefetch_; }
    for (uint32_t j = start; j < target; j++) {
        VkCommandBuffer cb = j < copyCbs.size() ? copyCbs[j] : VK_NULL_HANDLE;
        const VkResult r = prefetch(j, cb);
        if (r != VK_SUCCESS) return r;
    }
    return VK_SUCCESS;
}

WeightStream::ReadTicket
WeightStream::ticket(uint32_t index) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (device_ == VK_NULL_HANDLE || index >= cfg_.layers) return ReadTicket{};
    return ReadTicket{ semReady_, (uint64_t)index + 1, semDone_, (uint64_t)index + 1 };
}

VkResult
WeightStream::reset()
{
    const VkResult w = waitIdle();
    std::lock_guard<std::mutex> lock(mutex_);
    if (device_ == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;
    if (sliding_)
        for (uint32_t i = 0; i < nextPrefetch_ && i < cfg_.layers; i++)
            bindLocked(i, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, 0);
    // Timelines only go up, so a rewind needs fresh ones rather than a rewound value.
    for (VkSemaphore* s : { &semBound_, &semReady_, &semDone_ }) {
        vkDestroySemaphore(device_, *s, nullptr); *s = VK_NULL_HANDLE;
        if (makeTimeline(device_, s) != VK_SUCCESS) { device_ = VK_NULL_HANDLE;
                                                      return VK_ERROR_INITIALIZATION_FAILED; }
    }
    nextPrefetch_ = 0;
    return w;
}

VkResult
WeightStream::waitIdle()
{
    uint32_t upto;
    { std::lock_guard<std::mutex> lock(mutex_);
      if (device_ == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;
      upto = nextPrefetch_; }
    if (upto) {
        // `ready` is the last thing the transfer queue signals, so it is the honest thing to
        // wait on here. Waiting on `done` would hang whenever the caller stopped reading
        // early, which is a legal thing for a caller to do.
        uint64_t target = upto;
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1; wi.pSemaphores = &semReady_; wi.pValues = &target;
        const VkResult r = vkWaitSemaphores(device_, &wi, 30000000000ull);
        if (r != VK_SUCCESS) return r;
    }
    return backend_ ? backend_->waitIdle() : VK_SUCCESS;
}

WeightStream::Stats
WeightStream::stats() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

} // namespace eva
