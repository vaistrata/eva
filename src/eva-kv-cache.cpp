#include "eva-kv-cache.h"

#include <algorithm>
#include <cstdio>
#include <chrono>

namespace eva {

namespace {
// 요청 토큰 수 -> 실제로 붙일 토큰 수. 배수로 올리고 용량에서 자른다.
// ensure() 와 ensureAll() 이 반드시 같은 값을 써야 하므로 한 곳에 둔다.
//
// 배수는 max(페이지 한 장, growGranule) 이다. 페이지보다 잘게 잡을 수는 없다 -
// 한 페이지 안의 두 토큰은 따로 붙을 수 없으므로.
uint64_t wantFor(uint64_t tokens, uint64_t granule, uint32_t perPage, uint64_t cap)
{
    uint64_t T = perPage ? perPage : 1;
    if (granule > T) T = ((granule + T - 1) / T) * T;   // 페이지 배수로 맞춘 granule
    uint64_t want = ((tokens + T - 1) / T) * T;
    if (want > cap) want = cap;
    return want;
}
} // namespace

KVCache::Impl::~Impl()
{
    // 어댑터 버퍼(slots[i].buffer)는 arena 의 shard 를 감싸기만 하므로 여기서 건드리지
    // 않는다. Device::Impl 의 buffers 집합이 external 플래그를 보고 정리한다.
    if (arena) arena->waitIdle();
    arena.reset();
}


KVCache::Impl*
KVCacheFactory::make(VkDevice device, const MemoryTopology& topo, DeviceAllocator& alloc,
                VkQueue sparseQueue, const KVCacheCreateInfo& ci)
{
    if (device == VK_NULL_HANDLE || !ci.numTensors || !ci.tokensCapacity ||
        !ci.numKVHeads || !ci.headDim || !ci.bytesPerElement)
        return nullptr;

    const bool canSparse = topo.sparseBinding() && topo.sparseResidency() &&
                           topo.sparsePageSize() != 0 && sparseQueue != VK_NULL_HANDLE;
    const VkDeviceSize page = canSparse ? topo.sparsePageSize() : (64ull << 10);

    const uint64_t rowBytes   = (uint64_t)ci.headDim * ci.bytesPerElement;
    const uint64_t headStride = ci.tokensCapacity * rowBytes;
    const uint64_t needBytes  = (uint64_t)ci.numKVHeads * headStride;
    const uint64_t tensorBytes = ((needBytes + page - 1) / page) * page;

    // 텐서 하나가 디스크립터 하나로 이름 붙지 못하면 이 클래스는 쓸 수 없다. 오프셋을
    // 배선하는 대신 여기서 물러난다 - 주소 고정이 이 설계의 전부이므로.
    if (tensorBytes > topo.maxStorageBufferRange())
    {
        if (ci.logLevel >= 1)
            std::fprintf(stderr, "[eva] KVCache: 텐서 %.2f MiB > maxStorageBufferRange %.2f MiB\n",
                         tensorBytes / 1048576.0, topo.maxStorageBufferRange() / 1048576.0);
        return nullptr;
    }

    auto* im = new KVCache::Impl();
    im->vkDevice    = device;
    im->info        = ci;
    im->rowBytes    = rowBytes;
    im->headStride  = headStride;
    im->tensorBytes = tensorBytes;
    // head 하나의 프론티어가 페이지 한 장을 채우는 토큰 수. 0 이 되지 않게 최소 1.
    im->tokensPerPage = (uint32_t)std::max<uint64_t>(1, page / std::max<uint64_t>(rowBytes, 1));

    SparseArena::Config acfg{};
    acfg.virtualSize   = tensorBytes * ci.numTensors;
    acfg.shardSize     = tensorBytes;          // 텐서마다 자기 shard 의 오프셋 0 을 받는다
    acfg.usage         = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                       | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                       | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    acfg.preferredTier = MemoryTier::Device;
    acfg.logLevel      = ci.logLevel;

    im->arena = std::make_unique<SparseArena>(
        device, topo, alloc,
        canSparse ? makeVulkanBindBackend(device, sparseQueue) : makeNullBindBackend(),
        acfg);

    if (!im->arena->valid())
    {
        if (ci.logLevel >= 1)
            std::fprintf(stderr, "[eva] KVCache: 아레나 %.2f GiB 예약 실패\n",
                         acfg.virtualSize / 1073741824.0);
        delete im;
        return nullptr;
    }

    im->sparse = !im->arena->committedWhole();

    im->slots.resize(ci.numTensors);
    for (uint32_t i = 0; i < ci.numTensors; i++)
    {
        im->slots[i].region = im->arena->reserve(tensorBytes);
        if (im->slots[i].region.empty())
        {
            if (ci.logLevel >= 1)
                std::fprintf(stderr, "[eva] KVCache: 텐서 %u 예약 실패\n", i);
            delete im;
            return nullptr;
        }
        // shardSize == tensorBytes 이므로 base 는 shard 경계에 정확히 떨어져야 한다.
        // 아니면 range() 가 0 이 아닌 오프셋을 주고, 이 클래스의 전제가 깨진다.
        if (im->slots[i].region.base % tensorBytes != 0)
        {
            if (ci.logLevel >= 1)
                std::fprintf(stderr, "[eva] KVCache: 텐서 %u 가 shard 오프셋 0 이 아니다\n", i);
            delete im;
            return nullptr;
        }
    }

    // 상주가 아니면 아레나가 전부를 미리 붙였다는 뜻이다. 프론티어 개념이 없으므로
    // 처음부터 전 용량이 상주로 잡힌다 - 통계가 절약을 주장하지 않도록.
    if (!im->sparse)
        for (auto& s : im->slots) s.pinnedTokens = ci.tokensCapacity;

    if (ci.logLevel >= 1)
        std::fprintf(stderr,
            "[eva] KVCache: 텐서 %u x %.2f MiB = %.2f GiB 가상, 페이지 %llu KiB, "
            "페이지당 %u 토큰, %s\n",
            ci.numTensors, tensorBytes / 1048576.0, acfg.virtualSize / 1073741824.0,
            (unsigned long long)(im->arena->pageSize() >> 10), im->tokensPerPage,
            im->sparse ? "sparse" : "전부 커밋(폴백)");

    return im;
}



/////////////////////////////////////////////////////////////////////////////////////////
// 핸들 표면
/////////////////////////////////////////////////////////////////////////////////////////
bool     KVCache::sparse() const         { return impl().sparse; }
uint32_t KVCache::tensorCount() const    { return (uint32_t)impl().slots.size(); }
uint64_t KVCache::tensorBytes() const    { return impl().tensorBytes; }
uint64_t KVCache::tokensCapacity() const { return impl().info.tokensCapacity; }
uint32_t KVCache::tokensPerPage() const  { return impl().tokensPerPage; }

Buffer KVCache::buffer(uint32_t tensor) const
{
    EVA_ASSERT(tensor < impl().slots.size());
    return impl().slots[tensor].buffer;
}

KVCache::Grow KVCache::ensure(uint32_t tensor, uint64_t tokens)
{
    Impl& im = impl();
    if (tensor >= im.slots.size())        return Grow::Failed;
    if (tokens > im.info.tokensCapacity)  return Grow::Failed;

    Impl::Slot& s = im.slots[tensor];
    if (!im.sparse)                       return Grow::AlreadyResident;

    // 페이지 단위로 올려 잡는다. 한 페이지 안의 두 토큰은 따로 붙일 수 없으므로
    // 토큰 단위 프론티어는 존재하지 않는다.
    const uint64_t want = wantFor(tokens, im.info.growGranuleTokens,
                                  im.tokensPerPage, im.info.tokensCapacity);
    if (want <= s.pinnedTokens)           return Grow::AlreadyResident;

    const uint64_t addBytes = (want - s.pinnedTokens) * im.rowBytes * im.info.numKVHeads;
    if (im.info.residentBudgetBytes &&
        im.arena->stats().resident + im.pendingBytes + addBytes > im.info.residentBudgetBytes)
    {
        im.refusedGrowths++;
        return Grow::Refused;
    }

    // [Hkv, M, Dh] 이므로 토큰 축이 가운데다. 프론티어가 head 마다 하나씩 있고,
    // head 사이 간격은 M*rowBytes 다. 인접한 head 의 프론티어가 같은 페이지를 걸치면
    // 먼저 붙인 쪽이 붙이고 뒤쪽은 stagePin 이 건너뛴다 - 그래서 M 이 페이지에
    // 정렬되지 않아도 정확하다. 낭비는 텐서당 최대 Hkv 페이지다.
    const uint64_t len = (want - s.pinnedTokens) * im.rowBytes;
    for (uint32_t h = 0; h < im.info.numKVHeads; h++)
    {
        const uint64_t off = (uint64_t)h * im.headStride + s.pinnedTokens * im.rowBytes;
        const MemoryTier landed = im.arena->stagePin(s.region.slice(off, len), MemoryTier::Device);
        if (landed == MemoryTier::_count)
            return Grow::Failed;   // 빈 Region 이거나 진짜 OOM. pinFailures 로 구분한다
    }

    s.pinnedTokens = want;
    im.pendingBytes += addBytes;
    im.growthEvents++;
    return Grow::Ok;
}

KVCache::Grow KVCache::ensureAll(uint64_t tokens)
{
    Impl& im = impl();
    if (!im.sparse)                      return Grow::AlreadyResident;
    if (tokens > im.info.tokensCapacity) return Grow::Failed;

    // 예산은 텐서 하나씩 보면 안 된다. 앞쪽 텐서가 통과하고 뒤쪽이 거절되면 stage 만
    // 된 반쪽 상태가 남고, 그건 다음 commit() 이 예산을 넘겨 커밋해 버린다는 뜻이다.
    // 그래서 전체를 먼저 세고, 안 들어가면 아무것도 stage 하지 않는다.
    const uint64_t want = wantFor(tokens, im.info.growGranuleTokens,
                                  im.tokensPerPage, im.info.tokensCapacity);

    uint64_t total = 0;
    for (const auto& s : im.slots)
        if (want > s.pinnedTokens)
            total += (want - s.pinnedTokens) * im.rowBytes * im.info.numKVHeads;

    if (!total) return Grow::AlreadyResident;

    if (im.info.residentBudgetBytes &&
        im.arena->stats().resident + im.pendingBytes + total > im.info.residentBudgetBytes)
    {
        im.refusedGrowths++;
        return Grow::Refused;
    }

    Grow worst = Grow::AlreadyResident;
    for (uint32_t i = 0; i < (uint32_t)im.slots.size(); i++)
    {
        const Grow g = ensure(i, tokens);
        if (g == Grow::Failed)   return Grow::Failed;
        if (g == Grow::Refused)  worst = Grow::Refused;
        else if (g == Grow::Ok && worst == Grow::AlreadyResident) worst = Grow::Ok;
    }
    return worst;
}

Result KVCache::commit()
{
    Impl& im = impl();
    im.pendingBytes = 0;
    if (!im.arena->hasPendingBinds()) return (Result)VK_SUCCESS;

    const auto t0 = std::chrono::steady_clock::now();
    const VkResult r = im.arena->flush();
    if (r != VK_SUCCESS) return (Result)r;
    const auto t1 = std::chrono::steady_clock::now();

    // 바인드가 끝나기 전에 커널이 그 주소를 읽으면 strict residency 에서 조용히 0 이
    // 나온다 - 에러도 검증 경고도 없다(vkslide 에서 실측). 세마포어를 호출자에게
    // 노출하지 않는 대신 여기서 기다린다. 페이지 교차는 tokensPerPage 스텝마다
    // 한 번뿐이므로 이 대기가 스텝마다 일어나지는 않는다.
    const VkResult w = im.arena->waitIdle();
    const auto t2 = std::chrono::steady_clock::now();

    if (im.info.logLevel >= 2)
        std::fprintf(stderr, "[eva] KVCache::commit flush %.0f us + waitIdle %.0f us\n",
                     std::chrono::duration<double, std::micro>(t1 - t0).count(),
                     std::chrono::duration<double, std::micro>(t2 - t1).count());
    return (Result)w;
}

Result KVCache::releaseAll()
{
    Impl& im = impl();
    if (!im.sparse) return (Result)VK_SUCCESS;
    for (auto& s : im.slots)
    {
        if (!s.pinnedTokens) continue;
        for (uint32_t h = 0; h < im.info.numKVHeads; h++)
            im.arena->stageUnpin(s.region.slice((uint64_t)h * im.headStride,
                                                s.pinnedTokens * im.rowBytes));
        s.pinnedTokens = 0;
    }
    im.pendingBytes = 0;
    if (!im.arena->hasPendingBinds()) return (Result)VK_SUCCESS;
    const VkResult r = im.arena->flush();
    if (r != VK_SUCCESS) return (Result)r;
    return (Result)im.arena->waitIdle();
}

KVCache::Stats KVCache::stats() const
{
    const Impl& im = impl();
    const SparseArena::Stats a = im.arena->stats();
    Stats st{};
    st.reservedBytes   = a.reserved;
    st.residentBytes   = a.resident;
    st.residentPages   = a.residentPages;
    st.bindSubmissions = a.bindSubmissions;
    st.pagesBound      = a.pagesBound;
    st.pagesUnbound    = a.pagesUnbound;
    st.pinFailures     = a.pinFailures;
    st.growthEvents    = im.growthEvents;
    st.refusedGrowths  = im.refusedGrowths;
    for (const auto& s : im.slots)
        st.pinnedTokensMax = std::max(st.pinnedTokensMax, s.pinnedTokens);
    return st;
}

} // namespace eva
