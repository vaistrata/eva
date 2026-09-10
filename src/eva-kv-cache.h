#ifndef EVA_KV_CACHE_H
#define EVA_KV_CACHE_H

/////////////////////////////////////////////////////////////////////////////////////////
// KVCache::Impl - eva-runtime.h 의 KVCache 핸들 뒤편.
//
// 이 헤더는 Vulkan 을 안다. eva-runtime.h 는 모르는 채로 남는다 - vai/ 는 eva::Buffer 만
// 보면 되고, 그래서 KV 캐시 배선이 vai 쪽으로 Vulkan 을 새게 하지 않는다.
//
// 구현은 SparseArena 위에 얇게 앉는다. shardSize 를 텐서 하나의 크기로 못박아 두면
// reserve() 가 straddle 을 피하려고 다음 shard 로 밀어내므로, 텐서마다 자기 shard 의
// 오프셋 0 을 받는다. 그래서 디스크립터 범위가 (버퍼, 0, 정확한 크기) 가 된다.
/////////////////////////////////////////////////////////////////////////////////////////

#include "eva-runtime.h"
#include "eva-sparse-arena.h"

#include <memory>
#include <vector>

namespace eva {

struct KVCache::Impl
{
    VkDevice       vkDevice = VK_NULL_HANDLE;
    KVCacheCreateInfo info{};

    std::unique_ptr<SparseArena> arena;

    // 텐서 하나당: 예약한 Region, 그 shard 를 감싼 (비소유) eva::Buffer, 프론티어.
    struct Slot
    {
        Region   region{};
        Buffer   buffer{};
        uint64_t pinnedTokens = 0;
    };
    std::vector<Slot> slots;

    uint64_t rowBytes      = 0;   // headDim * bytesPerElement - 토큰 하나가 head 하나에서 쓰는 폭
    uint64_t headStride    = 0;   // tokensCapacity * rowBytes
    uint64_t tensorBytes   = 0;   // numKVHeads * headStride, 페이지 정렬
    uint32_t tokensPerPage = 0;
    bool     sparse        = false;

    uint64_t growthEvents = 0, refusedGrowths = 0;

    // stage 됐지만 아직 commit 되지 않은 바이트. 예산 검사가 arena 의 resident 만
    // 보면 한 스텝 안에서 여러 텐서를 stage 할 때 전부 통과해 버린다 - resident 는
    // flush 전까지 움직이지 않기 때문이다.
    uint64_t pendingBytes = 0;

    ~Impl();
};

// Device 만 가지고 있는 것(VkDevice, MemoryTopology, DeviceAllocator, sparse 큐)을 받아
// Impl 을 만든다. Device::Impl 은 eva-runtime.cpp 안에만 있고 Impl 정의는 여기에만
// 있으므로, 그 둘을 잇는 경계가 필요하다.
//
// sparse 를 못 쓰는 장치에서도 nullptr 이 아니라 sparse==false 인 Impl 을 돌려준다 -
// 호출자가 분기를 하나만 갖도록. 기하 자체가 불가능할 때만 nullptr 이다.
struct KVCacheFactory
{
    static KVCache::Impl* make(VkDevice, const MemoryTopology&, DeviceAllocator&,
                               VkQueue sparseQueue, const KVCacheCreateInfo&);
};
} // namespace eva

#endif // EVA_KV_CACHE_H
