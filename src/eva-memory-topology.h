#ifndef EVA_MEMORY_TOPOLOGY_H
#define EVA_MEMORY_TOPOLOGY_H

#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>

namespace eva {

/////////////////////////////////////////////////////////////////////////////////////////
// MemoryTier - where memory physically lives, from a shader's point of view.
//
// Ordered by distance from the compute cores: a lower tier is faster for the GPU to read,
// a higher tier is cheaper to grow. The allocator takes a tier as a request and reports
// back the tier it actually landed in, so a caller can notice a demotion.
/////////////////////////////////////////////////////////////////////////////////////////
enum class MemoryTier : uint32_t
{
    Device = 0,   // DEVICE_LOCAL, not host visible. Home for weights and activations.
    DeviceHost,   // DEVICE_LOCAL | HOST_VISIBLE. The BAR aperture: CPU writes land in VRAM
                  // with no staging copy. Scarce on some drivers, all of VRAM with ReBAR.
    HostPinned,   // HOST_VISIBLE | HOST_COHERENT. The GPU reads it across the bus.
    HostCached,   // + HOST_CACHED. Readback staging: slow for the GPU, fast for the CPU.
    _count
};

const char*           toString(MemoryTier);
VkMemoryPropertyFlags requiredProperties(MemoryTier);


/////////////////////////////////////////////////////////////////////////////////////////
// MemoryTopology - one physical device's memory layout, sparse capability and live budget.
//
// Replaces the linear "first superset wins" scan that used to pick a memory type. That scan
// could never reach a DEVICE_LOCAL | HOST_VISIBLE type (the BAR aperture sits after the
// plain VRAM types, so the loop always stopped first), had no idea how full a heap already
// was, and failed by assert rather than by falling back.
/////////////////////////////////////////////////////////////////////////////////////////
class MemoryTopology
{
public:
    MemoryTopology() = default;

    // `enabledDeviceExtensions` is what was actually passed to vkCreateDevice, not what the
    // driver advertises: querying a struct whose extension is not enabled yields undefined
    // values, so every optional query below is gated on this list.
    MemoryTopology(VkPhysicalDevice, const std::vector<const char*>& enabledDeviceExtensions);

    // Measures the sparse page granularity, only observable through a real sparse VkBuffer -
    // which needs a device, which in turn needs the feature list this class helps decide. So
    // the two cannot happen in one step. `sparseBindingEnabled` must say what vkCreateDevice
    // was actually given; creating the probe buffer is invalid usage when it was left off.
    void attachDevice(VkDevice, bool sparseBindingEnabled);

    bool valid() const { return physicalDevice_ != VK_NULL_HANDLE; }

    // ---- memory types ----
    // Type indices satisfying `tier`, most exact match first: a type carrying no flags beyond
    // what the tier asked for sorts ahead of one that is also host visible. Device therefore
    // yields plain VRAM before BAR and only lands in BAR once the plain types are full.
    const std::vector<uint32_t>& candidates(MemoryTier) const;
    std::vector<uint32_t>        candidates(MemoryTier, uint32_t memoryTypeBits) const;

    bool                  supports(MemoryTier) const;
    uint32_t              heapOf(uint32_t memoryTypeIndex) const;
    VkMemoryPropertyFlags propertiesOf(uint32_t memoryTypeIndex) const;
    MemoryTier            tierOf(uint32_t memoryTypeIndex) const;

    uint32_t memoryTypeCount() const { return memProps_.memoryTypeCount; }
    uint32_t memoryHeapCount() const { return memProps_.memoryHeapCount; }

    // ---- budget ----
    struct Budget
    {
        VkDeviceSize heapSize = 0;
        VkDeviceSize budget   = 0;   // what the driver says we may use right now
        VkDeviceSize used     = 0;   // what this process has allocated from the heap
        VkDeviceSize free() const { return budget > used ? budget - used : 0; }
    };
    // Live query; the numbers move as other processes come and go, so do not cache. Degrades
    // to (heapSize, heapSize, 0) without VK_EXT_memory_budget so callers never branch on it.
    Budget budget(uint32_t heapIndex) const;
    Budget budgetFor(MemoryTier) const;
    bool   hasLiveBudget() const { return hasMemoryBudget_; }

    // ---- sparse ----
    bool         sparseBinding()   const { return sparseBinding_; }
    bool         sparseResidency() const { return sparseResidencyBuffer_; }
    // Measured, never assumed: 64 KiB on the NVIDIA proprietary driver, 4 KiB on lavapipe.
    // Zero until attachDevice() runs.
    VkDeviceSize sparsePageSize()  const { return sparsePageSize_; }

    // Memory types a sparse buffer may bind to. Routinely far narrower than what an ordinary
    // buffer accepts: on the NVIDIA proprietary driver an ordinary storage buffer takes types
    // 0/1/3/4/5 while a sparse one takes only type 1 - plain device-local VRAM, no BAR and no
    // host memory. So sparse buffers page memory in and out of existence; they do not
    // relocate it to a cheaper tier. Anything that must live off-device needs an ordinary
    // buffer.
    uint32_t sparseMemoryTypeBits() const { return sparseMemoryTypeBits_; }

    // True when `tier` has at least one type a sparse buffer can actually bind. Residency
    // planning must consult this rather than supports(), which only says the tier exists.
    bool sparseCanBack(MemoryTier tier) const
    {
        return !candidates(tier, sparseMemoryTypeBits_).empty();
    }

    // Round up to whole sparse pages. Two regions sharing a page can never be bound or
    // unbound independently, so every arena reservation has to be page-granular.
    VkDeviceSize alignToPage(VkDeviceSize bytes) const
    {
        const VkDeviceSize p = sparsePageSize_;
        return p ? ((bytes + p - 1) / p) * p : bytes;
    }

    VkDeviceSize sparseAddressSpaceSize() const { return limits_.sparseAddressSpaceSize; }
    // True when reads of unbound pages return zero instead of undefined data. When it holds,
    // a residency bug surfaces as zeros rather than as plausible garbage.
    bool         nonResidentStrict() const { return nonResidentStrict_; }
    uint32_t     sparseBindQueueFamily() const { return sparseBindQueueFamily_; }

    // ---- limits that shape the arena ----
    // A single storage buffer binding cannot exceed this, so a virtual arena larger than it
    // must be split into shards and no tensor may straddle a shard boundary.
    VkDeviceSize maxStorageBufferRange() const { return limits_.maxStorageBufferRange; }
    VkDeviceSize minImportedHostPointerAlignment() const { return minImportedHostPtrAlign_; }
    VkDeviceSize nonCoherentAtomSize() const { return limits_.nonCoherentAtomSize; }

    // ---- bridge to a DRM-level view ----
    // The join key between what we do through Vulkan and what a DRM-level tracer sees.
    struct DrmNode
    {
        bool    valid = false, hasPrimary = false, hasRender = false;
        int64_t primaryMajor = -1, primaryMinor = -1, renderMajor = -1, renderMinor = -1;
    };
    const DrmNode& drmNode() const { return drm_; }

    // Whether device memory can leave Vulkan as an fd. Exporting is a create-time decision -
    // VkExportMemoryAllocateInfo must be in the pNext chain at vkAllocateMemory and cannot be
    // applied afterwards - so the allocator needs the answer before it cuts its first slab.
    bool canExportOpaqueFd()    const { return canExportOpaqueFd_; }
    bool canExportDmaBuf()      const { return canExportDmaBuf_; }
    bool canImportHostPointer() const { return canImportHostPtr_; }

    void dump() const;

private:
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;

    VkPhysicalDeviceMemoryProperties memProps_{};
    VkPhysicalDeviceLimits           limits_{};

    std::vector<uint32_t> candidates_[(uint32_t)MemoryTier::_count];

    bool     sparseBinding_ = false, sparseResidencyBuffer_ = false, nonResidentStrict_ = false;
    uint32_t sparseBindQueueFamily_ = ~0u;

    VkDeviceSize sparsePageSize_       = 0;
    uint32_t     sparseMemoryTypeBits_ = 0;

    bool hasMemoryBudget_ = false, canExportOpaqueFd_ = false;
    bool canExportDmaBuf_ = false, canImportHostPtr_  = false;

    VkDeviceSize minImportedHostPtrAlign_ = 0;
    DrmNode      drm_{};
};

} // namespace eva

#endif // EVA_MEMORY_TOPOLOGY_H
