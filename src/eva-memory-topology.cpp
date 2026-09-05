#include "eva-memory-topology.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>

namespace eva {

/////////////////////////////////////////////////////////////////////////////////////////
// Tier definitions
/////////////////////////////////////////////////////////////////////////////////////////
const char* toString(MemoryTier tier)
{
    switch (tier) {
        case MemoryTier::Device:     return "Device";
        case MemoryTier::DeviceHost: return "DeviceHost";
        case MemoryTier::HostPinned: return "HostPinned";
        case MemoryTier::HostCached: return "HostCached";
        case MemoryTier::_count:     break;
    }
    return "?";
}

VkMemoryPropertyFlags requiredProperties(MemoryTier tier)
{
    switch (tier) {
        case MemoryTier::Device:
            return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        case MemoryTier::DeviceHost:
            return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        case MemoryTier::HostPinned:
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        case MemoryTier::HostCached:
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        case MemoryTier::_count:
            break;
    }
    return 0;
}


/////////////////////////////////////////////////////////////////////////////////////////
// Construction
/////////////////////////////////////////////////////////////////////////////////////////
namespace {

bool listHas(const std::vector<const char*>& list, const char* name)
{
    for (const char* e : list)
        if (e && std::strcmp(e, name) == 0)
            return true;
    return false;
}

// How far a memory type overshoots what a tier asked for. Fewer extra property bits means
// a more exact match, which is what we want to hand out first: a plain DEVICE_LOCAL type
// should be spent before the DEVICE_LOCAL | HOST_VISIBLE one, because the latter is the
// BAR aperture and something else may genuinely need it to be host writable.
uint32_t overshoot(VkMemoryPropertyFlags have, VkMemoryPropertyFlags want)
{
    return (uint32_t)std::popcount((unsigned)(have & ~want));
}

} // namespace


MemoryTopology::MemoryTopology(VkPhysicalDevice physicalDevice,
                               const std::vector<const char*>& enabledDeviceExtensions)
: physicalDevice_(physicalDevice)
{
    if (physicalDevice_ == VK_NULL_HANDLE)
        return;

    hasMemoryBudget_ = listHas(enabledDeviceExtensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    const bool hasDrmExt = listHas(enabledDeviceExtensions, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME);
    const bool hasExtHostMem =
        listHas(enabledDeviceExtensions, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);

    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps_);

    // ---- properties, plus whatever optional chunks are actually enabled ----
    VkPhysicalDeviceDrmPropertiesEXT drmProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT};
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hostMemProps{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};

    void** tail = &props2.pNext;
    if (hasDrmExt)     { *tail = &drmProps;     tail = (void**)&drmProps.pNext; }
    if (hasExtHostMem) { *tail = &hostMemProps; tail = (void**)&hostMemProps.pNext; }

    vkGetPhysicalDeviceProperties2(physicalDevice_, &props2);
    limits_            = props2.properties.limits;
    nonResidentStrict_ = props2.properties.sparseProperties.residencyNonResidentStrict == VK_TRUE;

    if (hasDrmExt) {
        drm_.valid        = true;
        drm_.hasPrimary   = drmProps.hasPrimary == VK_TRUE;
        drm_.hasRender    = drmProps.hasRender  == VK_TRUE;
        drm_.primaryMajor = drmProps.primaryMajor;
        drm_.primaryMinor = drmProps.primaryMinor;
        drm_.renderMajor  = drmProps.renderMajor;
        drm_.renderMinor  = drmProps.renderMinor;
    }
    if (hasExtHostMem) {
        canImportHostPtr_        = true;
        minImportedHostPtrAlign_ = hostMemProps.minImportedHostPointerAlignment;
    }

    // ---- sparse features ----
    VkPhysicalDeviceFeatures features{};
    vkGetPhysicalDeviceFeatures(physicalDevice_, &features);
    sparseBinding_         = features.sparseBinding         == VK_TRUE;
    sparseResidencyBuffer_ = features.sparseResidencyBuffer == VK_TRUE;

    // ---- a queue family that can bind sparse memory ----
    // Prefer one without graphics: binding is a queue submission, and keeping it off the
    // family that runs compute means a bind batch does not sit behind a dispatch.
    {
        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qCount, families.data());

        uint32_t anySparse = ~0u;
        for (uint32_t i = 0; i < qCount; ++i) {
            if (!(families[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT))
                continue;
            if (anySparse == ~0u)
                anySparse = i;
            if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                sparseBindQueueFamily_ = i;
                break;
            }
        }
        if (sparseBindQueueFamily_ == ~0u)
            sparseBindQueueFamily_ = anySparse;
    }

    // ---- can device memory leave Vulkan as an fd? ----
    // Asked here and not at export time on purpose: VkExportMemoryAllocateInfo has to be
    // present when the memory is allocated, so the allocator needs the answer before it
    // cuts its first slab.
    {
        auto exportable = [&](VkExternalMemoryHandleTypeFlagBits handleType) {
            VkPhysicalDeviceExternalBufferInfo info{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
            info.usage      = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                            | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            info.handleType = handleType;

            VkExternalBufferProperties props{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
            vkGetPhysicalDeviceExternalBufferProperties(physicalDevice_, &info, &props);
            return (props.externalMemoryProperties.externalMemoryFeatures
                    & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0;
        };

        canExportOpaqueFd_ = listHas(enabledDeviceExtensions, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)
                          && exportable(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT);
        canExportDmaBuf_   = listHas(enabledDeviceExtensions, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME)
                          && exportable(VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    }

    // ---- candidate lists, most exact match first ----
    for (uint32_t t = 0; t < (uint32_t)MemoryTier::_count; ++t)
    {
        const VkMemoryPropertyFlags want = requiredProperties((MemoryTier)t);

        std::vector<std::pair<uint32_t, uint32_t>> scored;   // (overshoot, typeIndex)
        for (uint32_t i = 0; i < memProps_.memoryTypeCount; ++i) {
            const VkMemoryPropertyFlags have = memProps_.memoryTypes[i].propertyFlags;
            if ((have & want) == want)
                scored.emplace_back(overshoot(have, want), i);
        }
        // Stable order so the same device always produces the same plan; a residency bug
        // that only reproduces on every third run is not worth debugging.
        std::sort(scored.begin(), scored.end());

        auto& out = candidates_[t];
        out.reserve(scored.size());
        for (auto& [_, index] : scored)
            out.push_back(index);
    }
}


void MemoryTopology::attachDevice(VkDevice device, bool sparseBindingEnabled)
{
    device_ = device;

    if (device_ == VK_NULL_HANDLE || !sparseBinding_ || !sparseBindingEnabled)
        return;

    // The page granularity is not in any limit struct - the only way to learn it is to make
    // a sparse buffer and read back its alignment, which the spec defines to be the sparse
    // block size. Hence a throwaway buffer here rather than a constant somewhere.
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size        = 1ull << 30;   // large enough that no driver rounds the page size up to it
    info.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.flags       = VK_BUFFER_CREATE_SPARSE_BINDING_BIT
                     | (sparseResidencyBuffer_ ? VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT : 0);
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer probe = VK_NULL_HANDLE;
    if (vkCreateBuffer(device_, &info, nullptr, &probe) != VK_SUCCESS)
        return;   // page size stays 0, which callers read as "no sparse path here"

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, probe, &req);
    sparsePageSize_       = req.alignment;
    sparseMemoryTypeBits_ = req.memoryTypeBits;

    vkDestroyBuffer(device_, probe, nullptr);
}


/////////////////////////////////////////////////////////////////////////////////////////
// Queries
/////////////////////////////////////////////////////////////////////////////////////////
const std::vector<uint32_t>& MemoryTopology::candidates(MemoryTier tier) const
{
    static const std::vector<uint32_t> empty{};
    if (tier >= MemoryTier::_count)
        return empty;
    return candidates_[(uint32_t)tier];
}

std::vector<uint32_t> MemoryTopology::candidates(MemoryTier tier, uint32_t memoryTypeBits) const
{
    std::vector<uint32_t> out;
    for (uint32_t index : candidates(tier))
        if (memoryTypeBits & (1u << index))
            out.push_back(index);
    return out;
}

bool MemoryTopology::supports(MemoryTier tier) const
{
    return !candidates(tier).empty();
}

uint32_t MemoryTopology::heapOf(uint32_t memoryTypeIndex) const
{
    return memoryTypeIndex < memProps_.memoryTypeCount
         ? memProps_.memoryTypes[memoryTypeIndex].heapIndex
         : ~0u;
}

VkMemoryPropertyFlags MemoryTopology::propertiesOf(uint32_t memoryTypeIndex) const
{
    return memoryTypeIndex < memProps_.memoryTypeCount
         ? memProps_.memoryTypes[memoryTypeIndex].propertyFlags
         : 0;
}

MemoryTier MemoryTopology::tierOf(uint32_t memoryTypeIndex) const
{
    const VkMemoryPropertyFlags have = propertiesOf(memoryTypeIndex);

    MemoryTier best      = MemoryTier::_count;
    uint32_t   bestScore = ~0u;
    for (uint32_t t = 0; t < (uint32_t)MemoryTier::_count; ++t) {
        const VkMemoryPropertyFlags want = requiredProperties((MemoryTier)t);
        if ((have & want) != want)
            continue;
        const uint32_t score = overshoot(have, want);
        if (score < bestScore) {
            bestScore = score;
            best      = (MemoryTier)t;
        }
    }
    return best;
}


MemoryTopology::Budget MemoryTopology::budget(uint32_t heapIndex) const
{
    Budget out;
    if (heapIndex >= memProps_.memoryHeapCount)
        return out;

    out.heapSize = memProps_.memoryHeaps[heapIndex].size;

    if (!hasMemoryBudget_) {
        // Degrade to "the heap is as big as it says and nothing is used", so callers get a
        // usable number instead of having to know whether the extension is around.
        out.budget = out.heapSize;
        out.used   = 0;
        return out;
    }

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProps{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 props2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &budgetProps};
    vkGetPhysicalDeviceMemoryProperties2(physicalDevice_, &props2);

    out.budget = budgetProps.heapBudget[heapIndex];
    out.used   = budgetProps.heapUsage[heapIndex];

    // Some drivers report a zero budget rather than omitting the heap; treat that as
    // "no opinion" instead of "you may allocate nothing".
    if (out.budget == 0)
        out.budget = out.heapSize;

    return out;
}

MemoryTopology::Budget MemoryTopology::budgetFor(MemoryTier tier) const
{
    const auto& list = candidates(tier);
    return list.empty() ? Budget{} : budget(heapOf(list.front()));
}


/////////////////////////////////////////////////////////////////////////////////////////
// Diagnostics
/////////////////////////////////////////////////////////////////////////////////////////
namespace {

void humanBytes(VkDeviceSize bytes, char* out, size_t n)
{
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    int i = 0;
    double v = (double)bytes;
    while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
    std::snprintf(out, n, "%.2f %s", v, units[i]);
}

void appendFlag(char* buf, size_t n, const char* name)
{
    std::strncat(buf, name, n - std::strlen(buf) - 1);
    std::strncat(buf, " ", n - std::strlen(buf) - 1);
}

void describeFlags(VkMemoryPropertyFlags f, char* out, size_t n)
{
    out[0] = '\0';
    if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)     appendFlag(out, n, "DEVICE_LOCAL");
    if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)     appendFlag(out, n, "HOST_VISIBLE");
    if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)    appendFlag(out, n, "HOST_COHERENT");
    if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)      appendFlag(out, n, "HOST_CACHED");
    if (f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) appendFlag(out, n, "LAZY");
    if (out[0] == '\0') std::strncpy(out, "-", n);
}

} // namespace


void MemoryTopology::dump() const
{
    if (!valid()) {
        std::printf("[eva::MemoryTopology] invalid\n");
        return;
    }

    char b0[32], b1[32], b2[32], flags[128];

    std::printf("[eva::MemoryTopology]\n");

    std::printf("  heaps (%u):\n", memProps_.memoryHeapCount);
    for (uint32_t i = 0; i < memProps_.memoryHeapCount; ++i) {
        const Budget b = budget(i);
        humanBytes(b.heapSize, b0, sizeof b0);
        humanBytes(b.budget,   b1, sizeof b1);
        humanBytes(b.used,     b2, sizeof b2);
        std::printf("    heap %u: %-10s %-13s budget %-10s used %-10s\n",
            i, b0,
            (memProps_.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL" : "host",
            b1, b2);
    }

    std::printf("  types (%u):\n", memProps_.memoryTypeCount);
    for (uint32_t i = 0; i < memProps_.memoryTypeCount; ++i) {
        describeFlags(memProps_.memoryTypes[i].propertyFlags, flags, sizeof flags);
        std::printf("    type %2u -> heap %u  tier %-11s %s\n",
            i, memProps_.memoryTypes[i].heapIndex, toString(tierOf(i)), flags);
    }

    std::printf("  tier preference:\n");
    for (uint32_t t = 0; t < (uint32_t)MemoryTier::_count; ++t) {
        std::printf("    %-11s ", toString((MemoryTier)t));
        const auto& list = candidates_[t];
        if (list.empty()) {
            std::printf("(unsupported)\n");
            continue;
        }
        for (size_t k = 0; k < list.size(); ++k)
            std::printf("%s%u", k ? " > " : "", list[k]);
        std::printf("\n");
    }

    humanBytes(sparsePageSize_, b0, sizeof b0);
    humanBytes(limits_.sparseAddressSpaceSize, b1, sizeof b1);
    humanBytes(limits_.maxStorageBufferRange,  b2, sizeof b2);
    std::printf("  sparse: binding=%s residency=%s page=%s addrSpace=%s strict=%s bindQueueFamily=%u\n",
        sparseBinding_ ? "yes" : "no",
        sparseResidencyBuffer_ ? "yes" : "no",
        sparsePageSize_ ? b0 : "(not probed)",
        b1,
        nonResidentStrict_ ? "yes" : "no",
        sparseBindQueueFamily_);

    std::printf("  limits: maxStorageBufferRange=%s liveBudget=%s\n",
        b2, hasMemoryBudget_ ? "yes" : "no");

    std::printf("  export: opaqueFd=%s dmaBuf=%s importHostPtr=%s\n",
        canExportOpaqueFd_ ? "yes" : "no",
        canExportDmaBuf_   ? "yes" : "no",
        canImportHostPtr_  ? "yes" : "no");

    if (drm_.valid)
        std::printf("  drm node: primary=%lld:%lld render=%lld:%lld\n",
            (long long)drm_.primaryMajor, (long long)drm_.primaryMinor,
            (long long)drm_.renderMajor,  (long long)drm_.renderMinor);
    else
        std::printf("  drm node: (VK_EXT_physical_device_drm not enabled)\n");
}

} // namespace eva
