#include "eva-vulkan-sc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace eva::sc {

static uint64_t fnv1a(uint64_t h, const void* data, size_t size)
{
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i)
    {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t pipelineKey(const uint32_t* spirv, size_t sizeInBytes,
                     const VkSpecializationInfo* spec, uint32_t requiredSubgroupSize)
{
    uint64_t h = fnv1a(1469598103934665603ull, spirv, sizeInBytes);
    if (spec)
    {
        for (uint32_t i = 0; i < spec->mapEntryCount; ++i)
        {
            const VkSpecializationMapEntry& e = spec->pMapEntries[i];
            const uint64_t entry[3] = { e.constantID, e.offset, e.size };
            h = fnv1a(h, entry, sizeof(entry));
        }
        h = fnv1a(h, spec->pData, spec->dataSize);
    }
    return fnv1a(h, &requiredSubgroupSize, sizeof(requiredSubgroupSize));
}

const char* recordDir()
{
    static const char* dir = std::getenv("EVA_SC_RECORD_DIR");
    return dir;
}

bool hasExtension(const char* name)
{
    // Of the device extensions eva asks about, the ones the Vulkan SC registry
    // lists (vk.xml, supported="vulkansc").
    static const char* const supported[] = {
        "VK_KHR_synchronization2", "VK_EXT_subgroup_size_control", "VK_EXT_shader_atomic_float",
        "VK_EXT_robustness2", "VK_KHR_performance_query", "VK_KHR_swapchain",
    };
    for (const char* ext : supported)
        if (strcmp(ext, name) == 0)
            return true;
    return false;
}

static std::string hex(const uint8_t* bytes, size_t n)
{
    static const char* digits = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i)
    {
        s += digits[bytes[i] >> 4];
        s += digits[bytes[i] & 15];
    }
    return s;
}

void recordPipeline(uint64_t key, const Uuid& uuid)
{
    // One identifier per key: two pipelines that differ in something the key
    // leaves out would be told apart by the layer and not by eva.
    static std::unordered_map<uint64_t, Uuid> seen;
    auto [it, fresh] = seen.emplace(key, uuid);
    if (!fresh)
    {
        if (it->second != uuid)
        {
            fprintf(stderr, "Fatal: pipeline key %016llx recorded with two identifiers\n", (unsigned long long)key);
            std::abort();
        }
        return;
    }

    std::ofstream out(std::string(recordDir()) + "/pipelines.txt", std::ios::app);
    char keyText[17];
    snprintf(keyText, sizeof(keyText), "%016llx", (unsigned long long)key);
    out << keyText << ' ' << hex(uuid.data(), uuid.size()) << '\n';
}

void recordIdentity(const DeviceIdentity& identity)
{
    std::ofstream out(std::string(recordDir()) + "/device.txt");
    out << identity.vendorID << ' ' << identity.deviceID << ' ' << identity.driverID << ' '
        << identity.architecture << ' ' << identity.coreClusterCount << '\n';
}

#ifdef EVA_VULKAN_SC

static std::string scDir()
{
    const char* dir = std::getenv("EVA_SC_DIR");
    if (!dir)
    {
        fprintf(stderr, "Fatal: EVA_SC_DIR is not set (pipeline_cache.bin, pipelines.txt, object reservation)\n");
        std::abort();
    }
    return dir;
}

static std::vector<uint8_t> readFile(const std::string& path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
    {
        fprintf(stderr, "Fatal: cannot open %s\n", path.c_str());
        std::abort();
    }
    std::vector<uint8_t> data(size_t(in.tellg()));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size()));
    return data;
}

// reservation.txt: "<field> <count>" per line, the object counts of the record
// run (taken from the layer's *_objectResInfo.hpp). EVA_SC_RESERVE_SCALE
// multiplies the request counts; limits ("...Limit", "max...") are kept.
static void loadReservation(DeviceObjectReservationCreateInfo& r)
{
    using Info = DeviceObjectReservationCreateInfo;
    static const std::map<std::string, uint32_t Info::*> fields = {
#define F(name) { #name, &Info::name }
        F(semaphoreRequestCount), F(commandBufferRequestCount), F(fenceRequestCount),
        F(deviceMemoryRequestCount), F(bufferRequestCount), F(imageRequestCount), F(eventRequestCount),
        F(queryPoolRequestCount), F(bufferViewRequestCount), F(imageViewRequestCount),
        F(layeredImageViewRequestCount), F(pipelineCacheRequestCount), F(pipelineLayoutRequestCount),
        F(renderPassRequestCount), F(graphicsPipelineRequestCount), F(computePipelineRequestCount),
        F(descriptorSetLayoutRequestCount), F(samplerRequestCount), F(descriptorPoolRequestCount),
        F(descriptorSetRequestCount), F(framebufferRequestCount), F(commandPoolRequestCount),
        F(samplerYcbcrConversionRequestCount), F(surfaceRequestCount), F(swapchainRequestCount),
        F(displayModeRequestCount), F(subpassDescriptionRequestCount), F(attachmentDescriptionRequestCount),
        F(descriptorSetLayoutBindingRequestCount), F(descriptorSetLayoutBindingLimit),
        F(maxImageViewMipLevels), F(maxImageViewArrayLayers), F(maxLayeredImageViewMipLevels),
        F(maxOcclusionQueriesPerPool), F(maxPipelineStatisticsQueriesPerPool),
        F(maxTimestampQueriesPerPool), F(maxImmutableSamplersPerDescriptorSetLayout),
#undef F
    };

    const char* scaleText = std::getenv("EVA_SC_RESERVE_SCALE");
    const uint32_t scale = scaleText ? uint32_t(std::atoi(scaleText)) : 1u;

    const std::string path = scDir() + "/reservation.txt";
    std::ifstream in(path);
    if (!in)
    {
        fprintf(stderr, "Fatal: cannot open %s\n", path.c_str());
        std::abort();
    }
    std::string name;
    uint32_t count;
    while (in >> name >> count)
    {
        auto it = fields.find(name);
        if (it == fields.end())
        {
            fprintf(stderr, "Fatal: %s names an unknown reservation field %s\n", path.c_str(), name.c_str());
            std::abort();
        }
        const bool isRequest = name.size() > 12 && name.compare(name.size() - 12, 12, "RequestCount") == 0;
        r.*(it->second) = isRequest ? count * scale : count;
    }
}

const Assets& assets()
{
    static Assets* a = [] {
        Assets* a = new Assets;
        a->cacheData = readFile(scDir() + "/pipeline_cache.bin");
        a->cacheInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .flags = kPipelineCacheReadOnly | kPipelineCacheUseApplicationStorage,
            .initialDataSize = a->cacheData.size(),
            .pInitialData = a->cacheData.data(),
        };

        a->reservation.sType = kStructureTypeDeviceObjectReservationCreateInfo;
        loadReservation(a->reservation);
        a->reservation.pipelineCacheCreateInfoCount = 1;
        a->reservation.pPipelineCacheCreateInfos = &a->cacheInfo;
        a->reservation.pipelineCacheRequestCount = 1;

        a->poolSize = {
            .sType = kStructureTypePipelinePoolSize,
            .poolEntrySize = kPipelinePoolEntrySize,
            .poolEntryCount = a->reservation.computePipelineRequestCount,
        };
        a->reservation.pipelinePoolSizeCount = 1;
        a->reservation.pPipelinePoolSizes = &a->poolSize;
        return a;
    }();
    return *a;
}

const DeviceIdentity& recordedIdentity()
{
    static const DeviceIdentity identity = [] {
        const std::string path = scDir() + "/device.txt";
        std::ifstream in(path);
        DeviceIdentity id{};
        if (!(in >> id.vendorID >> id.deviceID >> id.driverID >> id.architecture >> id.coreClusterCount))
        {
            fprintf(stderr, "Fatal: cannot read %s\n", path.c_str());
            std::abort();
        }
        return id;
    }();
    return identity;
}

Uuid pipelineUuid(uint64_t key)
{
    static const std::unordered_map<uint64_t, Uuid> table = [] {
        std::unordered_map<uint64_t, Uuid> t;
        const std::string path = scDir() + "/pipelines.txt";
        std::ifstream in(path);
        if (!in)
        {
            fprintf(stderr, "Fatal: cannot open %s\n", path.c_str());
            std::abort();
        }
        std::string keyText, uuidText;
        while (in >> keyText >> uuidText)
        {
            Uuid uuid{};
            for (size_t i = 0; i < uuid.size(); ++i)
                uuid[i] = uint8_t(std::stoul(uuidText.substr(2 * i, 2), nullptr, 16));
            t.emplace(std::stoull(keyText, nullptr, 16), uuid);
        }
        return t;
    }();

    auto it = table.find(key);
    if (it == table.end())
    {
        fprintf(stderr, "Fatal: no recorded pipeline for key %016llx; the record run did not create it\n",
                (unsigned long long)key);
        std::abort();
    }
    return it->second;
}

static PFN_vkQueueSubmit2KHR queueSubmit2 = nullptr;
static PFN_vkCmdPipelineBarrier2KHR cmdPipelineBarrier2 = nullptr;

void bindDevice(VkDevice device)
{
    queueSubmit2 = (PFN_vkQueueSubmit2KHR)vkGetDeviceProcAddr(device, "vkQueueSubmit2KHR");
    cmdPipelineBarrier2 = (PFN_vkCmdPipelineBarrier2KHR)vkGetDeviceProcAddr(device, "vkCmdPipelineBarrier2KHR");
    if (!queueSubmit2 || !cmdPipelineBarrier2)
    {
        fprintf(stderr, "Fatal: the Vulkan SC device has no VK_KHR_synchronization2\n");
        std::abort();
    }
}

#endif // EVA_VULKAN_SC

} // namespace eva::sc

#ifdef EVA_VULKAN_SC

// The Vulkan functions eva calls that Vulkan SC does not have, so that the rest
// of eva reads the same in both builds.
extern "C" {

VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit2(VkQueue queue, uint32_t submitCount, const VkSubmitInfo2* pSubmits, VkFence fence)
{
    return eva::sc::queueSubmit2(queue, submitCount, pSubmits, fence);
}

VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer commandBuffer, const VkDependencyInfo* pDependencyInfo)
{
    eva::sc::cmdPipelineBarrier2(commandBuffer, pDependencyInfo);
}

// No shader modules in SC: a pipeline comes out of the cache by identifier.
VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule* pShaderModule)
{
    *pShaderModule = VK_NULL_HANDLE;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyShaderModule(VkDevice, VkShaderModule, const VkAllocationCallbacks*) {}

// SC frees these with the device: memory, command pools, descriptor pools and
// query pools have no destroy call.
VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL vkDestroyCommandPool(VkDevice, VkCommandPool, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorPool(VkDevice, VkDescriptorPool, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL vkDestroyQueryPool(VkDevice, VkQueryPool, const VkAllocationCallbacks*) {}

} // extern "C"

#endif // EVA_VULKAN_SC
