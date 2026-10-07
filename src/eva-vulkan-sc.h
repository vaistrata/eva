#ifndef EVA_VULKAN_SC_H
#define EVA_VULKAN_SC_H

// Vulkan SC support.
//
// Two builds take part. The record build is the ordinary Vulkan one, run once
// under VK_LAYER_KHRONOS_json_gen with EVA_SC_RECORD_DIR set: the layer writes
// each pipeline as JSON + SPIR-V and the object counts the run needed, and eva
// writes pipelines.txt, the identifier the layer gave every pipeline it created.
// A pipeline cache compiler turns the JSON into pipeline_cache.bin.
//
// The SC build (EVA_VULKAN_SC) links the Vulkan SC loader and reads that folder
// through EVA_SC_DIR: the cache and the object reservation go into device
// creation, and a pipeline is created by its recorded identifier, as SC has no
// shader modules.
//
// eva keeps compiling against the Vulkan headers, so the handful of SC
// structures it needs are stated here rather than pulled from vulkan_sc_core.h,
// which cannot be included alongside vulkan_core.h.

#include <vulkan/vulkan_core.h>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace eva::sc {

using Uuid = std::array<uint8_t, VK_UUID_SIZE>;

// What identifies a pipeline across the two builds: the SPIR-V and everything
// eva passes to the stage. Not the layout, which follows from the SPIR-V.
uint64_t pipelineKey(const uint32_t* spirv, size_t sizeInBytes,
                     const VkSpecializationInfo* spec, uint32_t requiredSubgroupSize);

// Record build: EVA_SC_RECORD_DIR, or null when not recording.
const char* recordDir();

// The GPU the record run ran on (device.txt): its identity, and the architecture
// and core count eva derives from vendor extensions that SC does not have.
struct DeviceIdentity {
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t driverID;
    uint32_t architecture;
    uint32_t coreClusterCount;
};
void recordIdentity(const DeviceIdentity& identity);

// Whether Vulkan SC 1.0 defines the extension. The record run keeps to these,
// so that the device it shows the application, and with it the kernels the
// application picks, is the one the SC run will see.
bool hasExtension(const char* name);
void recordPipeline(uint64_t key, const Uuid& uuid);

#ifdef EVA_VULKAN_SC

#define EVA_VKSC_API_VERSION_1_0 VK_MAKE_API_VERSION(1, 1, 0, 0)

constexpr VkStructureType kStructureTypePhysicalDeviceVulkanSC10Features = (VkStructureType)1000298000;
constexpr VkStructureType kStructureTypeDeviceObjectReservationCreateInfo = (VkStructureType)1000298002;
constexpr VkStructureType kStructureTypeCommandPoolMemoryReservationCreateInfo = (VkStructureType)1000298003;
constexpr VkStructureType kStructureTypePipelinePoolSize = (VkStructureType)1000298005;
constexpr VkStructureType kStructureTypePipelineOfflineCreateInfo = (VkStructureType)1000298010;
constexpr VkStructureType kStructureTypeFaultData = (VkStructureType)1000298007;
constexpr VkStructureType kStructureTypeFaultCallbackInfo = (VkStructureType)1000298008;
constexpr VkPipelineCacheCreateFlags kPipelineCacheReadOnly = 0x00000002u;
constexpr VkPipelineCacheCreateFlags kPipelineCacheUseApplicationStorage = 0x00000004u;

struct PhysicalDeviceVulkanSC10Features {
    VkStructureType sType;
    void*           pNext;
    VkBool32        shaderAtomicInstructions;
};

struct PipelinePoolSize {
    VkStructureType sType;
    const void*     pNext;
    VkDeviceSize    poolEntrySize;
    uint32_t        poolEntryCount;
};

struct DeviceObjectReservationCreateInfo {
    VkStructureType                  sType;
    const void*                      pNext;
    uint32_t                         pipelineCacheCreateInfoCount;
    const VkPipelineCacheCreateInfo* pPipelineCacheCreateInfos;
    uint32_t                         pipelinePoolSizeCount;
    const PipelinePoolSize*          pPipelinePoolSizes;
    uint32_t semaphoreRequestCount;
    uint32_t commandBufferRequestCount;
    uint32_t fenceRequestCount;
    uint32_t deviceMemoryRequestCount;
    uint32_t bufferRequestCount;
    uint32_t imageRequestCount;
    uint32_t eventRequestCount;
    uint32_t queryPoolRequestCount;
    uint32_t bufferViewRequestCount;
    uint32_t imageViewRequestCount;
    uint32_t layeredImageViewRequestCount;
    uint32_t pipelineCacheRequestCount;
    uint32_t pipelineLayoutRequestCount;
    uint32_t renderPassRequestCount;
    uint32_t graphicsPipelineRequestCount;
    uint32_t computePipelineRequestCount;
    uint32_t descriptorSetLayoutRequestCount;
    uint32_t samplerRequestCount;
    uint32_t descriptorPoolRequestCount;
    uint32_t descriptorSetRequestCount;
    uint32_t framebufferRequestCount;
    uint32_t commandPoolRequestCount;
    uint32_t samplerYcbcrConversionRequestCount;
    uint32_t surfaceRequestCount;
    uint32_t swapchainRequestCount;
    uint32_t displayModeRequestCount;
    uint32_t subpassDescriptionRequestCount;
    uint32_t attachmentDescriptionRequestCount;
    uint32_t descriptorSetLayoutBindingRequestCount;
    uint32_t descriptorSetLayoutBindingLimit;
    uint32_t maxImageViewMipLevels;
    uint32_t maxImageViewArrayLayers;
    uint32_t maxLayeredImageViewMipLevels;
    uint32_t maxOcclusionQueriesPerPool;
    uint32_t maxPipelineStatisticsQueriesPerPool;
    uint32_t maxTimestampQueriesPerPool;
    uint32_t maxImmutableSamplersPerDescriptorSetLayout;
};

struct CommandPoolMemoryReservationCreateInfo {
    VkStructureType sType;
    const void*     pNext;
    VkDeviceSize    commandPoolReservedSize;
    uint32_t        commandPoolMaxCommandBuffers;
};

struct PipelineOfflineCreateInfo {
    VkStructureType sType;
    const void*     pNext;
    uint8_t         pipelineIdentifier[VK_UUID_SIZE];
    uint32_t        matchControl;      // VK_PIPELINE_MATCH_CONTROL_APPLICATION_UUID_EXACT_MATCH = 0
    VkDeviceSize    poolEntrySize;
};

// The fault a Vulkan SC implementation reports (VkFaultData): how severe, and
// which kind. Level: 1 critical, 2 recoverable, 3 warning. Type: 2
// implementation, 3 system, 4 physical device, 5 command buffer full,
// 6 invalid API usage.
struct FaultData {
    VkStructureType sType;
    void*           pNext;
    uint32_t        faultLevel;
    uint32_t        faultType;
};

using FaultCallbackFunction = void (VKAPI_PTR*)(VkBool32 unrecordedFaults, uint32_t faultCount, const FaultData* pFaults);

struct FaultCallbackInfo {
    VkStructureType       sType;
    const void*           pNext;
    uint32_t              faultCount;
    FaultData*            pFaults;
    FaultCallbackFunction pfnFaultCallback;
};

// A fault the implementation reported through the callback registered at
// device creation, with the time it arrived (steady_clock, ms since the first
// call of faultClockMs()).
struct ReportedFault {
    double   timeMs;
    uint32_t level;
    uint32_t type;
};
double faultClockMs();
const FaultCallbackInfo& faultCallbackInfo();     // goes into the device's pNext chain
std::vector<ReportedFault> reportedFaults();      // the faults received so far
size_t reportedFaultCount();                      // their number, without the copy
ReportedFault lastReportedFault();                // the latest one; zeros when none arrived

// EVA_SC_INJECT_FAULT=1: makes one call the implementation must reject (a
// pipeline cache from data it was not told about) so that a fault is reported.
void injectFaultIfRequested(VkDevice device);
void injectFault(VkDevice device);                // the same call, unconditionally

// Every pipeline takes an entry of this size from the one pool reserved at
// device creation; it only has to cover the largest pipeline in the cache.
constexpr VkDeviceSize kPipelinePoolEntrySize = VkDeviceSize(64) << 20;

// The folder EVA_SC_DIR names, loaded once. The pointers in `reservation`
// stay valid for the life of the process, as SC requires of the cache data.
struct Assets {
    std::vector<uint8_t>              cacheData;
    VkPipelineCacheCreateInfo         cacheInfo{};
    PipelinePoolSize                  poolSize{};
    DeviceObjectReservationCreateInfo reservation{};
};
const Assets& assets();

constexpr uint32_t kDriverIdVulkanScEmulationOnVulkan = 27;   // VK_DRIVER_ID_VULKAN_SC_EMULATION_ON_VULKAN
const DeviceIdentity& recordedIdentity();

// Identifier recorded for a pipeline key; aborts with the key when the cache
// has no such pipeline, i.e. the run needs one the record run did not create.
Uuid pipelineUuid(uint64_t key);

// vkQueueSubmit2 / vkCmdPipelineBarrier2 are Vulkan 1.3 names and SC 1.0 is a
// Vulkan 1.2 API: they are reached through VK_KHR_synchronization2.
void bindDevice(VkDevice device);

#endif // EVA_VULKAN_SC

} // namespace eva::sc

#endif // EVA_VULKAN_SC_H
