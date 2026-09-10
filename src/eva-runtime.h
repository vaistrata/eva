#ifndef EVA_RUNTIME_H
#define EVA_RUNTIME_H

#include <vector>
#include <map>
#include <string>
#include <variant>
#include <optional>
#include <memory>
#include <tuple>
#include <array>
#include <utility>
#include <cstring>
#include <functional>

#define VULKAN_VERSION_1_3  // TODO: whether to use this or not depends on the system

#include "eva-error.h"
#include "eva-enums.h"


#define EVA_ATTACHMENT_UNUSED              (~0U)
#define EVA_FALSE                          0U
#define EVA_LOD_CLAMP_NONE                 1000.0F
#define EVA_QUEUE_FAMILY_IGNORED           (~0U)
#define EVA_REMAINING_ARRAY_LAYERS         (~0U)
#define EVA_REMAINING_MIP_LEVELS           (~0U)
#define EVA_SUBPASS_EXTERNAL               (~0U)
#define EVA_TRUE                           1U
#define EVA_WHOLE_SIZE                     (~0ULL)
#define EVA_MAX_MEMORY_TYPES               32U
#define EVA_MAX_PHYSICAL_DEVICE_NAME_SIZE  256U
#define EVA_UUID_SIZE                      16U
#define EVA_MAX_EXTENSION_NAME_SIZE        256U
#define EVA_MAX_DESCRIPTION_SIZE           256U
#define EVA_MAX_MEMORY_HEAPS               16U

namespace eva {

typedef uint32_t Bool32;
typedef uint64_t DeviceAddress;
typedef uint64_t DeviceSize;
typedef uint32_t Flags;
typedef uint32_t SampleMask;


class Runtime;
class Device;
class Queue;
class CommandPool;
class CommandBuffer;
class Fence;
class Semaphore;
class TimelineSemaphore;

class ShaderModule;
class ComputePipeline;
class GraphicsPipeline;

class Buffer;
class Image;
class ImageView;
class Sampler;

class DescriptorSetLayout;
class PipelineLayout;
class DescriptorPool;
class DescriptorSet;
class QueryPool;
class KVCache;

class Window;
class RaytracingPipeline;
class AccelerationStructure;


#define VULKAN_FRIENDS \
    friend class Runtime; \
    friend class Device; \
    friend class Queue; \
    friend class CommandPool; \
    friend class CommandBuffer; \
    friend class Fence; \
    friend class Semaphore; \
    friend class TimelineSemaphore; \
    friend class ShaderModule; \
    friend class ComputePipeline; \
    friend class GraphicsPipeline; \
    friend class RaytracingPipeline; \
    friend class Buffer; \
    friend class Image; \
    friend class ImageView; \
    friend class DescriptorSetLayout; \
    friend class PipelineLayout; \
    friend class DescriptorPool; \
    friend class DescriptorSet; \
    friend class QueryPool; \
    friend class KVCache; \
    friend class Window; \
    friend class AccelerationStructure; \
    friend class Submitting;



#define VULKAN_CLASS_COMMON \
    VULKAN_FRIENDS \
    struct Impl; Impl* pImpl; \
    Impl& impl() { return *pImpl; } \
    const Impl& impl() const { return *pImpl; } \


#define VULKAN_CLASS_COMMON2(class_name) \
    VULKAN_FRIENDS \
    struct Impl; \
    Impl** ppImpl; \
public: \
    class_name(class_name::Impl** ppImpl=nullptr) : ppImpl(ppImpl) {} \
    operator bool() const { return ppImpl && *ppImpl; } \
    bool operator==(const class_name&) const = default; \
    void destroy(); \
private: \
    Impl& impl() { return **ppImpl; } \
    const Impl& impl() const { return **ppImpl; } \



struct ShaderModuleCreateInfo;
struct ComputePipelineCreateInfo;
struct RaytracingPipelineCreateInfo;
struct BufferCreateInfo;
struct KVCacheCreateInfo;
struct ImageCreateInfo;
struct ImageViewDesc;
struct SamplerCreateInfo;
struct DescriptorPoolCreateInfo;
struct BufferRange;
struct SemaphoreStage;
struct QueueSelector;
struct BindingInfo;
struct DescriptorSetLayoutDesc;
struct PipelineLayoutDesc;

struct MemoryBarrier;
struct BufferMemoryBarrier;
struct ImageMemoryBarrier;

struct CopyRegion {
    uint64_t bufferOffset=0;
    uint32_t bufferRowLength=0;
    uint32_t bufferImageHeight=0;
    uint32_t offsetX=0;
    uint32_t offsetY=0;
    uint32_t offsetZ=0;
    uint32_t baseLayer=0;
    uint32_t width=0;
    uint32_t height=0;
    uint32_t depth=0;
    uint32_t layerCount=0;
};
struct WindowCreateInfo;



using BarrierInfo = std::variant<MemoryBarrier, BufferMemoryBarrier, ImageMemoryBarrier>;
using Pipeline = std::variant<ComputePipeline, GraphicsPipeline, RaytracingPipeline>;
using Resource = std::variant<Buffer, Image>;
using SubmissionBatchInfo = std::tuple<
    std::vector<SemaphoreStage>, 
    std::vector<CommandBuffer>, 
    std::vector<SemaphoreStage>
>; 


struct BufferDescriptor;
struct ImageDescriptor;
using Descriptor = std::variant<BufferDescriptor, ImageDescriptor, AccelerationStructure>;





/*
* From [Table 69. Required Limits] in the spec:
*/
namespace portable {
    constexpr uint32_t minMemoryMapAlignment = 64;      // min, vkMapMemory() minimum alignment
#ifdef EVA_ENABLE_RAYTRACING
    constexpr uint32_t shaderGroupHandleSize = 32;      // exact, Size of a shader group handle
    constexpr uint32_t shaderGroupBaseAlignment = 64;   // max, Alignment for SBT base addresses
    constexpr uint32_t shaderGroupHandleAlignment = 32; // max, Alignment for SBT record addresses
    constexpr uint32_t maxShaderGroupStride = 4096;     // min, Maximum SBT record size
#endif
}

#ifdef EVA_ENABLE_RAYTRACING
    struct AsBuildSizesInfo;
    struct AsCreateInfo;
    struct AsBuildInfo;
    struct ShaderBindingTable;
    struct ShaderGroupHandle;
#endif



struct DeviceSettings {
    bool enableGraphicsQueues;
    bool enableComputeQueues;
    bool enableTransferQueues;
#ifdef EVA_ENABLE_WINDOW
    bool enableWindow;
#endif
#ifdef EVA_ENABLE_RAYTRACING
    bool enableRaytracing;
#endif
    // bool operator==(const DeviceSettings&) const = default;
    bool operator<=(const DeviceSettings& other) const {
        return (!enableGraphicsQueues || other.enableGraphicsQueues) &&
               (!enableComputeQueues  || other.enableComputeQueues)  &&
               (!enableTransferQueues || other.enableTransferQueues)
#ifdef EVA_ENABLE_WINDOW
               && (!enableWindow      || other.enableWindow)
#endif
#ifdef EVA_ENABLE_RAYTRACING
               && (!enableRaytracing  || other.enableRaytracing)
#endif
               ;
    }
};


#ifdef EVA_ENABLE_PERFORMANCE_QUERY
struct PerformanceCounter {
    std::string name;
    std::string category;
    std::string description;
    PERFORMANCE_COUNTER_UNIT unit;
    PERFORMANCE_COUNTER_STORAGE storage;
    PERFORMANCE_COUNTER_SCOPE scope;
    PERFORMANCE_COUNTER_DESCRIPTION flags;
};

union PerformanceCounterResult {
    int32_t  i32;
    int64_t  i64;
    uint32_t u32;
    uint64_t u64;
    float    f32;
    double   f64;
};
#endif


enum QueueType {
    queue_graphics,
    queue_compute,
    queue_transfer, 
    queue_max,
};


enum class OwnershipTransferOpType {
    none,
    release,
    acquire,
};


class Runtime {
    VULKAN_CLASS_COMMON
    ~Runtime();
    Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Device createDevice(const DeviceSettings& settings);
    
public:
    static Runtime& get();    // singleton pattern
    uint32_t deviceCount() const;
    Device device(int gpuIndex=-1); 
    Device device(DeviceSettings settings);

#ifdef EVA_ENABLE_WINDOW
    Window createWindow(WindowCreateInfo info);
    void destroyWindow(Window window);
#endif
};


class Device {
    VULKAN_CLASS_COMMON2(Device)

public:
    void reportGPUQueueFamilies() const;
    void reportAssignedQueues() const;

    uint32_t queueCount(QueueType type) const;
    bool supportPresent(QueueType type) const;
    Queue queue(QueueType type, uint32_t index=0) const;
    QueueSelector queue(uint32_t index=0) const;

    CommandPool createCommandPool(QueueType type, COMMAND_POOL_CREATE flags=COMMAND_POOL_CREATE::NONE);
    CommandPool setDefalutCommandPool(QueueType type, CommandPool cmdPool);
    CommandBuffer newCommandBuffer(QueueType type, COMMAND_POOL_CREATE poolFlags=COMMAND_POOL_CREATE::NONE);
    std::vector<CommandBuffer> newCommandBuffers(uint32_t count, QueueType type, COMMAND_POOL_CREATE poolFlags=COMMAND_POOL_CREATE::NONE);
    
    Fence createFence(bool signaled=false);
    Result waitFences(std::vector<Fence> fences, bool waitAll, uint64_t timeout=uint64_t(-1));
    void resetFences(std::vector<Fence> fences);
    Semaphore createSemaphore();
    TimelineSemaphore createTimelineSemaphore(uint64_t initialValue=0);
    ShaderModule createShaderModule(const ShaderModuleCreateInfo& info);
    ComputePipeline createComputePipeline(const ComputePipelineCreateInfo& info);

    Buffer createBuffer(const BufferCreateInfo& info) ;

    // KV 캐시 전용 sparse 예약. 지원되지 않으면 유효하지 않은 핸들이 아니라
    // sparse()==false 인 핸들이 돌아온다 - 호출자는 경로를 하나만 기록한다.
    KVCache createKVCache(const KVCacheCreateInfo& info);
    Image createImage(const ImageCreateInfo& info);
    Sampler createSampler(const SamplerCreateInfo& info);
    DescriptorSetLayout createDescriptorSetLayout(DescriptorSetLayoutDesc desc); // call-by-value is ok because at least one copy is necessary for lvalue
    PipelineLayout createPipelineLayout(PipelineLayoutDesc desc);
    DescriptorPool createDescriptorPool(const DescriptorPoolCreateInfo& info);

    // ----- Cooperative matrix capability (queried once at device creation) -----
    // Member order mirrors VkCooperativeMatrixPropertiesKHR (minus sType/pNext).
    struct CooperativeMatrixProperties {
        uint32_t       M, N, K;
        COMPONENT_TYPE aType, bType, cType, resultType;
        bool           saturatingAccumulation;
        SCOPE          scope;
    };

    // True if the cooperativeMatrix (+ vulkanMemoryModel) feature was enabled.
    // Inspect cooperativeMatrixProperties() for a shape the kernel can use.
    bool supportsCooperativeMatrix() const;
    // Every cooperative-matrix shape reported by the device (all type combos).
    const std::vector<CooperativeMatrixProperties>& cooperativeMatrixProperties() const;
    // True if VK_EXT_pipeline_robustness is enabled
    // (ComputePipelineCreateInfo::robustBufferAccess is usable).
    bool supportsPipelineRobustness() const;
    // Device subgroup size (VkPhysicalDeviceSubgroupProperties.subgroupSize).
    uint32_t subgroupSize() const;
    // Supported ComputePipelineCreateInfo::requiredSubgroupSize range
    // (VkPhysicalDeviceSubgroupSizeControlProperties).
    uint32_t minSubgroupSize() const;
    uint32_t maxSubgroupSize() const;
    // True if compute shaders may use the subgroup arithmetic ops
    // (subgroupAdd / Mul / Min / Max and their variants).
    bool supportsSubgroupArithmetic() const;
    // Device identity (VkPhysicalDeviceProperties / VkPhysicalDeviceDriverProperties).
    uint32_t vendorID() const;
    uint32_t deviceID() const;
    DEVICE_TYPE deviceType() const;
    DRIVER_ID driverID() const;
    Architecture architectureID() const;
    // Shader core-cluster count (NVIDIA SM / AMD CU(instead of WGP) / Intel Xe-core); 0 when unknown.
    uint32_t coreClusterCount() const;
    // Max workgroup count a dispatch may use, per grid axis
    // (VkPhysicalDeviceLimits.maxComputeWorkGroupCount).
    std::array<uint32_t, 3> maxComputeWorkGroupCount() const;
    // Max total shared memory one workgroup may declare, in bytes
    // (VkPhysicalDeviceLimits.maxComputeSharedMemorySize).
    uint32_t maxComputeSharedMemorySize() const;

    // ---- device memory ----
    // Buffer memory is suballocated from slabs rather than one vkAllocateMemory per buffer.
    // These expose what that allocator did, which is otherwise invisible: in particular it
    // DEMOTES rather than failing, so a device-local request can silently land in host
    // memory and the only way to notice is the requested-vs-actual columns of the trace.
    //
    // Tracing costs a driver round trip per event, so it is off by default and meant to be
    // switched on around the run you care about.
    void     setMemoryTracing(bool on);
    void     writeMemoryTrace(std::FILE* out) const;   // tab-separated, one event per line
    void     clearMemoryTrace();
    // Bytes committed to the driver, bytes actually handed out, and how many slabs that took.
    // reserved/live is the fragmentation ratio.
    void     memoryUsage(uint64_t* reserved, uint64_t* live, uint32_t* slabs) const;
    // Alignment a storage-buffer descriptor's offset must be a multiple of, in
    // bytes (VkPhysicalDeviceLimits.minStorageBufferOffsetAlignment).
    uint32_t minStorageBufferOffsetAlignment() const;

    // Timestamp Query Pool
    bool supportsTimestampQueries() const;
    QueryPool createTimestampQueryPool(uint32_t queryCount);

#ifdef EVA_ENABLE_PERFORMANCE_QUERY
    // Performance Query (VK_KHR_performance_query)
    bool supportsPerformanceQueries() const;
    std::vector<PerformanceCounter> enumeratePerformanceCounters(QueueType type) const;
    uint32_t getPerformanceQueryPasses(QueueType type, const std::vector<uint32_t>& counterIndices) const;
    QueryPool createPerformanceQueryPool(QueueType type, const std::vector<uint32_t>& counterIndices, uint32_t queryCount=1);
    void acquireProfilingLock(uint64_t timeout = uint64_t(-1));
    void releaseProfilingLock();
#endif

#ifdef EVA_ENABLE_RAYTRACING
    RaytracingPipeline createRaytracingPipeline(const RaytracingPipelineCreateInfo& info);

    uint32_t shaderGroupHandleSize() const;
    uint32_t shaderGroupHandleAlignment() const;
    uint32_t shaderGroupBaseAlignment() const;
    uint32_t asBufferOffsetAlignment() const;
    uint32_t minAccelerationStructureScratchOffsetAlignment() const;
    AsBuildSizesInfo getBuildSizesInfo(const AsBuildInfo& info) const;
    AccelerationStructure createAccelerationStructure(const AsCreateInfo& info) ;
#endif

};


class Queue {
    VULKAN_CLASS_COMMON2(Queue)
    QueueType _type = queue_max;
public:

    QueueType type() const;

    uint32_t queueFamilyIndex() const;

    uint32_t index() const;

    float priority() const;

    Queue submit(
        CommandBuffer cmdBuffer
    );

    Queue submit(
        std::vector<CommandBuffer> cmdBuffers
    );

    Queue submit(
        std::vector<SubmissionBatchInfo>&& batches
    );

    Queue submit(
        std::vector<SubmissionBatchInfo>&& batches,
        std::optional<Fence> fence
    );

    Queue waitIdle();
};


class CommandPool {
    VULKAN_CLASS_COMMON2(CommandPool)
public:

    QueueType type() const;

    std::vector<CommandBuffer> newCommandBuffers(
        uint32_t count
    );

    CommandBuffer newCommandBuffer();
};


class CommandBuffer {
    VULKAN_CLASS_COMMON2(CommandBuffer)
public:

    QueueType type() const;

    uint32_t queueFamilyIndex() const;

    CommandBuffer submit(uint32_t index=0) const;

    Queue lastSubmittedQueue() const;
    
    void wait() const {
        lastSubmittedQueue().waitIdle();
    }

    CommandBuffer reset(bool keepCapacity=false);

    CommandBuffer begin(
        COMMAND_BUFFER_USAGE flags=COMMAND_BUFFER_USAGE::NONE
    );

    CommandBuffer end();

    CommandBuffer bindPipeline(
        Pipeline pipeline
    );

    CommandBuffer bindDescSets(
        PipelineLayout layout, 
        PIPELINE_BIND_POINT bindPoint,
        std::vector<DescriptorSet> descSets, 
        uint32_t firstSet=0
    );

    CommandBuffer bindDescSets(
        std::vector<DescriptorSet> descSets, 
        uint32_t firstSet=0
    );

    CommandBuffer setPushConstants(
        PipelineLayout layout, 
        SHADER_STAGE stageFlags, 
        uint32_t offset, 
        uint32_t size,
        const void* values
    );

    CommandBuffer setPushConstants(
        uint32_t offset, 
        uint32_t size, 
        const void* data
    );

    CommandBuffer barrier(
        std::vector<BarrierInfo> barrierInfos
    );

    CommandBuffer barrier(
        BarrierInfo barrierInfo
    );

	CommandBuffer copyBuffer(
        Buffer src, 
        Buffer dst, 
        uint64_t srcOffset = 0, 
        uint64_t dstOffset = 0, 
        uint64_t size = EVA_WHOLE_SIZE
    );

    CommandBuffer copyBuffer(
        BufferRange src,
        BufferRange dst 
    );

    CommandBuffer copyImage(
        Image src,
        Image dst,
        std::vector<CopyRegion> regions = {}
    );

    CommandBuffer copyBufferToImage(
        BufferRange src, 
        Image dst, 
        std::vector<CopyRegion> regions = {}
    );
    
    CommandBuffer copyImageToBuffer(
        Image src, 
        BufferRange dst, 
        std::vector<CopyRegion> regions = {}
    );

    CommandBuffer dispatch(
        uint32_t groupCountX, 
        uint32_t groupCountY=1, 
        uint32_t groupCountZ=1
    );

    CommandBuffer dispatch2(
        uint32_t numThreadsInX,
        uint32_t numThreadsInY=1,
        uint32_t numThreadsInZ=1
    );

    // Query commands
    CommandBuffer resetQueryPool(
        QueryPool pool,
        uint32_t firstQuery = 0,
        uint32_t queryCount = 0);

    CommandBuffer writeTimestamp(
        PIPELINE_STAGE stage,
        QueryPool pool,
        uint32_t query);

    CommandBuffer beginQuery(
        QueryPool pool,
        uint32_t query);

    CommandBuffer endQuery(
        QueryPool pool,
        uint32_t query);

#ifdef EVA_ENABLE_RAYTRACING
    CommandBuffer traceRays(
        ShaderBindingTable hitGroupSbt,
        uint32_t width,
        uint32_t height = 1,
        uint32_t depth = 1
    );

    CommandBuffer traceRays(
        uint32_t width,
        uint32_t height = 1,
        uint32_t depth = 1
    );

    CommandBuffer buildAccelerationStructures(
        const AsBuildInfo& info
    );

    CommandBuffer buildAccelerationStructures(
        const std::vector<AsBuildInfo>& infos
    );
#endif
};


class Fence {
    VULKAN_CLASS_COMMON2(Fence)
public:

    Result wait(bool autoReset = false, uint64_t timeout=uint64_t(-1)) const;
    void reset() const;
    bool isSignaled() const;
};


class Semaphore {
    VULKAN_CLASS_COMMON2(Semaphore)
public:

    SemaphoreStage operator()(PIPELINE_STAGE stage) const;

};


class TimelineSemaphore : public Semaphore {
public:
    TimelineSemaphore(Semaphore::Impl** ppImpl=nullptr) : Semaphore(ppImpl) {}

    SemaphoreStage operator()(uint64_t value, PIPELINE_STAGE stage=PIPELINE_STAGE::ALL_COMMANDS) const;

    uint64_t value() const;
    Result wait(uint64_t value, uint64_t timeout=uint64_t(-1)) const;
    void signal(uint64_t value) const;
};


class ShaderModule {
    VULKAN_CLASS_COMMON2(ShaderModule)
public:

    bool hasReflect() const;
    void discardReflect() ;
    PipelineLayoutDesc extractPipelineLayoutDesc() const;

    operator uint64_t() const;
};


class GraphicsPipeline {};


// One compiled unit a pipeline turned into. How a pipeline splits is the
// driver's call: a stage each, several stages merged, or the same shader once
// per dispatch width. Statistics and internal representations are per
// executable, not per pipeline, which is why they are indexed against this.
struct PipelineExecutable {
    std::string name;
    std::string description;
    uint32_t subgroupSize;
};


// One number the driver kept from compiling a pipeline. Names, units and which
// numbers exist at all are the driver's own — Intel reports instruction and
// memory-message counts, spill/fill, scratch size and dispatch width — so read
// these as diagnostics, never as something to branch on.
struct PipelineStatistic {
    std::string name;
    std::string description;
    std::string value;   // rendered from whichever of the driver's four formats it used
};


// A compiler-internal form of the pipeline. On the drivers that expose one this
// is the ISA disassembly, which is the only account of register allocation and
// scheduling that is not a guess.
struct PipelineInternalRepresentation {
    std::string name;
    std::string description;
    bool isText;
    std::vector<uint8_t> data;
};


class ComputePipeline {
    VULKAN_CLASS_COMMON2(ComputePipeline)
public:

    PipelineLayout layout() const;
    DescriptorSetLayout descSetLayout(uint32_t setId=0) const;

    // All three are empty unless the pipeline was created with captureStatistics
    // and the device took VK_KHR_pipeline_executable_properties.
    std::vector<PipelineExecutable> executables() const;
    std::vector<PipelineStatistic> statistics() const;
    std::vector<PipelineInternalRepresentation> internalRepresentations() const;
};





class Buffer {
    VULKAN_CLASS_COMMON2(Buffer)
public:
    
    uint8_t* map(
        uint64_t offset=0, 
        uint64_t size=EVA_WHOLE_SIZE
    );
 
    void flush(
        uint64_t offset=0, 
        uint64_t size=EVA_WHOLE_SIZE
    ) const;

    void invalidate(
        uint64_t offset=0, 
        uint64_t size=EVA_WHOLE_SIZE
    ) const;
    
    void unmap();
    uint64_t size() const;
    BUFFER_USAGE usage() const;
    MEMORY_PROPERTY memoryProperties() const;

    DeviceAddress deviceAddress() const;

    BufferRange operator()(
        uint64_t offset=0, 
        uint64_t size=EVA_WHOLE_SIZE
    );

};


/////////////////////////////////////////////////////////////////////////////////////////
// KVCache - 토큰이 늘어나는 만큼만 물리 메모리를 붙이는 KV 캐시 예약.
//
// 왜 Buffer 가 아니라 별도 타입인가: KV 캐시는 [Hkv, M, Dh] 레이아웃이라 토큰 축이
// 가운데 있다. 토큰이 하나 늘면 head 마다 Dh*bytes 만큼 서로 다른 위치에 붙는다 -
// 즉 프론티어가 하나가 아니라 Hkv 개다. 그걸 호출자가 알 필요는 없으니 여기서 감춘다.
//
// 주소는 고정이다. ensure() 가 무엇을 붙이든 buffer(i) 가 돌려주는 핸들과 그 오프셋은
// 바뀌지 않으므로, 기록해 둔 디스크립터와 커맨드 버퍼를 다시 쓸 필요가 없다.
// (측정 근거: NVK 에서 매핑 하나당 0.044~0.047 us, 제출 하한 9.3 us, 784,000 매핑까지 평평)
//
// sparse 를 못 쓰는 장치에서는 sparse()==false 가 되고 예약 전체가 처음부터 물리
// 메모리를 갖는다. 동작은 같고 절약만 사라진다 - 호출자가 분기할 이유는 통계뿐이다.
/////////////////////////////////////////////////////////////////////////////////////////
class KVCache {
    VULKAN_CLASS_COMMON2(KVCache)
    // Impl 은 위 매크로가 private 으로 선언한다. 정의는 eva-kv-cache.h 에 있고 그 안의
    // 팩토리가 Impl 을 이름으로 불러야 하므로, 그 팩토리 하나만 친구로 들인다.
    // (Vulkan 타입이 시그니처에 있어서 여기서 자유 함수로 선언할 수 없다)
    friend struct KVCacheFactory;
public:

    // ensure() 의 결과. Refused/Failed 를 성공과 같은 값으로 뭉개지 않는 이유는,
    // 붙지 않은 페이지를 읽으면 (strict residency 에서) 조용히 0 이 나오기 때문이다.
    enum class Grow {
        Ok,               // 새 페이지를 stage 했다. commit() 이 필요하다
        AlreadyResident,  // 이미 상주. 대부분의 디코드 스텝이 여기서 끝난다
        Refused,          // 예산이 거절했다 - 호출자가 컨텍스트를 줄여야 한다
        Failed,           // 용량 초과 또는 진짜 OOM
    };

    bool     sparse() const;          // false = 처음부터 전부 물리 백킹
    uint32_t tensorCount() const;
    uint64_t tensorBytes() const;     // 텐서 하나의 가상 크기 (= Hkv*M*Dh*bytes, 페이지 정렬)
    uint64_t tokensCapacity() const;  // M
    uint32_t tokensPerPage() const;   // 페이지 한 장이 담는 토큰 수 (head 하나 기준)

    // 디스크립터가 쓸 핸들. 오프셋 0, 크기는 텐서 크기 그대로다.
    Buffer   buffer(uint32_t tensor) const;
    // 그 버퍼 안에서 이 텐서가 시작하는 바이트. tensorsPerShard == 1 이면 항상 0 이다.
    uint64_t bufferOffset(uint32_t tensor) const;
    uint32_t tensorsPerShard() const;

    // tensor 의 프론티어를 tokens 까지 올린다. 제출은 하지 않는다.
    Grow     ensure(uint32_t tensor, uint64_t tokens);
    // 모든 텐서에 대해 ensure(). 하나라도 Ok 면 Ok, 하나라도 실패면 그 실패를 돌려준다.
    Grow     ensureAll(uint64_t tokens);

    // stage 된 것을 한 번의 vkQueueBindSparse 로 커밋하고 완료를 기다린다.
    Result   commit();

    // 프론티어를 0 으로 되돌리고 물리 메모리를 반납한다. 주소는 그대로다.
    Result   releaseAll();

    struct Stats {
        uint64_t reservedBytes = 0;   // 가상 - 항상 tensorCount * tensorBytes
        uint64_t residentBytes = 0;   // 지금 실제로 붙어 있는 것
        uint64_t residentPages = 0;
        uint64_t bindSubmissions = 0;
        uint64_t pagesBound = 0, pagesUnbound = 0;
        uint64_t pinFailures = 0;
        uint64_t growthEvents = 0;    // ensure() 가 실제로 stage 한 횟수
        uint64_t refusedGrowths = 0;
        uint64_t pinnedTokensMax = 0; // 어느 텐서든 가장 앞선 프론티어
    };
    Stats    stats() const;
};



class Image {
    VULKAN_CLASS_COMMON2(Image)
public:

    ImageView view() const;
    ImageView view(ImageViewDesc&& desc) const;

    // operator ImageMemoryBarrier() const;
    // ImageMemoryBarrier operator/(IMAGE_LAYOUT newLayout) const;
    ImageMemoryBarrier operator()(IMAGE_LAYOUT oldLayout, IMAGE_LAYOUT newLayout) const;
};


class ImageView {
    VULKAN_CLASS_COMMON2(ImageView)
public:
};


class Sampler{
    VULKAN_CLASS_COMMON2(Sampler)
public:
};


class DescriptorSetLayout {
    VULKAN_CLASS_COMMON2(DescriptorSetLayout)
public:

    // const VkDescriptorSetLayoutBinding& bindingInfo(
    //     uint32_t bindingId, 
    //     bool exact=true
    // ) const;
        
    // const std::map<uint32_t, VkDescriptorSetLayoutBinding>& bindingInfos() const;
};


class PipelineLayout {
    VULKAN_CLASS_COMMON2(PipelineLayout)
public:

    DescriptorSetLayout descSetLayout(uint32_t setId) const;
};


class DescriptorPool {
    VULKAN_CLASS_COMMON2(DescriptorPool)
public:

    std::vector<DescriptorSet> operator()(
        std::vector<DescriptorSetLayout> layouts
    );

    DescriptorSet operator()(DescriptorSetLayout layout);

    std::vector<DescriptorSet> operator()(
        DescriptorSetLayout layout,
        uint32_t count
    );

    template<typename... Layouts>
    auto operator()(Layouts... layouts) requires (std::is_same_v<Layouts, DescriptorSetLayout> && ...);
};


class DescriptorSet {
    VULKAN_CLASS_COMMON2(DescriptorSet)
public:

    DescriptorSet write(
        std::vector<Descriptor> descriptors, 
        uint32_t startBindingId=0, 
        uint32_t startArrayOffset=0
    );

    DescriptorSet operator=(
        std::vector<DescriptorSet>&& data
    );
};

inline DescriptorSet DescriptorPool::operator()(DescriptorSetLayout layout) 
{
    return (*this)(std::vector<DescriptorSetLayout>{layout})[0];
}

inline std::vector<DescriptorSet> DescriptorPool::operator()(DescriptorSetLayout layout, uint32_t count)
{
    return (*this)(std::vector<DescriptorSetLayout>(count, layout));
}

template<typename... Layouts>
auto DescriptorPool::operator()(Layouts... layouts) requires (std::is_same_v<Layouts, DescriptorSetLayout> && ...)
{
    auto sets = (*this)(std::vector<DescriptorSetLayout>{ layouts... });

    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return std::make_tuple(sets[I]...);
    }(std::index_sequence_for<Layouts...>{});
}


class QueryPool {
    VULKAN_CLASS_COMMON2(QueryPool)
public:
    uint32_t queryCount() const;
    void reset(uint32_t firstQuery = 0, uint32_t queryCount = 0);

    float timestampPeriod() const;   // nanoseconds per tick

    std::vector<uint64_t> getResults(uint32_t firstQuery=0, uint32_t queryCount = 0);
    double getElapsedMs(uint32_t startQuery, uint32_t endQuery);
};



///////////////////////////////////////////////////////////////////////////////
struct BindingInfo {
    uint32_t binding;
    DESCRIPTOR_TYPE descriptorType;
    uint32_t descriptorCount;
    SHADER_STAGE stageFlags;

    BindingInfo& operator|=(BindingInfo&& other) {
        if (binding == other.binding 
            && descriptorType == other.descriptorType 
            && descriptorCount == other.descriptorCount)
            stageFlags |= other.stageFlags;
        else
            throw;
        return *this;
    }
};


struct DescriptorSetLayoutDesc {
    std::map<uint32_t, BindingInfo> bindings;

    DescriptorSetLayoutDesc() = default;

    DescriptorSetLayoutDesc(const DescriptorSetLayoutDesc&) = default;
    
    DescriptorSetLayoutDesc(DescriptorSetLayoutDesc&& other) = default;

    DescriptorSetLayoutDesc& operator|=(DescriptorSetLayoutDesc&& other) 
    {
        for (auto& [rId, rBinding] : other.bindings) 
        {
            auto it = bindings.find(rId);
            if (it != bindings.end()) 
                it->second |= std::move(rBinding);
            else 
                bindings.emplace_hint(it, rId, std::move(rBinding));

        }
        return *this;
    }
};


struct PushConstantRange {
    SHADER_STAGE stageFlags;
    uint32_t offset;
    uint32_t size;

    PushConstantRange() : offset(0), size(0), stageFlags(SHADER_STAGE::NONE) {};

    PushConstantRange(uint32_t offset, uint32_t size, SHADER_STAGE stageFlags=SHADER_STAGE::NONE)
    : offset(offset), size(size), stageFlags(stageFlags) {}

    PushConstantRange(uint32_t size, SHADER_STAGE stageFlags=SHADER_STAGE::NONE)
    : offset(0), size(size), stageFlags(stageFlags) {}

    // PushConstantRange&& operator|=(SHADER_STAGE stageFlags) &&
    // {
    //     this->stageFlags |= stageFlags;
    //     return std::move(*this);
    // }

    PushConstantRange& operator|=(PushConstantRange&& other)
    {
        offset = std::min(offset, other.offset);
        size = std::max(offset + size, other.offset + other.size) - offset;
        stageFlags |= other.stageFlags;
        return *this;
    }
};




struct PipelineLayoutDesc {
    std::map<uint32_t, DescriptorSetLayoutDesc> setLayouts;
    /*
    * [VUID-VkPipelineLayoutCreateInfo-pPushConstantRanges-00292]
    * : Any two elements of pPushConstantRanges must not include the same stage in stageFlags.
    * 
    * std::vector<PushConstantRange> pushConstants;  
    *  - this cannot guarantee the above rule
    * 
    * std::map<SHADER_STAGE, PushConstantRange> pushConstants;
    *  - Each SHADER_STAGE key must contain only one bit, which is difficult to use.
    * 
    * PushConstantRange pushConstant;
    *  - It contains one push constant range across all shader stages in the pipeline.
    *  - It may be faster than the stage-granular version when calling vkCmdPushConstants.
    */
    std::unique_ptr<PushConstantRange> pushConstant; // only one push constant range is allowed

    PipelineLayoutDesc() = default;

    PipelineLayoutDesc(const PipelineLayoutDesc& other) 
    : setLayouts(other.setLayouts)
    , pushConstant(other.pushConstant ? std::make_unique<PushConstantRange>(*other.pushConstant) : nullptr)
    {}
    
    PipelineLayoutDesc(PipelineLayoutDesc&& other) = default;

    PipelineLayoutDesc& operator|=(PipelineLayoutDesc&& other)
    {
        for (auto& [rId, rSetLayout] : other.setLayouts) 
        {
            auto it = setLayouts.find(rId);
            if (it != setLayouts.end()) 
                it->second |= std::move(rSetLayout);
            else 
                setLayouts.emplace_hint(it, rId, std::move(rSetLayout));
        }

        if (other.pushConstant) 
        {
            if (pushConstant)
                *pushConstant |= std::move(*other.pushConstant);
            else
                pushConstant = std::move(other.pushConstant);
        }

        return *this;
    }

};


struct BufferDescriptor {
    Buffer buffer;
    uint64_t offset;
    uint64_t size;

    BufferDescriptor(Buffer buffer) 
    : buffer(buffer), offset(0), size(EVA_WHOLE_SIZE) {}

    BufferDescriptor(BufferRange range);
};


struct ImageDescriptor {
    std::optional<ImageView> imageView;
    std::optional<Sampler> sampler; 
    IMAGE_LAYOUT imageLayout;

    ImageDescriptor(Image image)
    : imageView(image.view())
    , imageLayout(IMAGE_LAYOUT::MAX_ENUM) {}

    ImageDescriptor(ImageView imageView)
    : imageView(imageView)
    , imageLayout(IMAGE_LAYOUT::MAX_ENUM) {}
};


inline ImageDescriptor&& operator/(ImageDescriptor&& image, Sampler sampler) 
{
    // EVA_ASSERT(!image.sampler);
    image.sampler = sampler;
    return std::move(image);
}

inline ImageDescriptor&& operator/(ImageDescriptor&& image, IMAGE_LAYOUT layout) 
{
    // EVA_ASSERT(image.imageLayout == IMAGE_LAYOUT::MAX_ENUM);
    image.imageLayout = layout;
    return std::move(image);
}


inline std::vector<DescriptorSet> operator,(DescriptorSet lhs, DescriptorSet rhs)
{
    return {lhs, rhs};
}



struct SpvBlob {
    std::shared_ptr<uint32_t[]> data;
    size_t sizeInBytes; // in bytes

    static SpvBlob readFrom(const char* filepath);
};


struct ShaderModuleCreateInfo {
    SHADER_STAGE stage;
    const SpvBlob& spv;
    bool withSpirvReflect = true;
};

using ShaderInput = std::variant<SpvBlob, ShaderModule>;


template <uint32_t ID, typename T>
struct ConstantID {
    T value;
    ConstantID(T v) : value(v) {}
};

// template <uint32_t ID>
// struct ConstantID<ID, bool> {
//     uint32_t value;
//     ConstantID(bool v) : value(v ? 1u : 0u) {}
// };


template<int ID, class T>
inline auto constant_id(T v)
{
    return ConstantID<ID, T>{v};
}

/*
* VUID-VkSpecializationMapEntry-constantID-00776: 
* If the specialization constant is of type boolean, size must be the byte size of VkBool32.
* And in Vulkan, VkBool32 is defined as uint32_t.
*/
template<int ID>
inline auto constant_id(bool v)
{
    return ConstantID<ID, uint32_t>{ v ? 1u : 0u };
}


struct SpecializationMapEntry {
    uint32_t    constantID;
    uint32_t    offset;
    size_t      size;
};


struct SpecializationInfo {
    uint32_t                        mapEntryCount;
    const SpecializationMapEntry*   pMapEntries;
    size_t                          dataSize;
    const void*                     pData;
};


class SpecializationConstant {
    std::map<uint32_t, std::vector<uint8_t>> orderedConstants; // key: constantID, value: bytes of the constant value

    mutable std::optional<std::vector<SpecializationMapEntry>> cachedMapEntries;
    mutable std::optional<std::vector<uint8_t>> cachedData;
    mutable std::optional<SpecializationInfo> cachedSpecInfo;
    
    void buildCache() const;
public:

    template<uint32_t ID, typename T>
    void addConstant(ConstantID<ID, T> constant) 
    {
        auto it = orderedConstants.find(ID);
        if (it != orderedConstants.end()) throw;

        std::vector<uint8_t> newConstant(sizeof(constant.value));
        std::memcpy(newConstant.data(), &constant.value, sizeof(constant.value));
        // *((T*) newConstant.data()) = constant.value; 
        orderedConstants.emplace_hint(it, ID, std::move(newConstant));

        cachedMapEntries.reset();
        cachedData.reset();
        cachedSpecInfo.reset();
    }

    template<typename... ConstantIDs>
    SpecializationConstant(ConstantIDs... constantIDs) 
    {
        (addConstant(constantIDs), ...);
    }
    
    SpecializationConstant() = default;

    SpecializationConstant(const SpecializationConstant& other) 
    : orderedConstants(other.orderedConstants) {}

    SpecializationConstant(SpecializationConstant&& other) 
    : orderedConstants(std::move(other.orderedConstants)) {}
    
    bool empty() const { return orderedConstants.empty(); }

    const SpecializationInfo* getInfo() const
    {
        if (empty())  return nullptr;
        buildCache();
        return &cachedSpecInfo.value();
    }

    bool operator==(const SpecializationConstant& other) const noexcept 
    {
        return orderedConstants == other.orderedConstants;
    }

    uint64_t hash() const noexcept;
};


struct ShaderStage {
    std::optional<ShaderInput> shader;
    SpecializationConstant specialization;

    ShaderStage() : shader(std::nullopt), specialization() {}
    ShaderStage(ShaderInput shader, SpecializationConstant&& spec ={})
    : shader(shader), specialization(std::move(spec)) {}

    template<typename ShaderType>
    ShaderStage(ShaderType shader, SpecializationConstant&& spec ={})
    : shader(shader), specialization(std::move(spec)) {}

    template<uint32_t ID, typename T>
    ShaderStage operator+(ConstantID<ID, T> constant) &&
    {
        specialization.addConstant(constant);
        return std::move(*this);
    }

    bool operator==(const ShaderStage& other) const noexcept;
};


template<uint32_t ID, typename T>
inline ShaderStage operator+(ShaderInput shader, ConstantID<ID, T> constant)
{
    return ShaderStage(shader) + constant;
}


struct ComputePipelineCreateInfo {
    ShaderStage csStage;
    std::optional<PipelineLayout> layout;
    bool autoLayoutAllowAllStages = false;
    uint32_t requiredSubgroupSize = 0;
    bool robustBufferAccess = false;   // per-pipeline robust storage/uniform buffer access (VK_EXT_pipeline_robustness)
    // Keep the compiler's statistics and internal representations queryable
    // (VK_KHR_pipeline_executable_properties). The spec lets a driver compile
    // differently when this is set, so leave it off for anything being timed.
    bool captureStatistics = false;
};


struct BufferCreateInfo {
    uint64_t size;
    BUFFER_USAGE usage;
    MEMORY_PROPERTY reqMemProps;
};


struct KVCacheCreateInfo {
    // 기하. tensorBytes = numKVHeads * tokensCapacity * headDim * bytesPerElement 이고,
    // 이걸 페이지 크기로 올림한 값이 텐서 하나가 차지하는 가상 공간이다.
    uint64_t tokensCapacity = 0;      // M - 이 예약이 담을 수 있는 최대 토큰 수
    uint32_t numKVHeads     = 0;      // Hkv
    uint32_t headDim        = 0;      // Dh
    uint32_t bytesPerElement = 2;     // fp16
    uint32_t numTensors     = 0;      // 보통 2 * 레이어 수 (K/V 각각)

    // 프론티어를 이 토큰 배수로 올려 붙인다. 0 = 페이지 한 장 단위(가장 촘촘).
    //
    // 왜 있는가: commit() 비용은 vkQueueBindSparse 자체(~200 us, 안정적)가 아니라
    // 그 뒤의 host wait 가 지배하고, 그것이 같은 바이너리에서 21 us ~ 4.2 ms 로
    // 흔들린다(실측). 그래서 줄여야 하는 것은 한 번의 비용이 아니라 횟수다.
    //
    // 주의 - 이것은 "미리 앞서 붙이기"가 아니다. 요청에 상수를 더하는 방식은 페이지
    // 경계의 위상만 옮기고 교차 빈도를 전혀 줄이지 않는다(실측: 상수 1024 를 더해도
    // commit 13회 그대로). 빈도를 줄이는 것은 배수를 키우는 것뿐이다.
    //
    // 대가는 최대 이 토큰 수만큼 미리 상주하는 것이다. llama3-3B 기하에서 1024 토큰이면
    // 약 114 MiB - 14 GiB 예약의 0.8% 다.
    uint64_t growGranuleTokens = 0;

    // 한 sparse 버퍼(shard)에 텐서를 몇 개 담을지. 0 = maxStorageBufferRange 가 허용하는
    // 최대치.
    //
    // 이것이 성장 비용을 지배한다. vkQueueBindSparse 한 번의 실행 시간은 매핑 수보다
    // **버퍼 수**에 붙는다 — 448 매핑을 4 개 이하의 버퍼에 담으면 0.6 us, 5 개부터
    // 버퍼당 약 6 us 로 선형이다(vkbindwait 실측, 4→5 에서 계단). 텐서마다 버퍼를
    // 하나씩 두면 llama3-3B 는 56 버퍼가 되어 343 us 가 된다.
    //
    // 1 = 텐서마다 버퍼 하나(이전 동작). 대가는 디스크립터 범위가 자기 텐서를 넘어
    // 같은 shard 의 뒤쪽 텐서까지 덮게 되는 것이다 — 경계 검사가 그만큼 느슨해진다.
    uint32_t tensorsPerShard = 0;

    // ensure() 가 이 값을 넘겨 붙이려 하면 Refused 를 돌려준다. 0 = 제한 없음.
    // heapBudget 을 여기 쓰지 마라 - NVK 는 힙을 넘겨 받아 주고 heapUsage 로 그걸
    // 그대로 보고한다(측정됨). 배치 근거로 쓸 수 있는 값이 아니다.
    uint64_t residentBudgetBytes = 0;

    uint32_t logLevel = 0;
};


struct ImageCreateInfo {
    IMAGE_CREATE flags = IMAGE_CREATE::NONE;
    FORMAT format;
    struct Extent {
        uint32_t width;
        uint32_t height = 1;
        uint32_t depth = 1;
    } extent;
    uint32_t arrayLayers = 1;
    IMAGE_USAGE usage;
    bool preInitialized = false; // if true, initialLayout is VK_IMAGE_LAYOUT_PREINITIALIZED, else VK_IMAGE_LAYOUT_UNDEFINED
    MEMORY_PROPERTY reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL;
};


struct ComponentMapping {
    COMPONENT_SWIZZLE r = COMPONENT_SWIZZLE::IDENTITY;
    COMPONENT_SWIZZLE g = COMPONENT_SWIZZLE::IDENTITY;
    COMPONENT_SWIZZLE b = COMPONENT_SWIZZLE::IDENTITY;
    COMPONENT_SWIZZLE a = COMPONENT_SWIZZLE::IDENTITY;
};


struct ImageViewDesc {
    IMAGE_VIEW_TYPE viewType = IMAGE_VIEW_TYPE::MAX_ENUM;
    FORMAT format = FORMAT::MAX_ENUM;
    ComponentMapping components = {};
    // VkImageSubresourceRange subresourceRange;

    bool operator==(const ImageViewDesc& other) const {
        return viewType == other.viewType &&
               format   == other.format   &&
               components.r == other.components.r &&
               components.g == other.components.g &&
               components.b == other.components.b &&
               components.a == other.components.a;
    }
};


struct SamplerCreateInfo {
    FILTER magFilter = FILTER::LINEAR;
    FILTER minFilter = FILTER::LINEAR;
    SAMPLER_MIPMAP_MODE mipmapMode = SAMPLER_MIPMAP_MODE::LINEAR;
    SAMPLER_ADDRESS_MODE addressModeU = SAMPLER_ADDRESS_MODE::REPEAT;
    SAMPLER_ADDRESS_MODE addressModeV = SAMPLER_ADDRESS_MODE::REPEAT;
    SAMPLER_ADDRESS_MODE addressModeW = SAMPLER_ADDRESS_MODE::REPEAT;
    float mipLodBias = 0.0f;
    Bool32 anisotropyEnable = EVA_FALSE;
    float maxAnisotropy = 1.0f;
    Bool32 compareEnable = EVA_FALSE;
    COMPARE_OP compareOp = COMPARE_OP::ALWAYS;
    float minLod = 0.0f;
    float maxLod = EVA_LOD_CLAMP_NONE;
    BORDER_COLOR borderColor = BORDER_COLOR::INT_OPAQUE_BLACK;
    Bool32 unnormalizedCoordinates = EVA_FALSE;
};


inline ImageView Image::view() const
{
    return view(ImageViewDesc{});
}


struct DescriptorPoolSize {
    DESCRIPTOR_TYPE type;
    uint32_t descriptorCount;
};


constexpr inline DescriptorPoolSize operator<=(
    DESCRIPTOR_TYPE type, int count)
{
    return {
        .type = type,
        .descriptorCount = (uint32_t)count,
    };
}


struct DescriptorPoolCreateInfo {
    std::vector<DescriptorPoolSize> maxTypes;
    uint32_t maxSets;
};



struct QueueSelector {
    const Device device;
    const uint32_t index;

    QueueSelector(Device device, uint32_t index) 
    : device(device), index(index) {} 

    Queue operator()(CommandBuffer cmdBuffer) const
    {
        return device.queue(cmdBuffer.type(), index);
    }

    Queue submit(CommandBuffer cmdBuffer) const
    {
        return (*this)(cmdBuffer).submit(cmdBuffer);
    }

    Queue submit(std::vector<CommandBuffer> cmdBuffers) const
    {
        return (*this)(cmdBuffers[0]).submit(std::move(cmdBuffers));
    }

    Queue submit(std::vector<SubmissionBatchInfo>&& batches, std::optional<Fence> fence = std::nullopt) const
    {
        return (*this)(std::get<1>(batches[0])[0]).submit(std::move(batches), fence);
    }
};




/*
버퍼 range class가 꼭 필요한가?
버퍼 range 필요 시점:
- vkMapMemory
- vkFlushMappedMemoryRanges, vkInvalidateMappedMemoryRanges
- vkCmdCopyBuffer, vkCmdUpdateBuffer, vkCmdFillBuffer 
- VkDescriptorBufferInfo (vkUpdateDescriptorSets의 인자)
- VkBufferMemoryBarrier 
- VkBufferViewCreateInfo 
*/
struct BufferRange {
    Buffer buffer;
    const uint64_t offset;
    const uint64_t size;

    BufferRange() 
    : buffer({})
    , offset(0)
    , size(0) {};
    
    // The null-buffer guard is load-bearing: Buffer::size() derefs an impl a null
    // buffer does not have. A null Buffer is how callers spell an absent binding.
    BufferRange(Buffer buffer,
        uint64_t offset=0,
        uint64_t size=EVA_WHOLE_SIZE)
    : buffer(buffer)
    , offset(offset)
    , size(!buffer ? 0 : (size==EVA_WHOLE_SIZE ? buffer.size() - offset : size)) {}

    BufferRange(const BufferRange&) = default;
    BufferRange(BufferRange&& other) = default;
    
    BufferRange& operator=(const BufferRange& other)
    {
        new (this) BufferRange(other);
        return *this;
    }

    BufferRange& operator=(BufferRange&& other)
    {
        new (this) BufferRange(std::move(other));
        return *this;
    }

    operator bool() const
    {
        return size != 0;
    }

    // Sub-range. offset is relative to this range's start, not the buffer's.
    BufferRange operator()(uint64_t offset, uint64_t size = EVA_WHOLE_SIZE) const
    {
        EVA_ASSERT(offset <= this->size);
        if (size == EVA_WHOLE_SIZE)
            size = this->size - offset;
        EVA_ASSERT(offset + size <= this->size);

        return {buffer, this->offset + offset, size};
    }

    void flush() const
    {
        buffer.flush(offset, size);
    }

    void invalidate() const
    {
        buffer.invalidate(offset, size);
    }

    BUFFER_USAGE usage() const
    {
        return buffer.usage();
    }

    MEMORY_PROPERTY memoryProperties() const
    {
        return buffer.memoryProperties();
    }

    DeviceAddress deviceAddress() const
    {
        return buffer.deviceAddress() + offset;
    }
};


inline BufferRange Buffer::operator()(uint64_t offset, uint64_t size)
{    
    if (size == EVA_WHOLE_SIZE) 
    {
        // EVA_ASSERT(offset < this->size());
        size = this->size() - offset;
    }
    // else EVA_ASSERT(offset + size <= this->size());

    return {*this, offset, size};
}


inline BufferDescriptor::BufferDescriptor(BufferRange range)
: buffer(range.buffer), offset(range.offset), size(range.size) {}



/*
The old layout must either be VK_IMAGE_LAYOUT_UNDEFINED, or match the
current layout of the image subresource range. If the old layout matches the current layout of the
image subresource range, the transition preserves the contents of that range. If the old layout is
VK_IMAGE_LAYOUT_UNDEFINED, the contents of that range may be discarded.
*/
/*
When transitioning the image to VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR or
VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, there is no need to delay subsequent processing,
or perform any visibility operations (as vkQueuePresentKHR performs automatic
visibility operations). To achieve this, the dstAccessMask member of the
VkImageMemoryBarrier should be 0, and the dstStageMask parameter should be
VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT.
*/
/*
• VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT is equivalent to VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT with
  VkAccessFlags2 set to 0 when specified in the second synchronization scope, but equivalent to
  VK_PIPELINE_STAGE_2_NONE in the first scope.
• VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT is equivalent to VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
  with VkAccessFlags2 set to 0 when specified in the first synchronization scope, but equivalent to
  VK_PIPELINE_STAGE_2_NONE in the second scope.
*/
/*
Accesses to the acceleration structure scratch buffers as identified by the
VkAccelerationStructureBuildGeometryInfoKHR::scratchData buffer device addresses must be
synchronized with the VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR pipeline stage and
an access type of (VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR). Accesses to each
VkAccelerationStructureBuildGeometryInfoKHR::srcAccelerationStructure and
VkAccelerationStructureBuildGeometryInfoKHR::dstAccelerationStructure must be synchronized
with the VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR pipeline stage and an access type
of VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR or
VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, as appropriate.
Accesses to other input buffers as identified by any used values of
VkAccelerationStructureGeometryMotionTrianglesDataNV::vertexData,
VkAccelerationStructureGeometryTrianglesDataKHR::vertexData,
VkAccelerationStructureGeometryTrianglesDataKHR::indexData,
VkAccelerationStructureGeometryTrianglesDataKHR::transformData,
VkAccelerationStructureGeometryAabbsDataKHR::data, and
VkAccelerationStructureGeometryInstancesDataKHR::data must be synchronized with the
VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR pipeline stage and an access type of
VK_ACCESS_SHADER_READ_BIT.
*/
struct SYNC_SCOPE {
    struct T {
        PIPELINE_STAGE stage;
        ACCESS access;
        
        bool operator<(const T& other) const {
            if (stage != other.stage) return stage < other.stage;
            return access < other.access;
        }

        // bool operator==(const T& other) const {
        //     return stage == other.stage && access == other.access;
        // }
    } scope;

    SYNC_SCOPE(T scope) : scope(scope) {}
    SYNC_SCOPE(PIPELINE_STAGE stage) : scope({stage, ACCESS::NONE}) {}

    inline static T NONE                = {PIPELINE_STAGE::NONE, ACCESS::NONE};
    inline static T ALL                 = {PIPELINE_STAGE::ALL_COMMANDS, ACCESS::MEMORY_READ | ACCESS::MEMORY_WRITE};
    inline static T ALL_READ            = {PIPELINE_STAGE::ALL_COMMANDS, ACCESS::MEMORY_READ};
    inline static T ALL_WRITE           = {PIPELINE_STAGE::ALL_COMMANDS, ACCESS::MEMORY_WRITE};
    inline static T COMPUTE_READ        = {PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ};
    inline static T COMPUTE_WRITE       = {PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE};
    inline static T RAYTRACING_READ     = {PIPELINE_STAGE::RAY_TRACING_SHADER, ACCESS::SHADER_READ};
    inline static T RAYTRACING_READ_AS  = {PIPELINE_STAGE::RAY_TRACING_SHADER, ACCESS::ACCELERATION_STRUCTURE_READ};
    inline static T RAYTRACING_WRITE    = {PIPELINE_STAGE::RAY_TRACING_SHADER, ACCESS::SHADER_WRITE};
    inline static T TRANSFER_SRC        = {PIPELINE_STAGE::TRANSFER, ACCESS::TRANSFER_READ};
    inline static T TRANSFER_DST        = {PIPELINE_STAGE::TRANSFER, ACCESS::TRANSFER_WRITE};
    inline static T ASBUILD_READ        = {PIPELINE_STAGE::ACCELERATION_STRUCTURE_BUILD, ACCESS::SHADER_READ};
    inline static T ASBUILD_READ_AS     = {PIPELINE_STAGE::ACCELERATION_STRUCTURE_BUILD, ACCESS::ACCELERATION_STRUCTURE_READ};
    inline static T ASBUILD_WRITE_AS    = {PIPELINE_STAGE::ACCELERATION_STRUCTURE_BUILD, ACCESS::ACCELERATION_STRUCTURE_WRITE};
    inline static T ASBUILD_READ_WRITE_AS  = {PIPELINE_STAGE::ACCELERATION_STRUCTURE_BUILD, ACCESS::ACCELERATION_STRUCTURE_READ | ACCESS::ACCELERATION_STRUCTURE_WRITE};
    inline static T PRESENT_SRC         = {(PIPELINE_STAGE)(uint64_t)-1, (ACCESS)(uint64_t)-1};
};

inline SYNC_SCOPE operator,(PIPELINE_STAGE stage, ACCESS access)
{
    return SYNC_SCOPE::T{stage, access};
}


struct MemoryBarrier {
    SYNC_SCOPE srcMask = SYNC_SCOPE::NONE;
    SYNC_SCOPE dstMask = SYNC_SCOPE::NONE;
};

inline MemoryBarrier operator/(SYNC_SCOPE mask1, SYNC_SCOPE mask2)
{
    return {mask1, mask2};
}

inline MemoryBarrier operator/(PIPELINE_STAGE stage1, PIPELINE_STAGE stage2)
{
    return {stage1, stage2};
}


struct BufferMemoryBarrier {
    SYNC_SCOPE srcMask = SYNC_SCOPE::NONE;
    SYNC_SCOPE dstMask = SYNC_SCOPE::NONE;
    OwnershipTransferOpType opType = OwnershipTransferOpType::none;
    QueueType pairedQueue = queue_max;
    // const Buffer& buffer;
    // uint64_t offset = 0;
    // uint64_t size = EVA_WHOLE_SIZE;
    BufferRange buffer;

    BufferMemoryBarrier(Buffer buffer) : buffer(buffer) {}
    BufferMemoryBarrier(BufferRange buffer) : buffer(buffer) {}
};

inline BufferMemoryBarrier&& operator/(SYNC_SCOPE mask, BufferMemoryBarrier&& barrier)
{
    barrier.srcMask = mask;
    return std::move(barrier);
}

inline BufferMemoryBarrier&& operator/(BufferMemoryBarrier&& barrier, SYNC_SCOPE mask)
{
    barrier.dstMask = mask;
    return std::move(barrier);
}

// inline BufferMemoryBarrier&& operator-(QueueType queueType, BufferMemoryBarrier&& barrier)
// {
//     barrier.opType = OwnershipTransferOpType::acquire;
//     barrier.pairedQueue = queueType;
//     return std::move(barrier);
// }

// inline BufferMemoryBarrier&& operator-(BufferMemoryBarrier&& barrier, QueueType queueType)
// {
//     barrier.opType = OwnershipTransferOpType::release;
//     barrier.pairedQueue = queueType;
//     return std::move(barrier);
// }


struct ImageMemoryBarrier {
    SYNC_SCOPE srcMask = SYNC_SCOPE::NONE;
    SYNC_SCOPE dstMask = SYNC_SCOPE::NONE;
    IMAGE_LAYOUT oldLayout = IMAGE_LAYOUT::UNDEFINED;
    IMAGE_LAYOUT newLayout = IMAGE_LAYOUT::UNDEFINED;
    OwnershipTransferOpType opType = OwnershipTransferOpType::none;
    QueueType pairedQueue = queue_max;
    Image image;
    // VkImageSubresourceRange subresourceRange = {};

    ImageMemoryBarrier(Image image) : image(image) {}
};


// inline Image::operator ImageMemoryBarrier() const 
// { 
//     return {
//         .image = *this,
//     };
// }

// inline ImageMemoryBarrier Image::operator/(IMAGE_LAYOUT newLayout) const 
// { 
//     return {
//         .newLayout = newLayout,
//         .image = *this,
//     };
// }

inline ImageMemoryBarrier Image::operator()(IMAGE_LAYOUT oldLayout, IMAGE_LAYOUT newLayout) const
{ 
    ImageMemoryBarrier barrier(*this);
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    return barrier;
}

inline ImageMemoryBarrier&& operator/(SYNC_SCOPE mask, ImageMemoryBarrier&& barrier)
{
    barrier.srcMask = mask;
    return std::move(barrier);
}

inline ImageMemoryBarrier&& operator/(ImageMemoryBarrier&& barrier, SYNC_SCOPE mask)
{
    barrier.dstMask = mask;
    return std::move(barrier);
}



struct SemaphoreStage {
    const Semaphore sem;
    const PIPELINE_STAGE stage;
    const uint64_t value;

    SemaphoreStage(
        Semaphore sem,
        PIPELINE_STAGE stage=PIPELINE_STAGE::ALL_COMMANDS,
        uint64_t value=0)
    : sem(sem), stage(stage), value(value) {}
};

inline SemaphoreStage Semaphore::operator()(PIPELINE_STAGE stage) const
{
    return {*this, stage};
}

inline SemaphoreStage TimelineSemaphore::operator()(uint64_t value, PIPELINE_STAGE stage) const
{
    return {*this, stage, value};
}


inline std::vector<SemaphoreStage> operator,(SemaphoreStage sem1, SemaphoreStage sem2)
{
    return {sem1, sem2};
}   

inline std::vector<SemaphoreStage>&& operator,(std::vector<SemaphoreStage>&& sems, SemaphoreStage sem)
{
    sems.push_back(sem);
    return std::move(sems);
}

inline std::vector<CommandBuffer> operator,(CommandBuffer cmdBuffer1, CommandBuffer cmdBuffer2)
{
    return {cmdBuffer1, cmdBuffer2};
}

inline std::vector<CommandBuffer>&& operator,(std::vector<CommandBuffer>&& cmdBuffers, CommandBuffer cmdBuffer)
{
    cmdBuffers.push_back(cmdBuffer);
    return std::move(cmdBuffers);
}

inline SubmissionBatchInfo operator/(SemaphoreStage sem, CommandBuffer cmdBuffer)
{
    return {{sem}, {cmdBuffer}, {}};
}

inline SubmissionBatchInfo operator/(std::vector<SemaphoreStage>&& sems, CommandBuffer cmdBuffer)
{
    return {std::move(sems), {cmdBuffer}, {}};
}

inline SubmissionBatchInfo operator/(CommandBuffer cmdBuffer, SemaphoreStage sem)
{
    return {{}, {cmdBuffer}, {sem}};
}

inline SubmissionBatchInfo operator/(CommandBuffer cmdBuffer, std::vector<SemaphoreStage>&& sems)
{
    return {{}, {cmdBuffer}, std::move(sems)};
}

inline SubmissionBatchInfo operator/(SemaphoreStage sem, std::vector<CommandBuffer>&& cmdBuffers)
{
    return {{sem}, std::move(cmdBuffers), {}};
}

inline SubmissionBatchInfo operator/(std::vector<SemaphoreStage>&& sems, std::vector<CommandBuffer>&& cmdBuffers)
{
    return {std::move(sems), std::move(cmdBuffers), {}};
}

inline SubmissionBatchInfo operator/(std::vector<CommandBuffer>&& cmdBuffers, SemaphoreStage sem)
{
    return {{}, std::move(cmdBuffers), {sem}};
}

inline SubmissionBatchInfo operator/(std::vector<CommandBuffer>&& cmdBuffers, std::vector<SemaphoreStage>&& sems)
{
    return {{}, std::move(cmdBuffers), std::move(sems)};
}

inline SubmissionBatchInfo&& operator/(SubmissionBatchInfo&& batch, SemaphoreStage sem)
{
    std::get<2>(batch).push_back(sem);
    return std::move(batch);
}

inline SubmissionBatchInfo&& operator/(SubmissionBatchInfo&& batch, std::vector<SemaphoreStage>&& sems)
{
    std::get<2>(batch) = std::move(sems);
    return std::move(batch);
}

struct Waiting {};
inline void waiting(Waiting w){}

struct Submitting {
private:
    friend Submitting operator<<(Queue queue, SubmissionBatchInfo&& batch);
    friend Submitting operator<<(Queue queue, CommandBuffer cmdBuffer);
    friend Submitting operator<<(Queue queue, std::vector<CommandBuffer>&& cmdBuffers);
    friend Submitting&& operator<<(Submitting&& submitting, SubmissionBatchInfo&& batch);
    friend Submitting&& operator<<(Submitting&& submitting, CommandBuffer cmdBuffer);
    friend Submitting&& operator<<(Submitting&& submitting, std::vector<CommandBuffer>&& cmdBuffers);
    friend void operator<<(Submitting&& submitting, Fence fence);
    friend void operator<<(Submitting&& submitting, void(Waiting));

    Submitting() = delete;
    Submitting(const Submitting&) = delete;
    Submitting(Submitting&&) = delete;

    Submitting(Queue queue, SubmissionBatchInfo&& batch) : queue(queue)
    {
        batches.emplace_back(std::move(batch));
    }
    Queue queue;
    std::vector<SubmissionBatchInfo> batches;
    bool isWaiting = false;
    std::optional<Fence> fence;
    
public:
    ~Submitting() { 
        queue.submit(std::move(batches), fence); 
       
        if (isWaiting) {
            queue.waitIdle();
        }
    }
};


inline Submitting operator<<(Queue queue, CommandBuffer cmdBuffer)
{
    return Submitting(queue, {{}, {cmdBuffer}, {}});
}

inline Submitting operator<<(Queue queue, std::vector<CommandBuffer>&& cmdBuffers)
{
	
    return Submitting(queue, SubmissionBatchInfo{
		    std::vector<SemaphoreStage>{}
		    , std::move(cmdBuffers)
		    , std::vector<SemaphoreStage>{}}
		    );
}

inline Submitting operator<<(Queue queue, SubmissionBatchInfo&& batch)
{
    return Submitting(queue, std::move(batch));
}

inline Submitting operator<<(QueueSelector queueSelector, CommandBuffer cmdBuffer)
{
    return operator<<(queueSelector(cmdBuffer), cmdBuffer);
}

inline Submitting operator<<(QueueSelector queueSelector, std::vector<CommandBuffer>&& cmdBuffers)
{
    return operator<<(queueSelector(cmdBuffers[0]), std::move(cmdBuffers));
}

inline Submitting operator<<(QueueSelector queueSelector, SubmissionBatchInfo&& batch)
{
    return operator<<(queueSelector(std::get<1>(batch)[0]), std::move(batch));
}

inline Submitting&& operator<<(Submitting&& submitting, CommandBuffer cmdBuffer)
{
    submitting.batches.emplace_back(
        std::vector<SemaphoreStage>{}, 
        std::vector<CommandBuffer>{cmdBuffer}, 
        std::vector<SemaphoreStage>{});
    return std::move(submitting);
}

inline Submitting&& operator<<(Submitting&& submitting, std::vector<CommandBuffer>&& cmdBuffers)
{
    submitting.batches.emplace_back(
        std::vector<SemaphoreStage>{}, 
        std::move(cmdBuffers), 
        std::vector<SemaphoreStage>{});
    return std::move(submitting);
}

inline Submitting&& operator<<(Submitting&& submitting, SubmissionBatchInfo&& batch)
{   
    submitting.batches.emplace_back(std::move(batch));
    return std::move(submitting);
}

inline void operator<<(Submitting&& submitting, Fence fence)
{
    submitting.fence = fence;
}

inline void operator<<(Submitting&& submitting, void(Waiting))
{
    submitting.isWaiting = true;
}





#ifdef EVA_ENABLE_WINDOW
struct WindowCreateInfo {
    const char* title;
    uint32_t width;
    uint32_t height;
    bool hidden = false;

    Device device;
    IMAGE_USAGE swapChainImageUsage;
    uint32_t minSwapChainImages = 0;
    FORMAT swapChainImageFormat = FORMAT::B8G8R8A8_SRGB;
    COLOR_SPACE swapChainImageColorSpace = COLOR_SPACE::SRGB_NONLINEAR;
    PRESENT_MODE preferredPresentMode = PRESENT_MODE::FIFO;

    /*
    * prePresentCommandBuffer needs to be allocated:
    * - prePresentCommandPool을 지정하였다면, 우선적으로 그것을 사용하여 생성
    * - prePresentCommandPool이 지정되지 않았다면, device의 prePresentCommandPoolType 타입의 (디바이스에 내제된)기본 커맨드 풀을 사용하여 생성
    */
    CommandPool prePresentCommandPool;
    QueueType prePresentCommandPoolType = queue_graphics; // must be compatible with the present queue family
    COMMAND_POOL_CREATE prePresentCommandPoolFlags = COMMAND_POOL_CREATE::NONE;
};


class Window {
    VULKAN_CLASS_COMMON2(Window)

public:
    const std::vector<Image>& swapChainImages() const;

    void recordPrePresentCommands(std::function<void(CommandBuffer, Image)> recordFunc);

    uint32_t acquireNextImageIndex(Semaphore onNextScImageWritable) const;
    void present(Queue queue, std::vector<Semaphore> waitSemaphore, uint32_t imageIndex) const;

    std::pair<CommandBuffer, Semaphore> getNextPresentingContext(Semaphore onNextScImageWritable) const;
    void present(Queue queue) const;

    bool shouldClose() const;
    void pollEvents() const;

    // Bring window to front and request user attention
    void focus() const;

    void setTitle(const char* title) const;

    // Input callback setters
    void setMouseButtonCallback(void (*callback)(int button, int action, double xpos, double ypos));
    void setKeyCallback(void (*callback)(int key, int action, int mods));
    void setCursorPosCallback(void (*callback)(double xpos, double ypos));
    void setScrollCallback(void (*callback)(double xoffset, double yoffset));
};
#endif // EVA_ENABLE_WINDOW


#ifdef EVA_ENABLE_RAYTRACING
class RaytracingPipeline {
    VULKAN_CLASS_COMMON2(RaytracingPipeline)
public:

    PipelineLayout layout() const;
    DescriptorSetLayout descSetLayout(uint32_t setId0=0) const;
    ShaderGroupHandle getHitGroupHandle(uint32_t groupIndex) const;
    void setHitGroupSbt(ShaderBindingTable sbt);
};


class AccelerationStructure {
    VULKAN_CLASS_COMMON2(AccelerationStructure)
public:
    DeviceAddress deviceAddress() const;
};


struct ShaderGroupHandle {
    uint8_t data[portable::shaderGroupHandleSize];
};


struct HitGroup {
    ShaderStage chitStage;
    ShaderStage ahitStage;
    ShaderStage isecStage;
};


struct RaytracingPipelineCreateInfo {
    ShaderStage rgenStage;
    std::vector<ShaderStage> missStages;
    std::vector<HitGroup> hitGroups;
    uint32_t maxRecursionDepth = 1;
    std::optional<PipelineLayout> layout;
    bool autoLayoutAllowAllStages = false;
};


// directly match VkTransformMatrixKHR
struct TransformMatrix {
    float    matrix[3][4];
};


// directly match VkAccelerationStructureInstanceKHR
struct AccelerationStructureInstance {
    TransformMatrix     transform;
    uint32_t            instanceCustomIndex:24;
    uint32_t            mask:8;
    uint32_t            instanceShaderBindingTableRecordOffset:24;
    GEOMETRY_INSTANCE   flags:8;
    uint64_t            accelerationStructureReference;
};


struct ShaderBindingTable {
    Buffer buffer;
    uint32_t recordSize;
    uint32_t numRecords;
};


struct AABB
{
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
};


struct StridedBuffer {
    BufferRange buffer;
    uint32_t stride;
};


struct AsBuildSizesInfo {
    DeviceSize accelerationStructureSize;
    DeviceSize updateScratchSize;
    DeviceSize buildScratchSize;
};


struct AsCreateInfo {
    ACCELERATION_STRUCTURE_TYPE asType;
    BufferRange internalBuffer;     // offset and size required for AS data storage
    uint64_t size;                  // Although it could be fed via internalBuffer.size, it is explicitly specified to prevent user mistakes
};


struct AsBuildInfo {
    BUILD_ACCELERATION_STRUCTURE buildFlags;
    GEOMETRY_TYPE geometryType;
    std::vector<uint32_t> primitiveCounts; // primitive count for each geometry
    AccelerationStructure srcAs;
    AccelerationStructure dstAs;
    BufferRange scratchBuffer;

    struct Triangles {
        StridedBuffer vertexInput;
        StridedBuffer indexInput;
        std::vector<uint32_t> vertexCounts;

        struct Geometry {
            GEOMETRY flags;
            StridedBuffer vertexInput;
            StridedBuffer indexInput;  // stride must be 0, 2, or 4
            BufferRange transformBuffer;
        };
        std::vector<Geometry> eachGeometry;
    };

    struct Aabbs{
        StridedBuffer aabbInput;

        struct Geometry {
            GEOMETRY flags;
            StridedBuffer aabbInput;
        };
        std::vector<Geometry> eachGeometry;
    };

    struct Instances{
        BufferRange instanceInput;
    };

    using Inputs = std::variant<Triangles, Aabbs, Instances>;
    Inputs inputs;
};
#else
class RaytracingPipeline {};
class AccelerationStructure {};
#endif // EVA_ENABLE_RAYTRACING


} // namespace eva



namespace std {
    template<>
    struct hash<eva::ShaderStage> {
        size_t operator()(const eva::ShaderStage& stage) const noexcept;
    };
} 

inline auto alignTo = [](auto value, auto alignment) -> decltype(value) {
    return (value + (decltype(value))alignment - 1) & ~((decltype(value))alignment - 1);
};


#endif // EVA_RUNTIME_H
