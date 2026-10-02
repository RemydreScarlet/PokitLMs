#include "pokitlms/gpu/vulkan_linear.hpp"
#include "pokitlms/vulkan_shaders.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace pokitlms::gpu {
namespace {
using Clock = std::chrono::steady_clock;
std::uint64_t ns_since(Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}
void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string("Vulkan ") + operation + ": " + std::to_string(result));
    }
}
template<class T> T info(VkStructureType type) {
    T value{};
    value.sType = type;
    return value;
}

struct Buffer {
    VkDevice device{};
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    void* mapped{};
    VkDeviceSize capacity{}, allocation{};
    VkDeviceSize atom_size{1};
    VkMemoryPropertyFlags flags{};
    bool coherent{};

    ~Buffer() { reset(); }
    void reset() noexcept {
        if (mapped) vkUnmapMemory(device, memory);
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
        mapped = nullptr;
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        capacity = allocation = 0;
    }
    VkDeviceSize range_size(VkDeviceSize used) const {
        if (!used || used > capacity) throw std::invalid_argument("invalid Vulkan mapped range size");
        const auto rounded = ((used + atom_size - 1) / atom_size) * atom_size;
        return rounded >= allocation ? VK_WHOLE_SIZE : rounded;
    }
    void flush(VkDeviceSize used) const {
        if (coherent) return;
        auto range = info<VkMappedMemoryRange>(VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
        range.memory = memory;
        range.size = range_size(used);
        check(vkFlushMappedMemoryRanges(device, 1, &range), "flush mapped buffer");
    }
    void invalidate(VkDeviceSize used) const {
        if (coherent) return;
        auto range = info<VkMappedMemoryRange>(VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
        range.memory = memory;
        range.size = range_size(used);
        check(vkInvalidateMappedMemoryRanges(device, 1, &range), "invalidate mapped buffer");
    }
};

bool supported_type(std::uint32_t type) {
    switch (type) {
        case 0: case 1: case 2: case 3: case 8: case 12: case 13: case 14: case 30: return true;
        default: return false;
    }
}
}  // namespace

class VulkanLinearBackend::Impl {
public:
    explicit Impl(const VulkanLinearOptions& options) {
        if (options.tile_bytes < 256 || options.tile_bytes > 128U * 1024U * 1024U) {
            throw std::invalid_argument("Vulkan weight tile must be between 256 bytes and 128 MiB");
        }
        try { initialize(options); }
        catch (...) { shutdown(); throw; }
    }
    ~Impl() { shutdown(); }

    void initialize(const VulkanLinearOptions& options) {
        options_vectorized_q4_k = options.use_vectorized_q4_k;
        model_cache_mode = options.model_cache_mode;
        if (model_cache_mode != VulkanModelCacheMode::Automatic &&
            model_cache_mode != VulkanModelCacheMode::Disabled) {
            throw std::invalid_argument("invalid Vulkan model cache mode");
        }
        auto app = info<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
        app.pApplicationName = "PokitLMs";
        app.apiVersion = VK_API_VERSION_1_1;
        auto create = info<VkInstanceCreateInfo>(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
        create.pApplicationInfo = &app;
        check(vkCreateInstance(&create, nullptr, &instance), "create instance");
        std::uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerate devices");
        if (!count) throw std::runtime_error("Vulkan has no physical device");
        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "enumerate devices");
        if (options.device_index != std::numeric_limits<std::uint32_t>::max()) {
            if (options.device_index >= count) throw std::invalid_argument("Vulkan device index out of range");
            physical = devices[options.device_index];
        } else {
            int best = -1;
            for (auto candidate : devices) {
                VkPhysicalDeviceProperties p{};
                vkGetPhysicalDeviceProperties(candidate, &p);
                int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 :
                            p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 0;
                if (score > best) { best = score; physical = candidate; }
            }
        }
        vkGetPhysicalDeviceProperties(physical, &properties);
        if (properties.limits.maxComputeWorkGroupInvocations < 128 ||
            properties.limits.maxComputeWorkGroupSize[0] < 128) {
            throw std::runtime_error("Vulkan device cannot execute 128-thread compute groups");
        }
        vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
        std::uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, families.data());
        queue_family = queue_count;
        for (std::uint32_t i = 0; i < queue_count; ++i) {
            if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && families[i].queueCount) {
                queue_family = i;
                break;
            }
        }
        if (queue_family == queue_count) throw std::runtime_error("Vulkan device has no compute queue");
        timestamp_bits = families[queue_family].timestampValidBits;

        auto subgroup = info<VkPhysicalDeviceSubgroupProperties>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES);
        auto properties2 = info<VkPhysicalDeviceProperties2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
        properties2.pNext = &subgroup;
        auto get_properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2"));
        if (get_properties2) get_properties2(physical, &properties2);
        const auto needed = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        // SPIR-V 1.3 does not itself guarantee that all subgroup lanes are
        // populated. The row/column mapping requires full, fixed-size groups.
        std::uint32_t extension_count = 0;
        check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &extension_count, nullptr), "enumerate extensions");
        std::vector<VkExtensionProperties> extensions(extension_count);
        check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &extension_count, extensions.data()), "enumerate extensions");
        const bool size_control = std::any_of(extensions.begin(), extensions.end(), [](const auto& extension) {
            return std::strcmp(extension.extensionName, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) == 0;
        });
        has_memory_budget = std::any_of(extensions.begin(), extensions.end(), [](const auto& extension) {
            return std::strcmp(extension.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
        });
        auto full = info<VkPhysicalDeviceSubgroupSizeControlFeaturesEXT>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT);
        auto features = info<VkPhysicalDeviceFeatures2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
        features.pNext = &full;
        const auto get_features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2"));
        if (size_control && get_features2) get_features2(physical, &features);
        use_subgroups = options.use_subgroups &&
            (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
            (subgroup.supportedOperations & needed) == needed && subgroup.subgroupSize &&
            subgroup.subgroupSize <= 128 && 128 % subgroup.subgroupSize == 0 &&
            size_control && full.computeFullSubgroups;
        subgroup_size = use_subgroups ? subgroup.subgroupSize : 0;
        rows_per_group = use_subgroups ? 128 / subgroup.subgroupSize : 1;
        description = properties.deviceName;
        description += use_subgroups ? " subgroup=" + std::to_string(subgroup.subgroupSize) : " workgroup=128";

        float priority = 0.5F;
        auto queue_create = info<VkDeviceQueueCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
        queue_create.queueFamilyIndex = queue_family;
        queue_create.queueCount = 1;
        queue_create.pQueuePriorities = &priority;
        auto device_create = info<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
        device_create.queueCreateInfoCount = 1;
        device_create.pQueueCreateInfos = &queue_create;
        const char* size_control_extension = VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME;
        std::array<const char*, 2> enabled_extensions{};
        std::uint32_t enabled_extension_count = 0;
        if (use_subgroups) {
            full.subgroupSizeControl = VK_FALSE;
            full.computeFullSubgroups = VK_TRUE;
            device_create.pNext = &full;
            enabled_extensions[enabled_extension_count++] = size_control_extension;
        }
        if (has_memory_budget)
            enabled_extensions[enabled_extension_count++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
        device_create.enabledExtensionCount = enabled_extension_count;
        device_create.ppEnabledExtensionNames = enabled_extensions.data();
        check(vkCreateDevice(physical, &device_create, nullptr, &device), "create device");
        vkGetDeviceQueue(device, queue_family, 0, &queue);
        tile_bytes = std::min<VkDeviceSize>(options.tile_bytes,
            properties.limits.maxStorageBufferRange - 4) & ~VkDeviceSize(3);
        bool prefer_cached_weights = false;
        switch (options.weight_memory_mode) {
            case VulkanWeightMemoryMode::Automatic:
                prefer_cached_weights = properties.vendorID == 0x10de &&
                    properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
                break;
            case VulkanWeightMemoryMode::Local:
                break;
            case VulkanWeightMemoryMode::Cached:
                prefer_cached_weights = true;
                break;
            default:
                throw std::invalid_argument("invalid Vulkan weight memory mode");
        }
        // Spare bytes allow a uint load containing a final halfword.
        for (auto& buffer : weights) {
            ensure_buffer(buffer, tile_bytes + 4, prefer_cached_weights,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            counters.allocated_bytes += buffer.allocation;
        }
        counters.weight_memory_flags = weights[0].flags;

        auto pool_create = info<VkCommandPoolCreateInfo>(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
        pool_create.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_create.queueFamilyIndex = queue_family;
        check(vkCreateCommandPool(device, &pool_create, nullptr, &command_pool), "create command pool");
        auto commands = info<VkCommandBufferAllocateInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
        commands.commandPool = command_pool;
        commands.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commands.commandBufferCount = 2;
        check(vkAllocateCommandBuffers(device, &commands, command_buffers.data()), "allocate command buffers");
        auto fence_create = info<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
        for (auto& fence : fences) check(vkCreateFence(device, &fence_create, nullptr, &fence), "create fence");
        if (timestamp_bits) {
            auto queries = info<VkQueryPoolCreateInfo>(VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);
            queries.queryType = VK_QUERY_TYPE_TIMESTAMP;
            queries.queryCount = 4;
            check(vkCreateQueryPool(device, &queries, nullptr, &query_pool), "create timestamp pool");
        }

        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        for (std::uint32_t i = 0; i < bindings.size(); ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        auto layout = info<VkDescriptorSetLayoutCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
        layout.bindingCount = bindings.size();
        layout.pBindings = bindings.data();
        check(vkCreateDescriptorSetLayout(device, &layout, nullptr, &descriptor_layout), "create descriptor layout");
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6};
        auto descriptors = info<VkDescriptorPoolCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
        descriptors.maxSets = 2;
        descriptors.poolSizeCount = 1;
        descriptors.pPoolSizes = &pool_size;
        check(vkCreateDescriptorPool(device, &descriptors, nullptr, &descriptor_pool), "create descriptor pool");
        std::array<VkDescriptorSetLayout, 2> layouts{descriptor_layout, descriptor_layout};
        auto sets = info<VkDescriptorSetAllocateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
        sets.descriptorPool = descriptor_pool;
        sets.descriptorSetCount = 2;
        sets.pSetLayouts = layouts.data();
        check(vkAllocateDescriptorSets(device, &sets, descriptor_sets.data()), "allocate descriptors");
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
        auto pipeline_create = info<VkPipelineLayoutCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
        pipeline_create.setLayoutCount = 1;
        pipeline_create.pSetLayouts = &descriptor_layout;
        pipeline_create.pushConstantRangeCount = 1;
        pipeline_create.pPushConstantRanges = &push;
        check(vkCreatePipelineLayout(device, &pipeline_create, nullptr, &pipeline_layout), "create pipeline layout");
        auto shader_create = info<VkShaderModuleCreateInfo>(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
        shader_create.codeSize = use_subgroups ? sizeof(shaders::linear_subgroup) : sizeof(shaders::linear_workgroup);
        shader_create.pCode = use_subgroups ? shaders::linear_subgroup : shaders::linear_workgroup;
        check(vkCreateShaderModule(device, &shader_create, nullptr, &shader), "create shader module");
    }

    void ensure_buffer(Buffer& buffer, VkDeviceSize size, bool prefer_cached = false,
                       VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) {
        size = (size + 3) & ~VkDeviceSize(3);
        if (buffer.capacity >= size) return;
        buffer.reset();
        buffer.device = device;
        auto create = info<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
        create.size = size;
        create.usage = usage;
        create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device, &create, nullptr, &buffer.buffer), "create storage buffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
        int best = -1;
        std::uint32_t memory_type = memory_properties.memoryTypeCount;
        for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            const auto flags = memory_properties.memoryTypes[i].propertyFlags;
            if (!(requirements.memoryTypeBits & (1U << i)) || !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
            int score = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? (prefer_cached ? 2 : 4) : 0) +
                        (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? (prefer_cached ? 4 : 2) : 0) +
                        (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ? 1 : 0);
            if (score > best) { best = score; memory_type = i; }
        }
        if (memory_type == memory_properties.memoryTypeCount) throw std::runtime_error("Vulkan has no mapped storage memory");
        auto allocate = info<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = memory_type;
        check(vkAllocateMemory(device, &allocate, nullptr, &buffer.memory), "allocate mapped memory");
        check(vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0), "bind storage buffer");
        check(vkMapMemory(device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped), "map storage buffer");
        buffer.coherent = memory_properties.memoryTypes[memory_type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        buffer.flags = memory_properties.memoryTypes[memory_type].propertyFlags;
        buffer.atom_size = properties.limits.nonCoherentAtomSize;
        buffer.capacity = size;
        buffer.allocation = requirements.size;
    }

    VkPipeline pipeline(std::uint32_t type, bool vectorized_q4_k) {
        const auto key = type | (vectorized_q4_k ? 0x100U : 0U);
        if (auto found = pipelines.find(key); found != pipelines.end()) return found->second;
        const auto start = Clock::now();
        struct SpecializationData { std::uint32_t type; VkBool32 vectorized_q4_k; } data{
            type, vectorized_q4_k ? VK_TRUE : VK_FALSE};
        const std::array<VkSpecializationMapEntry, 2> entries{{
            {0, offsetof(SpecializationData, type), sizeof(data.type)},
            {1, offsetof(SpecializationData, vectorized_q4_k), sizeof(data.vectorized_q4_k)}}};
        VkSpecializationInfo specialization{static_cast<std::uint32_t>(entries.size()), entries.data(),
                                            sizeof(data), &data};
        auto stage = info<VkPipelineShaderStageCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";
        stage.pSpecializationInfo = &specialization;
        if (use_subgroups) stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
        auto create = info<VkComputePipelineCreateInfo>(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);
        create.stage = stage;
        create.layout = pipeline_layout;
        VkPipeline result{};
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &create, nullptr, &result), "create compute pipeline");
        try { pipelines.emplace(key, result); }
        catch (...) { vkDestroyPipeline(device, result, nullptr); throw; }
        counters.pipeline_time_ns += ns_since(start);
        return result;
    }

    void wait_slot(std::size_t slot) {
        if (!pending[slot]) return;
        auto start = Clock::now();
        check(vkWaitForFences(device, 1, &fences[slot], VK_TRUE, 60'000'000'000ULL), "wait compute fence");
        counters.wait_time_ns += ns_since(start);
        pending[slot] = false;
        if (pending_timestamps[slot]) {
            std::array<std::uint64_t, 2> times{};
            check(vkGetQueryPoolResults(device, query_pool, slot * 2, 2, sizeof(times), times.data(),
                                       sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "read GPU timestamps");
            const auto mask = timestamp_bits == 64 ? ~std::uint64_t(0) : (std::uint64_t(1) << timestamp_bits) - 1;
            counters.gpu_time_ns += static_cast<std::uint64_t>(
                static_cast<double>((times[1] - times[0]) & mask) * properties.limits.timestampPeriod);
            pending_timestamps[slot] = false;
        }
    }

    std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> available_heap_bytes() const {
        std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> available{};
        if (!has_memory_budget) return available;
        const auto get_memory_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceMemoryProperties2"));
        if (!get_memory_properties) return available;
        auto budget = info<VkPhysicalDeviceMemoryBudgetPropertiesEXT>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT);
        auto properties2 = info<VkPhysicalDeviceMemoryProperties2>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2);
        properties2.pNext = &budget;
        get_memory_properties(physical, &properties2);
        for (std::uint32_t heap = 0; heap < properties2.memoryProperties.memoryHeapCount; ++heap) {
            if (!(properties2.memoryProperties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
            available[heap] = budget.heapBudget[heap] > budget.heapUsage[heap]
                ? budget.heapBudget[heap] - budget.heapUsage[heap] : 0;
        }
        return available;
    }

    VkDeviceSize system_available_bytes() const {
#if defined(__linux__)
        std::ifstream meminfo("/proc/meminfo");
        std::string key, unit;
        std::uint64_t value = 0;
        while (meminfo >> key >> value >> unit) {
            if (key == "MemAvailable:" && unit == "kB") {
                if (value > std::numeric_limits<VkDeviceSize>::max() / 1024U)
                    return std::numeric_limits<VkDeviceSize>::max();
                return value * 1024U;
            }
        }
#endif
        return 0;
    }

    VkDeviceSize model_cache_reserve_bytes() const {
        return properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
            ? 1024ULL * 1024U * 1024U
            : 2ULL * 1024U * 1024U * 1024U;
    }

    bool ensure_device_local_buffer(Buffer& buffer, VkDeviceSize size,
                                    const std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS>& heap_available) {
        const auto reserve_bytes = model_cache_reserve_bytes();
        size = (size + 3) & ~VkDeviceSize(3);
        if (buffer.capacity >= size) return true;
        buffer.reset();
        buffer.device = device;
        auto create = info<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
        create.size = size;
        create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        const auto create_result = vkCreateBuffer(device, &create, nullptr, &buffer.buffer);
        if (create_result == VK_ERROR_OUT_OF_DEVICE_MEMORY || create_result == VK_ERROR_OUT_OF_HOST_MEMORY)
            return false;
        check(create_result, "create resident model buffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
        int best_score = -1;
        std::uint32_t memory_type = memory_properties.memoryTypeCount;
        for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            if (!(requirements.memoryTypeBits & (1U << i))) continue;
            const auto& type = memory_properties.memoryTypes[i];
            if (!(type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
            const auto heap = type.heapIndex;
            if (heap_available[heap] < requirements.size ||
                heap_available[heap] - requirements.size < reserve_bytes) continue;
            int score = (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? 0 : 2;
            if (heap_available[heap] - requirements.size > reserve_bytes * 2) ++score;
            if (score > best_score) { best_score = score; memory_type = i; }
        }
        if (memory_type == memory_properties.memoryTypeCount) {
            buffer.reset();
            return false;
        }
        auto allocate = info<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = memory_type;
        const auto allocate_result = vkAllocateMemory(device, &allocate, nullptr, &buffer.memory);
        if (allocate_result == VK_ERROR_OUT_OF_DEVICE_MEMORY || allocate_result == VK_ERROR_OUT_OF_HOST_MEMORY) {
            buffer.reset();
            return false;
        }
        check(allocate_result, "allocate resident model memory");
        const auto bind_result = vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0);
        if (bind_result == VK_ERROR_OUT_OF_DEVICE_MEMORY || bind_result == VK_ERROR_OUT_OF_HOST_MEMORY) {
            buffer.reset();
            return false;
        }
        check(bind_result, "bind resident model memory");
        buffer.flags = memory_properties.memoryTypes[memory_type].propertyFlags;
        buffer.capacity = size;
        buffer.allocation = requirements.size;
        return true;
    }

    void submit_copy(std::size_t slot, VkDeviceSize destination_offset, VkDeviceSize bytes) {
        auto command = command_buffers[slot];
        check(vkResetCommandBuffer(command, 0), "reset model upload command buffer");
        auto begin = info<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(command, &begin), "begin model upload command buffer");

        auto source_barrier = info<VkBufferMemoryBarrier>(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER);
        source_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        source_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        source_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        source_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        source_barrier.buffer = weights[slot].buffer;
        source_barrier.offset = 0;
        source_barrier.size = bytes;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 1, &source_barrier, 0, nullptr);

        VkBufferCopy copy{};
        copy.srcOffset = 0;
        copy.dstOffset = destination_offset;
        copy.size = bytes;
        vkCmdCopyBuffer(command, weights[slot].buffer, model_weights.buffer, 1, &copy);

        auto destination_barrier = info<VkBufferMemoryBarrier>(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER);
        destination_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        destination_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        destination_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        destination_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        destination_barrier.buffer = model_weights.buffer;
        destination_barrier.offset = destination_offset;
        destination_barrier.size = bytes;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, 1, &destination_barrier, 0, nullptr);
        check(vkEndCommandBuffer(command), "end model upload command buffer");
        check(vkResetFences(device, 1, &fences[slot]), "reset model upload fence");
        auto submit_info = info<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command;
        check(vkQueueSubmit(queue, 1, &submit_info, fences[slot]), "submit model upload");
        pending[slot] = true;
        pending_timestamps[slot] = false;
    }

    bool prepare_model_cache(const model::TensorReader& reader) {
        const auto& file = reader.model_file();
        if (model_cache_mode == VulkanModelCacheMode::Disabled || !file || !has_memory_budget) return false;
        if (model_cache_file == file) return counters.model_cache_active;
        for (std::size_t slot = 0; slot < pending.size(); ++slot) wait_slot(slot);
        model_weights.reset();
        counters.model_cache_active = false;
        counters.model_cache_capacity_bytes = 0;
        counters.model_cache_uploaded_bytes = 0;
        counters.model_cache_upload_time_ns = 0;
        model_cache_file = file;

        const auto file_size = file->size();
        if (file_size == 0 || file_size > std::numeric_limits<VkDeviceSize>::max() - 3) return false;
        const auto heaps = available_heap_bytes();
        const auto max_heap = *std::max_element(heaps.begin(), heaps.end());
        const auto reserve_bytes = model_cache_reserve_bytes();
        if (max_heap < reserve_bytes || file_size > max_heap - reserve_bytes) return false;
        if (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            const auto available_system = system_available_bytes();
            if (available_system < reserve_bytes || file_size > available_system - reserve_bytes)
                return false;
        }
        if (!ensure_device_local_buffer(model_weights, file_size, heaps)) return false;

        const auto upload_start = Clock::now();
        std::uint64_t offset = 0;
        std::size_t tile = 0;
        while (offset < file_size) {
            const auto slot = tile++ % weights.size();
            wait_slot(slot);
            const auto bytes = static_cast<std::size_t>(std::min<std::uint64_t>(tile_bytes, file_size - offset));
            const auto copy_bytes = (bytes + 3U) & ~std::size_t(3U);
            file->read_into_uncached(offset, {static_cast<std::byte*>(weights[slot].mapped), bytes});
            if (copy_bytes > bytes)
                std::memset(static_cast<std::byte*>(weights[slot].mapped) + bytes, 0, copy_bytes - bytes);
            weights[slot].flush(copy_bytes);
            submit_copy(slot, offset, copy_bytes);
            counters.model_cache_uploaded_bytes += bytes;
            offset += bytes;
        }
        for (std::size_t slot = 0; slot < pending.size(); ++slot) wait_slot(slot);
        counters.model_cache_capacity_bytes = model_weights.allocation;
        counters.model_cache_upload_time_ns += ns_since(upload_start);
        counters.model_cache_active = true;
        counters.allocated_bytes += model_weights.allocation;
        return true;
    }

    void submit(std::size_t slot, VkPipeline compute, std::uint32_t columns,
                std::uint32_t rows, std::uint32_t row_bytes, std::uint32_t first,
                std::uint32_t rows_per_workgroup) {
        auto command = command_buffers[slot];
        check(vkResetCommandBuffer(command, 0), "reset command buffer");
        auto begin = info<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(command, &begin), "begin command buffer");
        auto barrier = info<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
        if (query_pool) {
            vkCmdResetQueryPool(command, query_pool, slot * 2, 2);
            vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, slot * 2);
        }
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, compute);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                                &descriptor_sets[slot], 0, nullptr);
        const std::array<std::uint32_t, 4> push{columns, rows, row_bytes, first};
        vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push.data());
        vkCmdDispatch(command, (rows + rows_per_workgroup - 1) / rows_per_workgroup, 1, 1);
        if (query_pool) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, slot * 2 + 1);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(command), "end command buffer");
        check(vkResetFences(device, 1, &fences[slot]), "reset compute fence");
        auto submit_info = info<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command;
        check(vkQueueSubmit(queue, 1, &submit_info, fences[slot]), "submit compute");
        pending[slot] = true;
        pending_timestamps[slot] = query_pool != VK_NULL_HANDLE;
        ++counters.dispatches;
    }

    bool linear(const model::TensorReader& reader, std::span<const float> input,
                std::span<float> output) {
        const auto& tensor = reader.tensor();
        if (!supported_type(tensor.type)) return false;
        if (input.empty() || output.empty() || tensor.dimensions.empty() ||
            tensor.dimensions.front() != input.size() || reader.row_count() != output.size()) {
            throw std::invalid_argument("Vulkan matrix dimensions do not match input/output");
        }
        if (input.size() > std::numeric_limits<std::uint32_t>::max() ||
            output.size() > std::numeric_limits<std::uint32_t>::max() || reader.row_bytes() > tile_bytes ||
            input.size_bytes() > properties.limits.maxStorageBufferRange ||
            output.size_bytes() > properties.limits.maxStorageBufferRange) return false;
        std::size_t expected_row_bytes = 0;
        if (tensor.type == 0) expected_row_bytes = input.size() * 4;
        else if (tensor.type == 1 || tensor.type == 30) expected_row_bytes = input.size() * 2;
        else {
            const std::size_t block = tensor.type >= 12 ? 256 : 32;
            if (input.size() % block) throw std::invalid_argument("Vulkan quantized row is not block-aligned");
            const std::size_t bytes = tensor.type == 2 ? 18 : tensor.type == 3 ? 20 :
                tensor.type == 8 ? 34 : tensor.type == 12 ? 144 : tensor.type == 13 ? 176 : 210;
            expected_row_bytes = input.size() / block * bytes;
        }
        if (reader.row_bytes() != expected_row_bytes) throw std::invalid_argument("Vulkan encoded row size mismatch");
        std::lock_guard lock(mutex);
        if (failed) throw std::runtime_error("Vulkan backend failed during an earlier operation");
        try {
            const bool vectorized_q4_k = options_vectorized_q4_k && use_subgroups &&
                (subgroup_size == 16 || subgroup_size == 32) &&
                (tensor.type == 12 || tensor.type == 13);
            const std::uint32_t dispatch_rows_per_group = vectorized_q4_k ? 4 : rows_per_group;
            auto compute = pipeline(tensor.type, vectorized_q4_k);
            const bool model_cached = prepare_model_cache(reader);
            ensure_buffer(input_buffer, input.size_bytes());
            ensure_buffer(output_buffer, output.size_bytes(), true);
            std::memcpy(input_buffer.mapped, input.data(), input.size_bytes());
            auto cache_start = Clock::now();
            input_buffer.flush(input.size_bytes());
            counters.cache_time_ns += ns_since(cache_start);
            const auto row_bytes = reader.row_bytes();
            const auto storage_offset_alignment = properties.limits.minStorageBufferOffsetAlignment;
            const bool use_model_cache = model_cached && tensor.payload_size &&
                *tensor.payload_size <= properties.limits.maxStorageBufferRange &&
                (!storage_offset_alignment ||
                    (tensor.file_offset % storage_offset_alignment == 0 &&
                     row_bytes % storage_offset_alignment == 0));
            for (std::size_t slot = 0; slot < 2; ++slot) {
                std::array<VkDescriptorBufferInfo, 3> buffers{{
                    {use_model_cache ? model_weights.buffer : weights[slot].buffer,
                     use_model_cache ? tensor.file_offset : 0,
                     use_model_cache ? *tensor.payload_size : weights[slot].capacity},
                    {input_buffer.buffer, 0, input_buffer.capacity},
                    {output_buffer.buffer, 0, output_buffer.capacity}}};
                std::array<VkWriteDescriptorSet, 3> writes{};
                for (std::uint32_t i = 0; i < 3; ++i) {
                    writes[i] = info<VkWriteDescriptorSet>(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
                    writes[i].dstSet = descriptor_sets[slot];
                    writes[i].dstBinding = i;
                    writes[i].descriptorCount = 1;
                    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    writes[i].pBufferInfo = &buffers[i];
                }
                vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
            }
            auto dispatch_tile_bytes = tile_bytes;
            if (use_model_cache && properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                // Resident weights do not use the mapped transfer-window size
                // for compute; larger row batches reduce dispatch and fence overhead.
                constexpr VkDeviceSize resident_dispatch_tile_bytes = 64ULL * 1024U * 1024U;
                dispatch_tile_bytes = std::max(dispatch_tile_bytes, resident_dispatch_tile_bytes);
            }
            auto rows_per_tile = static_cast<std::size_t>(dispatch_tile_bytes / row_bytes);
            rows_per_tile = std::min<std::size_t>(rows_per_tile,
                static_cast<std::size_t>(properties.limits.maxComputeWorkGroupCount[0]) * dispatch_rows_per_group);
            if (use_model_cache) counters.weight_bytes += *tensor.payload_size;
            std::size_t tile = 0;
            for (std::size_t first = 0; first < output.size(); ++tile) {
                const auto slot = tile % 2;
                wait_slot(slot);
                const auto rows = std::min<std::size_t>(rows_per_tile, output.size() - first);
                const auto bytes = rows * row_bytes;
                if (use_model_cache) {
                    VkDescriptorBufferInfo weight_buffer{
                        model_weights.buffer,
                        tensor.file_offset + static_cast<VkDeviceSize>(first * row_bytes),
                        *tensor.payload_size - first * row_bytes};
                    auto weight_write = info<VkWriteDescriptorSet>(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
                    weight_write.dstSet = descriptor_sets[slot];
                    weight_write.dstBinding = 0;
                    weight_write.descriptorCount = 1;
                    weight_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    weight_write.pBufferInfo = &weight_buffer;
                    vkUpdateDescriptorSets(device, 1, &weight_write, 0, nullptr);
                }
                if (!use_model_cache) {
                    const auto start = Clock::now();
                    reader.read_rows_into(first, rows,
                        {static_cast<std::byte*>(weights[slot].mapped), bytes});
                    counters.read_time_ns += ns_since(start);
                    counters.weight_bytes += bytes;
                    cache_start = Clock::now();
                    weights[slot].flush(bytes);
                    counters.cache_time_ns += ns_since(cache_start);
                }
                submit(slot, compute, input.size(), rows, row_bytes, first,
                       dispatch_rows_per_group);
                first += rows;
            }
            wait_slot(0);
            wait_slot(1);
            cache_start = Clock::now();
            output_buffer.invalidate(output.size_bytes());
            counters.cache_time_ns += ns_since(cache_start);
            std::memcpy(output.data(), output_buffer.mapped, output.size_bytes());
            ++counters.linear_calls;
            if (vectorized_q4_k) ++counters.vectorized_q4_k_calls;
            counters.allocated_bytes = input_buffer.allocation + output_buffer.allocation;
            for (const auto& buffer : weights) counters.allocated_bytes += buffer.allocation;
            counters.allocated_bytes += model_weights.allocation;
            return true;
        } catch (...) {
            failed = true;
            (void)vkDeviceWaitIdle(device);
            throw;
        }
    }

    void shutdown() noexcept {
        if (device) (void)vkDeviceWaitIdle(device);
        input_buffer.reset();
        output_buffer.reset();
        model_weights.reset();
        for (auto& buffer : weights) buffer.reset();
        model_cache_file.reset();
        for (auto& [type, pipeline] : pipelines) vkDestroyPipeline(device, pipeline, nullptr);
        pipelines.clear();
        if (shader) vkDestroyShaderModule(device, shader, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        if (descriptor_layout) vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
        if (query_pool) vkDestroyQueryPool(device, query_pool, nullptr);
        for (auto fence : fences) if (fence) vkDestroyFence(device, fence, nullptr);
        if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
        device = VK_NULL_HANDLE;
        instance = VK_NULL_HANDLE;
    }

    mutable std::mutex mutex;
    VulkanLinearStats counters;
    std::string description;
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memory_properties{};
    VkDevice device{};
    VkQueue queue{};
    std::uint32_t queue_family{}, rows_per_group{}, timestamp_bits{};
    std::uint32_t subgroup_size{};
    VkDeviceSize tile_bytes{};
    bool use_subgroups{}, options_vectorized_q4_k{}, failed{}, has_memory_budget{};
    VulkanModelCacheMode model_cache_mode{VulkanModelCacheMode::Automatic};
    std::array<Buffer, 2> weights;
    Buffer input_buffer, output_buffer, model_weights;
    std::shared_ptr<storage::ModelFile> model_cache_file;
    VkCommandPool command_pool{};
    std::array<VkCommandBuffer, 2> command_buffers{};
    std::array<VkFence, 2> fences{};
    std::array<bool, 2> pending{};
    std::array<bool, 2> pending_timestamps{};
    VkQueryPool query_pool{};
    VkDescriptorSetLayout descriptor_layout{};
    VkDescriptorPool descriptor_pool{};
    std::array<VkDescriptorSet, 2> descriptor_sets{};
    VkPipelineLayout pipeline_layout{};
    VkShaderModule shader{};
    std::unordered_map<std::uint32_t, VkPipeline> pipelines;
};

VulkanLinearBackend::VulkanLinearBackend(const VulkanLinearOptions& options)
    : impl_(std::make_unique<Impl>(options)) {}
VulkanLinearBackend::~VulkanLinearBackend() = default;
bool VulkanLinearBackend::try_linear(const model::TensorReader& weights,
                                     std::span<const float> input, std::span<float> output) {
    return impl_->linear(weights, input, output);
}
VulkanLinearStats VulkanLinearBackend::stats() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->counters;
}
std::string VulkanLinearBackend::device_name() const { return impl_->description; }
}  // namespace pokitlms::gpu
