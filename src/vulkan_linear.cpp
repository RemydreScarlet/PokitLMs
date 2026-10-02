#include "pokitlms/gpu/vulkan_linear.hpp"
#include "pokitlms/vulkan_shaders.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>
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
        if (use_subgroups) {
            full.subgroupSizeControl = VK_FALSE;
            full.computeFullSubgroups = VK_TRUE;
            device_create.pNext = &full;
            device_create.enabledExtensionCount = 1;
            device_create.ppEnabledExtensionNames = &size_control_extension;
        }
        check(vkCreateDevice(physical, &device_create, nullptr, &device), "create device");
        vkGetDeviceQueue(device, queue_family, 0, &queue);
        tile_bytes = std::min<VkDeviceSize>(options.tile_bytes,
            properties.limits.maxStorageBufferRange - 4) & ~VkDeviceSize(3);
        // Spare bytes allow a uint load containing a final halfword.
        for (auto& buffer : weights) {
            ensure_buffer(buffer, tile_bytes + 4, options.prefer_host_cached_weights);
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

    void ensure_buffer(Buffer& buffer, VkDeviceSize size, bool prefer_cached = false) {
        size = (size + 3) & ~VkDeviceSize(3);
        if (buffer.capacity >= size) return;
        buffer.reset();
        buffer.device = device;
        auto create = info<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
        create.size = size;
        create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
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
        if (query_pool) {
            std::array<std::uint64_t, 2> times{};
            check(vkGetQueryPoolResults(device, query_pool, slot * 2, 2, sizeof(times), times.data(),
                                       sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "read GPU timestamps");
            const auto mask = timestamp_bits == 64 ? ~std::uint64_t(0) : (std::uint64_t(1) << timestamp_bits) - 1;
            counters.gpu_time_ns += static_cast<std::uint64_t>(
                static_cast<double>((times[1] - times[0]) & mask) * properties.limits.timestampPeriod);
        }
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
            ensure_buffer(input_buffer, input.size_bytes());
            ensure_buffer(output_buffer, output.size_bytes(), true);
            std::memcpy(input_buffer.mapped, input.data(), input.size_bytes());
            auto cache_start = Clock::now();
            input_buffer.flush(input.size_bytes());
            counters.cache_time_ns += ns_since(cache_start);
            for (std::size_t slot = 0; slot < 2; ++slot) {
                std::array<VkDescriptorBufferInfo, 3> buffers{{
                    {weights[slot].buffer, 0, weights[slot].capacity},
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
            const auto row_bytes = reader.row_bytes();
            auto rows_per_tile = tile_bytes / row_bytes;
            rows_per_tile = std::min<std::size_t>(rows_per_tile,
                static_cast<std::size_t>(properties.limits.maxComputeWorkGroupCount[0]) * dispatch_rows_per_group);
            std::size_t tile = 0;
            for (std::size_t first = 0; first < output.size(); ++tile) {
                const auto slot = tile % 2;
                wait_slot(slot);
                const auto rows = std::min<std::size_t>(rows_per_tile, output.size() - first);
                const auto bytes = rows * row_bytes;
                const auto start = Clock::now();
                reader.read_rows_into(first, rows,
                    {static_cast<std::byte*>(weights[slot].mapped), bytes});
                counters.read_time_ns += ns_since(start);
                counters.weight_bytes += bytes;
                cache_start = Clock::now();
                weights[slot].flush(bytes);
                counters.cache_time_ns += ns_since(cache_start);
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
        for (auto& buffer : weights) buffer.reset();
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
    bool use_subgroups{}, options_vectorized_q4_k{}, failed{};
    std::array<Buffer, 2> weights;
    Buffer input_buffer, output_buffer;
    VkCommandPool command_pool{};
    std::array<VkCommandBuffer, 2> command_buffers{};
    std::array<VkFence, 2> fences{};
    std::array<bool, 2> pending{};
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
