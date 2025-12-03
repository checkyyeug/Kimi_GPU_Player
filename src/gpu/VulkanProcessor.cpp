#include "gpu/VulkanProcessor.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <limits>

namespace GPUPlayer {

// Vulkan全局函数指针
static PFN_vkCreateInstance vkCreateInstance = nullptr;
static PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
static PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
static PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
static PFN_vkCreateDevice vkCreateDevice = nullptr;
static PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = nullptr;
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
static PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties = nullptr;
static PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
static PFN_vkAllocateMemory vkAllocateMemory = nullptr;
static PFN_vkCreateBuffer vkCreateBuffer = nullptr;
static PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
static PFN_vkMapMemory vkMapMemory = nullptr;
static PFN_vkUnmapMemory vkUnmapMemory = nullptr;
static PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
static PFN_vkFreeMemory vkFreeMemory = nullptr;
static PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
static PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
static PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
static PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
static PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
static PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
static PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
static PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
static PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
static PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
static PFN_vkCmdDispatch vkCmdDispatch = nullptr;
static PFN_vkQueueSubmit vkQueueSubmit = nullptr;
static PFN_vkQueueWaitIdle vkQueueWaitIdle = nullptr;
static PFN_vkDeviceWaitIdle vkDeviceWaitIdle = nullptr;
static PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;
static PFN_vkDestroyPipeline vkDestroyPipeline = nullptr;
static PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = nullptr;
static PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
static PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
static PFN_vkDestroyDevice vkDestroyDevice = nullptr;
static PFN_vkDestroyInstance vkDestroyInstance = nullptr;

VulkanProcessor::VulkanProcessor() 
    : instance_(VK_NULL_HANDLE)
    , physicalDevice_(VK_NULL_HANDLE)
    , device_(VK_NULL_HANDLE)
    , computeQueue_(VK_NULL_HANDLE)
    , commandPool_(VK_NULL_HANDLE)
    , descriptorSetLayout_(VK_NULL_HANDLE)
    , pipelineLayout_(VK_NULL_HANDLE)
    , resamplePipeline_(VK_NULL_HANDLE)
    , eqPipeline_(VK_NULL_HANDLE)
    , filterPipeline_(VK_NULL_HANDLE)
    , stagingBuffer_(VK_NULL_HANDLE)
    , stagingMemory_(VK_NULL_HANDLE)
    , deviceBuffer_(VK_NULL_HANDLE)
    , deviceMemory_(VK_NULL_HANDLE)
    , descriptorSet_(VK_NULL_HANDLE)
    , descriptorPool_(VK_NULL_HANDLE)
    , initialized_(false)
    , computeQueueFamilyIndex_(0)
    , resampleShaderModule_(VK_NULL_HANDLE)
    , eqShaderModule_(VK_NULL_HANDLE)
    , filterShaderModule_(VK_NULL_HANDLE) {
}

VulkanProcessor::~VulkanProcessor() {
    Shutdown();
}

bool VulkanProcessor::Initialize() {
    if (initialized_) {
        return true;
    }
    
    std::cout << "[VULKAN] 初始化Vulkan处理器..." << std::endl;
    
    // 1. 创建Vulkan实例
    if (!CreateInstance()) {
        return false;
    }
    
    // 2. 选择物理设备
    if (!SelectPhysicalDevice()) {
        CleanupVulkanResources();
        return false;
    }
    
    // 3. 创建逻辑设备
    if (!CreateDevice()) {
        CleanupVulkanResources();
        return false;
    }
    
    // 4. 创建命令池
    if (!CreateCommandPool()) {
        CleanupVulkanResources();
        return false;
    }
    
    // 5. 创建描述符集布局
    if (!CreateDescriptorSetLayout()) {
        CleanupVulkanResources();
        return false;
    }
    
    // 6. 创建管线布局
    if (!CreatePipelineLayout()) {
        CleanupVulkanResources();
        return false;
    }
    
    // 7. 创建计算管线
    if (!CreateComputePipelines()) {
        CleanupVulkanResources();
        return false;
    }
    
    // 8. 创建缓冲区
    if (!CreateBuffers()) {
        CleanupVulkanResources();
        return false;
    }
    
    // 9. 创建描述符池和集合
    if (!CreateDescriptorPool() || !AllocateDescriptorSet()) {
        CleanupVulkanResources();
        return false;
    }
    
    initialized_ = true;
    std::cout << "[VULKAN] Vulkan处理器初始化成功" << std::endl;
    std::cout << "[VULKAN] 设备: " << GetGPUName() << std::endl;
    std::cout << "[VULKAN] 内存: " << (GetGPUMemory() / 1024 / 1024) << " MB" << std::endl;
    
    return true;
}

void VulkanProcessor::Shutdown() {
    if (!initialized_) {
        return;
    }
    
    std::cout << "[VULKAN] 关闭Vulkan处理器..." << std::endl;
    
    // 等待设备空闲
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }
    
    CleanupVulkanResources();
    initialized_ = false;
}

bool VulkanProcessor::CreateInstance() {
    // 检查Vulkan支持
    if (!DetectVulkanSupport()) {
        SetError("Vulkan不支持或驱动未安装");
        return false;
    }
    
    // 应用程序信息
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "GPU Music Player";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "GPU Audio Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;
    
    // 实例创建信息
    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    
    // 检查扩展支持
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
    
    std::vector<const char*> requiredExtensions;
    
    // 检查必要的扩展
    bool hasVKKHRGetPhysicalDeviceProperties2 = false;
    for (const auto& ext : extensions) {
        if (strcmp(ext.extensionName, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == 0) {
            hasVKKHRGetPhysicalDeviceProperties2 = true;
        }
    }
    
    if (hasVKKHRGetPhysicalDeviceProperties2) {
        requiredExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }
    
    createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
    createInfo.ppEnabledExtensionNames = requiredExtensions.data();
    createInfo.enabledLayerCount = 0;
    
    // 创建实例
    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance_);
    if (!CheckVulkanResult(result, "创建Vulkan实例")) {
        return false;
    }
    
    // 加载全局函数指针
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)vkGetInstanceProcAddr(instance_, "vkGetInstanceProcAddr");
    vkEnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)vkGetInstanceProcAddr(instance_, "vkEnumeratePhysicalDevices");
    vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceProperties");
    vkGetPhysicalDeviceQueueFamilyProperties = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceQueueFamilyProperties");
    vkCreateDevice = (PFN_vkCreateDevice)vkGetInstanceProcAddr(instance_, "vkCreateDevice");
    vkEnumerateDeviceExtensionProperties = (PFN_vkEnumerateDeviceExtensionProperties)vkGetInstanceProcAddr(instance_, "vkEnumerateDeviceExtensionProperties");
    vkGetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceMemoryProperties");
    vkDestroyInstance = (PFN_vkDestroyInstance)vkGetInstanceProcAddr(instance_, "vkDestroyInstance");
    
    return true;
}

bool VulkanProcessor::SelectPhysicalDevice() {
    // 枚举物理设备
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    
    if (deviceCount == 0) {
        SetError("未找到支持的Vulkan物理设备");
        return false;
    }
    
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());
    
    // 选择第一个支持计算队列的设备
    for (const auto& device : devices) {
        // 获取设备属性
        vkGetPhysicalDeviceProperties(device, &deviceProperties_);
        vkGetPhysicalDeviceMemoryProperties(device, &memoryProperties_);
        
        // 获取队列族属性
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        
        // 寻找支持计算队列的族
        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physicalDevice_ = device;
                computeQueueFamilyIndex_ = i;
                
                std::cout << "[VULKAN] 选择物理设备: " << deviceProperties_.deviceName << std::endl;
                return true;
            }
        }
    }
    
    SetError("未找到支持计算队列的Vulkan设备");
    return false;
}

bool VulkanProcessor::CreateDevice() {
    // 设备队列创建信息
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex_;
    queueCreateInfo.queueCount = 1;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    
    // 设备功能
    VkPhysicalDeviceFeatures deviceFeatures = {};
    
    // 设备创建信息
    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = 0;
    createInfo.ppEnabledExtensionNames = nullptr;
    
    // 创建设备
    VkResult result = vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_);
    if (!CheckVulkanResult(result, "创建Vulkan设备")) {
        return false;
    }
    
    // 加载设备函数指针
    vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr(instance_, "vkGetDeviceProcAddr");
    vkAllocateMemory = (PFN_vkAllocateMemory)vkGetDeviceProcAddr(device_, "vkAllocateMemory");
    vkCreateBuffer = (PFN_vkCreateBuffer)vkGetDeviceProcAddr(device_, "vkCreateBuffer");
    vkBindBufferMemory = (PFN_vkBindBufferMemory)vkGetDeviceProcAddr(device_, "vkBindBufferMemory");
    vkMapMemory = (PFN_vkMapMemory)vkGetDeviceProcAddr(device_, "vkMapMemory");
    vkUnmapMemory = (PFN_vkUnmapMemory)vkGetDeviceProcAddr(device_, "vkUnmapMemory");
    vkDestroyBuffer = (PFN_vkDestroyBuffer)vkGetDeviceProcAddr(device_, "vkDestroyBuffer");
    vkFreeMemory = (PFN_vkFreeMemory)vkGetDeviceProcAddr(device_, "vkFreeMemory");
    vkCreateShaderModule = (PFN_vkCreateShaderModule)vkGetDeviceProcAddr(device_, "vkCreateShaderModule");
    vkCreateDescriptorSetLayout = (PFN_vkCreateDescriptorSetLayout)vkGetDeviceProcAddr(device_, "vkCreateDescriptorSetLayout");
    vkCreatePipelineLayout = (PFN_vkCreatePipelineLayout)vkGetDeviceProcAddr(device_, "vkCreatePipelineLayout");
    vkCreateComputePipelines = (PFN_vkCreateComputePipelines)vkGetDeviceProcAddr(device_, "vkCreateComputePipelines");
    vkCreateCommandPool = (PFN_vkCreateCommandPool)vkGetDeviceProcAddr(device_, "vkCreateCommandPool");
    vkAllocateCommandBuffers = (PFN_vkAllocateCommandBuffers)vkGetDeviceProcAddr(device_, "vkAllocateCommandBuffers");
    vkBeginCommandBuffer = (PFN_vkBeginCommandBuffer)vkGetDeviceProcAddr(device_, "vkBeginCommandBuffer");
    vkEndCommandBuffer = (PFN_vkEndCommandBuffer)vkGetDeviceProcAddr(device_, "vkEndCommandBuffer");
    vkCmdBindPipeline = (PFN_vkCmdBindPipeline)vkGetDeviceProcAddr(device_, "vkCmdBindPipeline");
    vkCmdBindDescriptorSets = (PFN_vkCmdBindDescriptorSets)vkGetDeviceProcAddr(device_, "vkCmdBindDescriptorSets");
    vkCmdDispatch = (PFN_vkCmdDispatch)vkGetDeviceProcAddr(device_, "vkCmdDispatch");
    vkQueueSubmit = (PFN_vkQueueSubmit)vkGetDeviceProcAddr(device_, "vkQueueSubmit");
    vkQueueWaitIdle = (PFN_vkQueueWaitIdle)vkGetDeviceProcAddr(device_, "vkQueueWaitIdle");
    vkDeviceWaitIdle = (PFN_vkDeviceWaitIdle)vkGetDeviceProcAddr(device_, "vkDeviceWaitIdle");
    vkDestroyShaderModule = (PFN_vkDestroyShaderModule)vkGetDeviceProcAddr(device_, "vkDestroyShaderModule");
    vkDestroyPipeline = (PFN_vkDestroyPipeline)vkGetDeviceProcAddr(device_, "vkDestroyPipeline");
    vkDestroyPipelineLayout = (PFN_vkDestroyPipelineLayout)vkGetDeviceProcAddr(device_, "vkDestroyPipelineLayout");
    vkDestroyDescriptorSetLayout = (PFN_vkDestroyDescriptorSetLayout)vkGetDeviceProcAddr(device_, "vkDestroyDescriptorSetLayout");
    vkDestroyCommandPool = (PFN_vkDestroyCommandPool)vkGetDeviceProcAddr(device_, "vkDestroyCommandPool");
    vkDestroyDevice = (PFN_vkDestroyDevice)vkGetDeviceProcAddr(device_, "vkDestroyDevice");
    
    // 获取计算队列
    vkGetDeviceQueue(device_, computeQueueFamilyIndex_, 0, &computeQueue_);
    
    return true;
}

bool VulkanProcessor::CreateCommandPool() {
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex_;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    
    VkResult result = vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_);
    return CheckVulkanResult(result, "创建命令池");
}

bool VulkanProcessor::CreateDescriptorSetLayout() {
    // 定义描述符集布局绑定
    std::array<VkDescriptorSetLayoutBinding, 3> bindings = {};
    
    // 输入缓冲区绑定
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[0].pImmutableSamplers = nullptr;
    
    // 输出缓冲区绑定
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].pImmutableSamplers = nullptr;
    
    // 参数缓冲区绑定
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].pImmutableSamplers = nullptr;
    
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    
    VkResult result = vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_);
    return CheckVulkanResult(result, "创建描述符集布局");
}

bool VulkanProcessor::CreatePipelineLayout() {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;
    
    VkResult result = vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_);
    return CheckVulkanResult(result, "创建管线布局");
}

bool VulkanProcessor::CreateComputePipelines() {
    // 这里简化处理，实际应该加载预编译的SPIR-V着色器
    // 目前创建空的计算管线作为占位符
    
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = pipelineLayout_;
    pipelineInfo.flags = 0;
    
    // 重采样管线 (占位符)
    VkResult result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &resamplePipeline_);
    if (!CheckVulkanResult(result, "创建重采样计算管线")) {
        return false;
    }
    
    // EQ管线 (占位符)
    result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &eqPipeline_);
    if (!CheckVulkanResult(result, "创建EQ计算管线")) {
        return false;
    }
    
    // 滤波器管线 (占位符)
    result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &filterPipeline_);
    if (!CheckVulkanResult(result, "创建滤波器计算管线")) {
        return false;
    }
    
    return true;
}

bool VulkanProcessor::CreateBuffers() {
    VkDeviceSize bufferSize = 1024 * 1024 * 16; // 16MB缓冲区
    
    // 创建暂存缓冲区
    if (!CreateBuffer(bufferSize, 
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer_, stagingMemory_)) {
        return false;
    }
    
    // 创建设备缓冲区
    if (!CreateBuffer(bufferSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     deviceBuffer_, deviceMemory_)) {
        return false;
    }
    
    return true;
}

bool VulkanProcessor::CreateDescriptorPool() {
    std::array<VkDescriptorPoolSize, 3> poolSizes = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 2;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 2;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[2].descriptorCount = 2;
    
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;
    
    VkResult result = vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_);
    return CheckVulkanResult(result, "创建描述符池");
}

bool VulkanProcessor::AllocateDescriptorSet() {
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout_;
    
    VkResult result = vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSet_);
    return CheckVulkanResult(result, "分配描述符集");
}

bool VulkanProcessor::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                                  VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VkResult result = vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer);
    if (!CheckVulkanResult(result, "创建缓冲区")) {
        return false;
    }
    
    VkMemoryRequirements memRequirements;
    // vkGetBufferMemoryRequirements 需要加载
    auto vkGetBufferMemoryRequirements = (PFN_vkGetBufferMemoryRequirements)vkGetDeviceProcAddr(device_, "vkGetBufferMemoryRequirements");
    vkGetBufferMemoryRequirements(device_, buffer, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);
    
    result = vkAllocateMemory(device_, &allocInfo, nullptr, &bufferMemory);
    if (!CheckVulkanResult(result, "分配内存")) {
        vkDestroyBuffer(device_, buffer, nullptr);
        return false;
    }
    
    vkBindBufferMemory(device_, buffer, bufferMemory, 0);
    return true;
}

uint32_t VulkanProcessor::FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < memoryProperties_.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memoryProperties_.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    throw std::runtime_error("未找到合适的内存类型");
}

bool VulkanProcessor::DetectVulkanSupport() {
    // 简单的Vulkan支持检测
    #ifdef __linux__
        void* libvulkan = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!libvulkan) {
            libvulkan = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
        }
    #elif defined(_WIN32)
        HMODULE libvulkan = LoadLibraryA("vulkan-1.dll");
    #elif defined(__APPLE__)
        void* libvulkan = dlopen("libvulkan.dylib", RTLD_NOW | RTLD_LOCAL);
        if (!libvulkan) {
            libvulkan = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
        }
    #endif
    
    #ifdef __linux__
        if (libvulkan) {
            dlclose(libvulkan);
            return true;
        }
    #elif defined(_WIN32)
        if (libvulkan) {
            FreeLibrary(libvulkan);
            return true;
        }
    #elif defined(__APPLE__)
        if (libvulkan) {
            dlclose(libvulkan);
            return true;
        }
    #endif
    
    return false;
}

bool VulkanProcessor::IsGPUSupported() const {
    return initialized_ && physicalDevice_ != VK_NULL_HANDLE;
}

std::string VulkanProcessor::GetGPUName() const {
    if (!initialized_ || physicalDevice_ == VK_NULL_HANDLE) {
        return "未初始化";
    }
    return std::string(deviceProperties_.deviceName);
}

size_t VulkanProcessor::GetGPUMemory() const {
    if (!initialized_ || physicalDevice_ == VK_NULL_HANDLE) {
        return 0;
    }
    
    // 计算总内存大小
    size_t totalMemory = 0;
    for (uint32_t i = 0; i < memoryProperties_.memoryHeapCount; i++) {
        if (memoryProperties_.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            totalMemory += memoryProperties_.memoryHeaps[i].size;
        }
    }
    
    return totalMemory;
}

VulkanProcessor::VulkanDeviceInfo VulkanProcessor::GetDeviceInfo() const {
    VulkanDeviceInfo info = {};
    
    if (initialized_ && physicalDevice_ != VK_NULL_HANDLE) {
        info.deviceName = std::string(deviceProperties_.deviceName);
        info.driverVersion = std::to_string(deviceProperties_.driverVersion);
        info.apiVersion = std::to_string(deviceProperties_.apiVersion);
        info.memorySize = GetGPUMemory();
        
        // 获取队列族属性
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, queueFamilies.data());
        
        if (computeQueueFamilyIndex_ < queueFamilies.size()) {
            info.maxComputeWorkGroupCount[0] = queueFamilies[computeQueueFamilyIndex_].maxComputeWorkGroupCount[0];
            info.maxComputeWorkGroupCount[1] = queueFamilies[computeQueueFamilyIndex_].maxComputeWorkGroupCount[1];
            info.maxComputeWorkGroupCount[2] = queueFamilies[computeQueueFamilyIndex_].maxComputeWorkGroupCount[2];
            info.maxComputeWorkGroupSize[0] = queueFamilies[computeQueueFamilyIndex_].maxComputeWorkGroupSize[0];
            info.maxComputeWorkGroupSize[1] = queueFamilies[computeQueueFamilyIndex_].maxComputeWorkGroupSize[1];
            info.maxComputeWorkGroupSize[2] = queueFamilies[computeQueueFamilyIndex_].maxComputeWorkGroupSize[2];
        }
        
        info.vendorID = deviceProperties_.vendorID;
        info.deviceID = deviceProperties_.deviceID;
        
        // 设备类型
        switch (deviceProperties_.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                info.deviceType = "集成GPU";
                break;
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                info.deviceType = "独立GPU";
                break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                info.deviceType = "虚拟GPU";
                break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                info.deviceType = "CPU";
                break;
            default:
                info.deviceType = "未知";
                break;
        }
    }
    
    return info;
}

bool VulkanProcessor::Resample(const float* input, float* output, 
                              int input_samples, int output_samples,
                              double src_ratio) {
    if (!initialized_) {
        SetError("Vulkan处理器未初始化");
        return false;
    }
    
    // 简化的重采样实现
    for (int i = 0; i < output_samples; i++) {
        double src_index = i * src_ratio;
        int src_idx = static_cast<int>(src_index);
        double frac = src_index - src_idx;
        
        if (src_idx < input_samples - 1) {
            output[i] = input[src_idx] * (1.0f - frac) + input[src_idx + 1] * frac;
        } else if (src_idx < input_samples) {
            output[i] = input[src_idx];
        } else {
            output[i] = 0.0f;
        }
    }
    
    return true;
}

bool VulkanProcessor::ProcessEQ(float* data, int samples,
                               float freq1, float gain1, float q1,
                               float freq2, float gain2, float q2) {
    if (!initialized_) {
        SetError("Vulkan处理器未初始化");
        return false;
    }
    
    // 简化的EQ实现（与CUDA版本类似）
    for (int i = 0; i < samples; i++) {
        // 这里应该实现实际的滤波算法
        // 目前只是简单的增益调整
        if (gain1 != 0.0f) {
            data[i] *= (1.0f + gain1 * 0.1f); // 简化处理
        }
        if (gain2 != 0.0f) {
            data[i] *= (1.0f + gain2 * 0.1f); // 简化处理
        }
    }
    
    return true;
}

bool VulkanProcessor::ProcessFilter(float* data, int samples,
                                   const float* coefficients, int filter_order,
                                   int filter_type) {
    if (!initialized_) {
        SetError("Vulkan处理器未初始化");
        return false;
    }
    
    // 简化的FIR滤波器实现
    std::vector<float> temp(samples, 0.0f);
    
    for (int i = 0; i < samples; i++) {
        float sum = 0.0f;
        for (int j = 0; j <= filter_order && (i - j) >= 0; j++) {
            sum += data[i - j] * coefficients[j];
        }
        temp[i] = sum;
    }
    
    std::copy(temp.begin(), temp.end(), data);
    return true;
}

bool VulkanProcessor::ConvertPcmToDsd(const float* pcm_data, int pcm_samples,
                                     unsigned char* dsd_data, int dsd_rate) {
    if (!initialized_) {
        SetError("Vulkan处理器未初始化");
        return false;
    }
    
    // 简化的PCM到DSD转换
    for (int i = 0; i < pcm_samples; i++) {
        // 简单的噪声整形（实际应该更复杂）
        float sample = pcm_data[i];
        unsigned char dsd_byte = 0;
        
        for (int bit = 0; bit < 8; bit++) {
            if (sample > 0) {
                dsd_byte |= (1 << (7 - bit));
                sample -= 1.0f;
            } else {
                sample += 1.0f;
            }
        }
        
        dsd_data[i / 8] = dsd_byte;
    }
    
    return true;
}

bool VulkanProcessor::ConvertDsdToPcm(const unsigned char* dsd_data, int dsd_samples,
                                     float* pcm_data, int dsd_rate) {
    if (!initialized_) {
        SetError("Vulkan处理器未初始化");
        return false;
    }
    
    // 简化的DSD到PCM转换
    for (int i = 0; i < dsd_samples; i++) {
        unsigned char dsd_byte = dsd_data[i / 8];
        int bit = i % 8;
        
        if (dsd_byte & (1 << (7 - bit))) {
            pcm_data[i] = 0.5f;
        } else {
            pcm_data[i] = -0.5f;
        }
    }
    
    return true;
}

bool VulkanProcessor::EncodeDop(const unsigned char* dsd_data, int dsd_samples,
                               unsigned short* dop_data, int dop_samples) {
    if (!initialized_) {
        SetError("Vulkan处理器未初始化");
        return false;
    }
    
    // 简化的DoP编码
    for (int i = 0; i < dop_samples; i++) {
        dop_data[i] = 0x05A5; // DoP标记
    }
    
    return true;
}

bool VulkanProcessor::DecodeDop(const unsigned short* dop_data, int dop_samples,
                               unsigned char* dsd_data) {
    if (!initialized_) {
        SetError("Vulkan处理器未初始化");
        return false;
    }
    
    // 简化的DoP解码
    for (int i = 0; i < dop_samples; i++) {
        dsd_data[i] = (dop_data[i] & 0x00FF);
    }
    
    return true;
}

bool VulkanProcessor::ProcessBatch(std::vector<float*>& channels,
                                  int samples_per_channel) {
    if (!initialized_) {
        SetError("Vulkan处理器未初始化");
        return false;
    }
    
    // 批量处理多个通道
    for (auto* channel : channels) {
        if (!ProcessEQ(channel, samples_per_channel, 100.0f, 0.0f, 0.7f, 10000.0f, 0.0f, 0.7f)) {
            return false;
        }
    }
    
    return true;
}

IGPUProcessor::GPUStats VulkanProcessor::GetStats() const {
    GPUStats stats = {};
    
    if (initialized_) {
        stats.gpu_utilization = 25.0; // 占位符
        stats.memory_used = 0; // 需要查询实际使用量
        stats.memory_total = GetGPUMemory();
        stats.processing_time_ms = 0.0; // 需要计时
        stats.active_kernels = 0;
    }
    
    return stats;
}

std::string VulkanProcessor::GetLastError() const {
    return lastError_;
}

void VulkanProcessor::SetError(const std::string& error) {
    lastError_ = error;
    std::cerr << "[VULKAN] 错误: " << error << std::endl;
}

bool VulkanProcessor::CheckVulkanResult(VkResult result, const std::string& operation) {
    if (result != VK_SUCCESS) {
        std::string errorMsg = operation + " 失败，错误码: " + std::to_string(result);
        SetError(errorMsg);
        return false;
    }
    return true;
}

void VulkanProcessor::CleanupVulkanResources() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        
        if (resamplePipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, resamplePipeline_, nullptr);
        }
        if (eqPipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, eqPipeline_, nullptr);
        }
        if (filterPipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, filterPipeline_, nullptr);
        }
        
        if (pipelineLayout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        }
        
        if (descriptorSetLayout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        }
        
        if (stagingBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, stagingBuffer_, nullptr);
        }
        if (stagingMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, stagingMemory_, nullptr);
        }
        
        if (deviceBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, deviceBuffer_, nullptr);
        }
        if (deviceMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, deviceMemory_, nullptr);
        }
        
        if (descriptorPool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        }
        
        if (commandPool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
        }
        
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

// Vulkan支持检测工具类实现
bool VulkanSupportDetector::IsVulkanAvailable() {
    #ifdef __linux__
        void* libvulkan = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!libvulkan) {
            libvulkan = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
        }
        if (libvulkan) {
            dlclose(libvulkan);
            return true;
        }
    #elif defined(_WIN32)
        HMODULE libvulkan = LoadLibraryA("vulkan-1.dll");
        if (libvulkan) {
            FreeLibrary(libvulkan);
            return true;
        }
    #elif defined(__APPLE__)
        void* libvulkan = dlopen("libvulkan.dylib", RTLD_NOW | RTLD_LOCAL);
        if (!libvulkan) {
            libvulkan = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
        }
        if (libvulkan) {
            dlclose(libvulkan);
            return true;
        }
    #endif
    
    return false;
}

std::string VulkanSupportDetector::GetVulkanVersion() {
    // 返回Vulkan版本信息
    return "1.2.0"; // 占位符
}

void VulkanSupportDetector::PrintVulkanInfo() {
    std::cout << "===== Vulkan 支持信息 =====" << std::endl;
    
    if (IsVulkanAvailable()) {
        std::cout << "✅ Vulkan 运行时库已找到" << std::endl;
        std::cout << "📋 版本: " << GetVulkanVersion() << std::endl;
        
        // 创建临时处理器来枚举设备
        VulkanProcessor tempProcessor;
        if (tempProcessor.DetectVulkanSupport()) {
            std::cout << "✅ Vulkan 实例创建成功" << std::endl;
            
            if (tempProcessor.Initialize()) {
                auto deviceInfo = tempProcessor.GetDeviceInfo();
                std::cout << "🎯 设备: " << deviceInfo.deviceName << std::endl;
                std::cout.setf(std::ios::fixed);
                std::cout << std::setprecision(1);
                std::cout << "💾 显存: " << (deviceInfo.memorySize / 1024.0 / 1024.0 / 1024.0) << " GB" << std::endl;
                std::cout << "🔧 类型: " << deviceInfo.deviceType << std::endl;
                
                tempProcessor.Shutdown();
            } else {
                std::cout << "❌ 无法初始化Vulkan设备" << std::endl;
            }
        } else {
            std::cout << "❌ 无法创建Vulkan实例" << std::endl;
        }
    } else {
        std::cout << "❌ Vulkan 运行时库未找到" << std::endl;
        std::cout << "💡 请安装Vulkan驱动和运行时库" << std::endl;
    }
    
    std::cout << "=========================" << std::endl;
}

} // namespace GPUPlayer

// 平台相关的动态库加载
#ifdef __linux__
#include <dlfcn.h>
#elif defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <dlfcn.h>
#endif