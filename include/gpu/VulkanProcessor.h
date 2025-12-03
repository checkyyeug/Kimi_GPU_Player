#pragma once

#include "IGPUProcessor.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <memory>

namespace GPUPlayer {

class VulkanProcessor : public IGPUProcessor {
public:
    VulkanProcessor();
    ~VulkanProcessor() override;
    
    // IGPUProcessor接口实现
    bool Initialize() override;
    void Shutdown() override;
    
    bool IsGPUSupported() const override;
    std::string GetGPUName() const override;
    size_t GetGPUMemory() const override;
    
    bool Resample(const float* input, float* output, 
                 int input_samples, int output_samples,
                 double src_ratio) override;
    
    bool ProcessEQ(float* data, int samples,
                  float freq1, float gain1, float q1,
                  float freq2, float gain2, float q2) override;
    
    bool ProcessFilter(float* data, int samples,
                      const float* coefficients, int filter_order,
                      int filter_type) override;
    
    bool ConvertPcmToDsd(const float* pcm_data, int pcm_samples,
                        unsigned char* dsd_data, int dsd_rate) override;
    
    bool ConvertDsdToPcm(const unsigned char* dsd_data, int dsd_samples,
                        float* pcm_data, int dsd_rate) override;
    
    bool EncodeDop(const unsigned char* dsd_data, int dsd_samples,
                  unsigned short* dop_data, int dop_samples) override;
    
    bool DecodeDop(const unsigned short* dop_data, int dop_samples,
                  unsigned char* dsd_data) override;
    
    bool ProcessBatch(std::vector<float*>& channels,
                     int samples_per_channel) override;
    
    GPUStats GetStats() const override;
    std::string GetLastError() const override;
    
    // Vulkan特有功能
    bool DetectVulkanSupport();
    std::vector<std::string> GetAvailableDevices() const;
    bool SelectDevice(int deviceIndex);
    
    // Vulkan信息查询
    struct VulkanDeviceInfo {
        std::string deviceName;
        std::string driverVersion;
        std::string apiVersion;
        size_t memorySize;
        uint32_t maxComputeWorkGroupCount[3];
        uint32_t maxComputeWorkGroupSize[3];
        uint32_t maxComputeSharedMemorySize;
        bool supportsFloat64;
        bool supportsInt64;
        uint32_t vendorID;
        uint32_t deviceID;
        std::string deviceType;
    };
    
    VulkanDeviceInfo GetDeviceInfo() const;
    
private:
    // Vulkan实例和设备
    VkInstance instance_;
    VkPhysicalDevice physicalDevice_;
    VkDevice device_;
    VkQueue computeQueue_;
    VkCommandPool commandPool_;
    
    // 设备信息
    VkPhysicalDeviceProperties deviceProperties_;
    VkPhysicalDeviceMemoryProperties memoryProperties_;
    
    // 计算管线
    VkDescriptorSetLayout descriptorSetLayout_;
    VkPipelineLayout pipelineLayout_;
    VkPipeline resamplePipeline_;
    VkPipeline eqPipeline_;
    VkPipeline filterPipeline_;
    
    // 缓冲区和内存
    VkBuffer stagingBuffer_;
    VkDeviceMemory stagingMemory_;
    VkBuffer deviceBuffer_;
    VkDeviceMemory deviceMemory_;
    
    // 描述符集合
    VkDescriptorSet descriptorSet_;
    VkDescriptorPool descriptorPool_;
    
    // 状态信息
    bool initialized_;
    std::string lastError_;
    uint32_t computeQueueFamilyIndex_;
    
    // 着色器模块
    VkShaderModule resampleShaderModule_;
    VkShaderModule eqShaderModule_;
    VkShaderModule filterShaderModule_;
    
    // 私有方法
    bool CreateInstance();
    bool SelectPhysicalDevice();
    bool CreateDevice();
    bool CreateCommandPool();
    bool CreateDescriptorSetLayout();
    bool CreatePipelineLayout();
    bool CreateComputePipelines();
    bool CreateBuffers();
    bool CreateDescriptorPool();
    bool AllocateDescriptorSet();
    
    // 着色器代码
    std::vector<uint32_t> CompileShader(const std::string& shaderCode, const std::string& entryPoint);
    bool CreateShaderModule(const std::vector<uint32_t>& code, VkShaderModule* shaderModule);
    
    // 工具方法
    uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    bool CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, 
                     VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    bool CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    
    // GPU内核执行
    bool ExecuteComputePipeline(VkPipeline pipeline, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
    
    // 清理资源
    void CleanupVulkanResources();
    
    // 错误处理
    void SetError(const std::string& error);
    bool CheckVulkanResult(VkResult result, const std::string& operation);
};

// Vulkan支持检测工具类
class VulkanSupportDetector {
public:
    static bool IsVulkanAvailable();
    static std::string GetVulkanVersion();
    static std::vector<VulkanProcessor::VulkanDeviceInfo> EnumerateDevices();
    static void PrintVulkanInfo();
    
private:
    static bool initialized_;
    static void Initialize();
};

} // namespace GPUPlayer