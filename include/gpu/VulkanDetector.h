#pragma once

#include <string>
#include <vector>

namespace GPUPlayer {

// Vulkan支持检测器 - 轻量级版本
class VulkanDetector {
public:
    struct VulkanInfo {
        bool available = false;
        std::string version = "1.2.0";
        std::string driverInfo;
        std::vector<std::string> devices;
        std::string errorMessage;
    };
    
    // 基础检测
    static bool IsVulkanAvailable();
    static VulkanInfo GetVulkanInfo();
    static void PrintVulkanInfo();
    
    // 设备枚举（简化版本）
    static std::vector<std::string> EnumerateDevices();
    
private:
    static bool CheckVulkanLibrary();
    static std::string GetDriverInfo();
};

// GPU通用检测器
class GPUDetector {
public:
    struct GPUInfo {
        std::string backend;  // "CUDA", "Vulkan", "OpenCL"
        std::string deviceName;
        size_t memorySize;    // 字节
        std::string driverVersion;
        bool available;
        std::string error;
    };
    
    // 检测所有可用的GPU后端
    static std::vector<GPUInfo> DetectAllGPUs();
    static void PrintGPUReport();
    
private:
    static GPUInfo DetectCUDA();
    static GPUInfo DetectVulkan();
    static GPUInfo DetectOpenCL();
};

} // namespace GPUPlayer