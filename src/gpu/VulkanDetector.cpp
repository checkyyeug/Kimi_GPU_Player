#include "gpu/VulkanDetector.h"
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __linux__
#include <dlfcn.h>
#elif defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace GPUPlayer {

bool VulkanDetector::IsVulkanAvailable() {
    return CheckVulkanLibrary();
}

bool VulkanDetector::CheckVulkanLibrary() {
#ifdef __linux__
    // 尝试加载Vulkan库
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

VulkanDetector::VulkanInfo VulkanDetector::GetVulkanInfo() {
    VulkanInfo info;
    
    info.available = IsVulkanAvailable();
    
    if (info.available) {
        info.driverInfo = GetDriverInfo();
        info.devices = EnumerateDevices();
    } else {
        info.errorMessage = "Vulkan运行时库未找到";
    }
    
    return info;
}

std::string VulkanDetector::GetDriverInfo() {
    std::string driverInfo;
    
#ifdef __linux__
    // 尝试读取NVIDIA驱动信息
    std::ifstream nvidiaVersion("/proc/driver/nvidia/version");
    if (nvidiaVersion.is_open()) {
        std::string line;
        if (std::getline(nvidiaVersion, line)) {
            driverInfo = "NVIDIA: " + line;
        }
        nvidiaVersion.close();
    }
    
    // 尝试读取AMDGPU信息
    if (driverInfo.empty()) {
        std::ifstream amdgpuInfo("/sys/class/drm/card0/device/vendor");
        if (amdgpuInfo.is_open()) {
            std::string vendor;
            amdgpuInfo >> vendor;
            if (vendor == "0x1002") { // AMD
                driverInfo = "AMD GPU detected";
            }
            amdgpuInfo.close();
        }
    }
    
    // 尝试检测Intel GPU
    if (driverInfo.empty()) {
        std::ifstream intelInfo("/sys/class/drm/card0/device/vendor");
        if (intelInfo.is_open()) {
            std::string vendor;
            intelInfo >> vendor;
            if (vendor == "0x8086") { // Intel
                driverInfo = "Intel GPU detected";
            }
            intelInfo.close();
        }
    }
#endif
    
    if (driverInfo.empty()) {
        driverInfo = "通用Vulkan驱动";
    }
    
    return driverInfo;
}

std::vector<std::string> VulkanDetector::EnumerateDevices() {
    std::vector<std::string> devices;
    
    if (!IsVulkanAvailable()) {
        return devices;
    }
    
    // 简化版本的设备枚举
    // 实际应该通过Vulkan API获取设备列表
    
#ifdef __linux__
    // 通过DRM子系统检测GPU设备
    std::ifstream cards("/proc/driver/nvidia/gpus");
    if (cards.is_open()) {
        std::string line;
        int gpuIndex = 0;
        while (std::getline(cards, line)) {
            if (line.find("Model:")) {
                devices.push_back("NVIDIA GPU " + std::to_string(gpuIndex));
                gpuIndex++;
            }
        }
        cards.close();
    }
    
    // 检测DRM设备
    if (devices.empty()) {
        std::ifstream drmCards("/sys/class/drm/version");
        if (drmCards.is_open()) {
            std::string version;
            std::getline(drmCards, version);
            
            // 简单检测GPU数量
            for (int i = 0; i < 4; i++) {
                std::string gpuPath = "/sys/class/drm/card" + std::to_string(i) + "/device/vendor";
                std::ifstream vendorFile(gpuPath);
                if (vendorFile.is_open()) {
                    std::string vendor;
                    vendorFile >> vendor;
                    
                    std::string deviceName;
                    if (vendor == "0x10de") {
                        deviceName = "NVIDIA GPU " + std::to_string(i);
                    } else if (vendor == "0x1002") {
                        deviceName = "AMD GPU " + std::to_string(i);
                    } else if (vendor == "0x8086") {
                        deviceName = "Intel GPU " + std::to_string(i);
                    } else {
                        deviceName = "Unknown GPU " + std::to_string(i) + " (Vendor: " + vendor + ")";
                    }
                    
                    devices.push_back(deviceName);
                    vendorFile.close();
                }
            }
            drmCards.close();
        }
    }
#endif
    
    if (devices.empty()) {
        devices.push_back("通用Vulkan兼容设备");
    }
    
    return devices;
}

void VulkanDetector::PrintVulkanInfo() {
    std::cout << "===== Vulkan 支持信息 =====" << std::endl;
    
    auto info = GetVulkanInfo();
    
    if (info.available) {
        std::cout << "✅ Vulkan 运行时库已找到" << std::endl;
        std::cout << "📋 版本: " << info.version << std::endl;
        std::cout << "🚛 驱动: " << info.driverInfo << std::endl;
        
        if (!info.devices.empty()) {
            std::cout << "🎯 检测到的设备:" << std::endl;
            for (const auto& device : info.devices) {
                std::cout << "  • " << device << std::endl;
            }
        }
        
        std::cout << "🎵 GPU音乐播放器的Vulkan后端可以正常工作" << std::endl;
    } else {
        std::cout << "❌ Vulkan 运行时库未找到" << std::endl;
        std::cout << "💡 " << info.errorMessage << std::endl;
        std::cout << std::endl;
        std::cout << "🔧 安装方法:" << std::endl;
        std::cout << "  Ubuntu/Debian: sudo apt install libvulkan1 vulkan-tools" << std::endl;
        std::cout << "  通用: 从 https://vulkan.lunarg.com/ 下载Vulkan SDK" << std::endl;
        std::cout << "  NVIDIA用户: 确保安装了最新版NVIDIA驱动" << std::endl;
        std::cout << "  AMD用户: 确保安装了最新版AMDGPU驱动" << std::endl;
        std::cout << "  Intel用户: 确保安装了最新版Intel显卡驱动" << std::endl;
    }
    
    std::cout << "=========================" << std::endl;
}

// GPUDetector实现
std::vector<GPUDetector::GPUInfo> GPUDetector::DetectAllGPUs() {
    std::vector<GPUInfo> gpuList;
    
    // 检测CUDA
    auto cudaInfo = DetectCUDA();
    if (cudaInfo.available) {
        gpuList.push_back(cudaInfo);
    }
    
    // 检测Vulkan
    auto vulkanInfo = DetectVulkan();
    if (vulkanInfo.available) {
        gpuList.push_back(vulkanInfo);
    }
    
    // 检测OpenCL (简化版本)
    auto openclInfo = DetectOpenCL();
    if (openclInfo.available) {
        gpuList.push_back(openclInfo);
    }
    
    return gpuList;
}

GPUDetector::GPUInfo GPUDetector::DetectCUDA() {
    GPUInfo info;
    info.backend = "CUDA";
    info.available = false;
    
    // 这里应该调用CUDA检测代码
    // 目前返回不可用状态
    info.error = "CUDA检测未实现";
    
    return info;
}

GPUDetector::GPUInfo GPUDetector::DetectVulkan() {
    GPUInfo info;
    info.backend = "Vulkan";
    info.available = false;
    
    if (VulkanDetector::IsVulkanAvailable()) {
        info.available = true;
        
        auto vulkanInfo = VulkanDetector::GetVulkanInfo();
        if (!vulkanInfo.devices.empty()) {
            info.deviceName = vulkanInfo.devices[0];
        } else {
            info.deviceName = "Vulkan兼容设备";
        }
        
        info.driverVersion = vulkanInfo.version;
        info.memorySize = 0; // 需要更复杂的检测
    } else {
        info.error = "Vulkan运行时库未找到";
    }
    
    return info;
}

GPUDetector::GPUInfo GPUDetector::DetectOpenCL() {
    GPUInfo info;
    info.backend = "OpenCL";
    info.available = false;
    
    // OpenCL检测将在后续实现
    info.error = "OpenCL检测未实现";
    
    return info;
}

void GPUDetector::PrintGPUReport() {
    std::cout << "==========================================" << std::endl;
    std::cout << "     GPU音乐播放器 - GPU检测报告" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << std::endl;
    
    auto gpuList = DetectAllGPUs();
    
    if (gpuList.empty()) {
        std::cout << "❌ 未检测到支持的GPU后端" << std::endl;
        std::cout << std::endl;
        std::cout << "🔧 建议:" << std::endl;
        std::cout << "1. 确保安装了最新的GPU驱动程序" << std::endl;
        std::cout << "2. 安装相应的GPU计算库:" << std::endl;
        std::cout << "   • CUDA: NVIDIA GPU + CUDA Toolkit" << std::endl;
        std::cout << "   • Vulkan: 任何现代GPU + Vulkan运行时" << std::endl;
        std::cout << "   • OpenCL: 通用GPU计算库" << std::endl;
        std::cout << "3. 检查硬件兼容性" << std::endl;
        return;
    }
    
    std::cout << "✅ 检测到 " << gpuList.size() << " 个GPU后端" << std::endl;
    std::cout << std::endl;
    
    for (const auto& gpu : gpuList) {
        std::cout << "🎯 " << gpu.backend << " 后端:" << std::endl;
        
        if (gpu.available) {
            std::cout << "  ✅ 状态: 可用" << std::endl;
            std::cout << "  🎯 设备: " << gpu.deviceName << std::endl;
            
            if (gpu.memorySize > 0) {
                std::cout << "  💾 内存: " << (gpu.memorySize / 1024 / 1024) << " MB" << std::endl;
            }
            
            if (!gpu.driverVersion.empty()) {
                std::cout << "  🚛 驱动: " << gpu.driverVersion << std::endl;
            }
        } else {
            std::cout << "  ❌ 状态: 不可用" << std::endl;
            std::cout << "  🔧 原因: " << gpu.error << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    // 推荐最佳后端
    std::cout << "🏆 推荐配置:" << std::endl;
    
    bool hasCUDA = false;
    bool hasVulkan = false;
    
    for (const auto& gpu : gpuList) {
        if (gpu.available) {
            if (gpu.backend == "CUDA") {
                hasCUDA = true;
            } else if (gpu.backend == "Vulkan") {
                hasVulkan = true;
            }
        }
    }
    
    if (hasCUDA) {
        std::cout << "  🥇 首选: CUDA后端 (性能最佳)" << std::endl;
    }
    
    if (hasVulkan) {
        std::cout << "  🥈 备选: Vulkan后端 (跨平台，性能好)" << std::endl;
    }
    
    if (!hasCUDA && !hasVulkan) {
        std::cout << "  ⚠️  警告: 未找到高性能GPU后端" << std::endl;
        std::cout << "  💡 建议: 安装GPU驱动和相应的计算库" << std::endl;
    }
    
    std::cout << "==========================================" << std::endl;
}

} // namespace GPUPlayer