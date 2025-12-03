#include "gpu/VulkanDetector.h"
#include <iostream>
#include <iomanip>

using namespace GPUPlayer;

void PrintHeader() {
    std::cout << "==========================================" << std::endl;
    std::cout << "     GPU音乐播放器 - GPU检测工具 v2.0" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << std::endl;
}

void PrintUsage(const char* programName) {
    std::cout << "使用方法: " << programName << " [选项]" << std::endl;
    std::cout << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --vulkan, -v    只显示Vulkan信息" << std::endl;
    std::cout << "  --all, -a       显示所有GPU后端信息" << std::endl;
    std::cout << "  --json, -j      以JSON格式输出" << std::endl;
    std::cout << "  --help, -h      显示帮助信息" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  " << programName << "           # 基本GPU检测" << std::endl;
    std::cout << "  " << programName << " --vulkan   # 详细Vulkan检测" << std::endl;
    std::cout << "  " << programName << " --all      # 所有后端检测" << std::endl;
    std::cout << "  " << programName << " --json     # JSON格式输出" << std::endl;
}

void PrintVulkanDetailedInfo() {
    std::cout << "🔍 Vulkan详细检测:" << std::endl;
    std::cout << std::endl;
    
    VulkanDetector::PrintVulkanInfo();
    
    std::cout << std::endl;
    std::cout << "🔧 技术信息:" << std::endl;
    
    auto info = VulkanDetector::GetVulkanInfo();
    
    if (info.available) {
        std::cout << "  📋 API版本: " << info.version << std::endl;
        std::cout << "  🚛 驱动信息: " << info.driverInfo << std::endl;
        
        if (!info.devices.empty()) {
            std::cout << "  🎯 设备数量: " << info.devices.size() << std::endl;
            std::cout << "  📱 设备列表:" << std::endl;
            for (size_t i = 0; i < info.devices.size(); i++) {
                std::cout << "    [" << i << "] " << info.devices[i] << std::endl;
            }
        }
        
        // 检测建议
        std::cout << std::endl;
        std::cout << "💡 使用建议:" << std::endl;
        std::cout << "  ✅ Vulkan后端可以正常使用" << std::endl;
        std::cout << "  🎵 适合音频处理的GPU加速" << std::endl;
        std::cout << "  🔧 支持并行计算和内存管理" << std::endl;
    } else {
        std::cout << "  ❌ Vulkan不可用" << std::endl;
        std::cout << "  🔧 需要安装Vulkan运行时库" << std::endl;
    }
}

void PrintAllGPUInfo() {
    std::cout << "🔍 全GPU后端检测:" << std::endl;
    std::cout << std::endl;
    
    GPUDetector::PrintGPUReport();
    
    std::cout << std::endl;
    std::cout << "⚡ 性能建议:" << std::endl;
    
    auto gpuList = GPUDetector::DetectAllGPUs();
    
    if (gpuList.empty()) {
        std::cout << "  ⚠️  未检测到GPU加速支持" << std::endl;
        std::cout << "  💡 程序将使用CPU处理模式" << std::endl;
        return;
    }
    
    // 分析推荐配置
    bool hasCUDA = false;
    bool hasVulkan = false;
    
    for (const auto& gpu : gpuList) {
        if (gpu.available && gpu.backend == "CUDA") {
            hasCUDA = true;
        } else if (gpu.available && gpu.backend == "Vulkan") {
            hasVulkan = true;
        }
    }
    
    if (hasCUDA) {
        std::cout << "  🥇 推荐: CUDA后端" << std::endl;
        std::cout << "     • 最佳性能表现" << std::endl;
        std::cout << "     • 专为NVIDIA GPU优化" << std::endl;
        std::cout << "     • 成熟的音频处理生态" << std::endl;
    }
    
    if (hasVulkan) {
        std::cout << "  🥈 推荐: Vulkan后端" << std::endl;
        std::cout << "     • 跨平台兼容性" << std::endl;
        std::cout << "     • 现代GPU架构支持" << std::endl;
        std::cout << "     • 低延迟音频处理" << std::endl;
    }
    
    if (!hasCUDA && !hasVulkan) {
        std::cout << "  ⚠️  无GPU加速可用" << std::endl;
        std::cout << "  💡 将使用CPU处理模式" << std::endl;
        std::cout << "  🔧 建议安装GPU驱动和计算库" << std::endl;
    }
}

void PrintJSONOutput() {
    std::cout << "{" << std::endl;
    
    // Vulkan信息
    auto vulkanInfo = VulkanDetector::GetVulkanInfo();
    std::cout << "  \"vulkan\": {" << std::endl;
    std::cout << "    \"available\": " << (vulkanInfo.available ? "true" : "false") << "," << std::endl;
    std::cout << "    \"version\": \"" << vulkanInfo.version << "\"," << std::endl;
    std::cout << "    \"driver\": \"" << vulkanInfo.driverInfo << "\"," << std::endl;
    std::cout << "    \"devices\": [";
    
    for (size_t i = 0; i < vulkanInfo.devices.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << "\"" << vulkanInfo.devices[i] << "\"";
    }
    std::cout << "]" << std::endl;
    std::cout << "  }," << std::endl;
    
    // 所有GPU信息
    auto gpuList = GPUDetector::DetectAllGPUs();
    std::cout << "  \"backends\": [" << std::endl;
    
    for (size_t i = 0; i < gpuList.size(); i++) {
        const auto& gpu = gpuList[i];
        std::cout << "    {" << std::endl;
        std::cout << "      \"backend\": \"" << gpu.backend << "\"," << std::endl;
        std::cout << "      \"available\": " << (gpu.available ? "true" : "false") << "," << std::endl;
        std::cout << "      \"device\": \"" << gpu.deviceName << "\"," << std::endl;
        std::cout << "      \"driver_version\": \"" << gpu.driverVersion << "\"," << std::endl;
        std::cout << "      \"memory_mb\": " << (gpu.memorySize / 1024 / 1024) << "," << std::endl;
        std::cout << "      \"error\": \"" << gpu.error << "\"" << std::endl;
        std::cout << "    }";
        if (i < gpuList.size() - 1) std::cout << ",";
        std::cout << std::endl;
    }
    
    std::cout << "  ]" << std::endl;
    std::cout << "}" << std::endl;
}

int main(int argc, char* argv[]) {
    // 解析命令行参数
    bool showVulkanOnly = false;
    bool showAll = false;
    bool jsonOutput = false;
    bool showHelp = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--vulkan" || arg == "-v") {
            showVulkanOnly = true;
        } else if (arg == "--all" || arg == "-a") {
            showAll = true;
        } else if (arg == "--json" || arg == "-j") {
            jsonOutput = true;
        } else if (arg == "--help" || arg == "-h") {
            showHelp = true;
        }
    }
    
    if (showHelp) {
        PrintUsage(argv[0]);
        return 0;
    }
    
    if (jsonOutput) {
        PrintJSONOutput();
        return 0;
    }
    
    if (!showVulkanOnly && !showAll) {
        // 默认输出
        PrintHeader();
        
        // 基本GPU检测
        std::cout << "🔍 基本GPU检测:" << std::endl;
        GPUDetector::PrintGPUReport();
        
        std::cout << std::endl;
        std::cout << "💡 使用 '--vulkan' 查看详细Vulkan信息" << std::endl;
        std::cout << "💡 使用 '--all' 查看所有后端详细信息" << std::endl;
        std::cout << "💡 使用 '--json' 获取机器可读格式" << std::endl;
        
    } else if (showVulkanOnly) {
        PrintHeader();
        PrintVulkanDetailedInfo();
        
    } else if (showAll) {
        PrintHeader();
        PrintAllGPUInfo();
    }
    
    return 0;
}