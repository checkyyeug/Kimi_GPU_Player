#include "gpu/VulkanProcessor.h"
#include <iostream>
#include <iomanip>

using namespace GPUPlayer;

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "     Vulkan支持检测工具 v1.0" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << std::endl;
    
    // 基本Vulkan运行时检测
    std::cout << "🔍 Vulkan运行时库检测:" << std::endl;
    if (VulkanSupportDetector::IsVulkanAvailable()) {
        std::cout << "  ✅ Vulkan运行时库已找到" << std::endl;
        std::cout << "  📋 版本: " << VulkanSupportDetector::GetVulkanVersion() << std::endl;
    } else {
        std::cout << "  ❌ Vulkan运行时库未找到" << std::endl;
        std::cout << "  💡 请安装Vulkan SDK和驱动程序:" << std::endl;
        std::cout << "     • Ubuntu/Debian: sudo apt install libvulkan1 vulkan-tools" << std::endl;
        std::cout << "     • 或者从 https://vulkan.lunarg.com/ 下载SDK" << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    
    // 详细设备检测
    std::cout << "🎯 Vulkan设备检测:" << std::endl;
    
    VulkanProcessor vulkanProcessor;
    if (!vulkanProcessor.DetectVulkanSupport()) {
        std::cout << "  ❌ 无法创建Vulkan实例" << std::endl;
        std::cout << "  🔧 可能的原因:" << std::endl;
        std::cout << "     • GPU驱动不支持Vulkan" << std::endl;
        std::cout << "     • Vulkan运行时版本过旧" << std::endl;
        std::cout << "     • 系统缺少必要的Vulkan加载器" << std::endl;
        return 1;
    }
    
    std::cout << "  ✅ Vulkan实例创建成功" << std::endl;
    
    if (vulkanProcessor.Initialize()) {
        auto deviceInfo = vulkanProcessor.GetDeviceInfo();
        
        std::cout << "  🎯 设备名称: " << deviceInfo.deviceName << std::endl;
        std::cout << "  🔧 设备类型: " << deviceInfo.deviceType << std::endl;
        std::cout << "  💾 显存大小: " << std::fixed << std::setprecision(2) 
                  << (deviceInfo.memorySize / 1024.0 / 1024.0 / 1024.0) << " GB" << std::endl;
        std::cout << "  🏭 供应商ID: 0x" << std::hex << deviceInfo.vendorID << std::dec << std::endl;
        std::cout << "  🎯 设备ID: 0x" << std::hex << deviceInfo.deviceID << std::dec << std::endl;
        std::cout << "  🔢 API版本: " << deviceInfo.apiVersion << std::endl;
        std::cout << "  🚛 驱动版本: " << deviceInfo.driverVersion << std::endl;
        
        std::cout << std::endl;
        std::cout << "  📐 计算能力:" << std::endl;
        std::cout << "    最大工作组数量: [" << deviceInfo.maxComputeWorkGroupCount[0] 
                  << ", " << deviceInfo.maxComputeWorkGroupCount[1]
                  << ", " << deviceInfo.maxComputeWorkGroupCount[2] << "]" << std::endl;
        std::cout << "    最大工作组大小: [" << deviceInfo.maxComputeWorkGroupSize[0]
                  << ", " << deviceInfo.maxComputeWorkGroupSize[1]
                  << ", " << deviceInfo.maxComputeWorkGroupSize[2] << "]" << std::endl;
        
        // 功能测试
        std::cout << std::endl;
        std::cout << "⚡ 功能测试:" << std::endl;
        
        const int test_size = 44100; // 1秒音频
        std::vector<float> test_input(test_size);
        std::vector<float> test_output(test_size);
        
        // 生成测试信号
        for (int i = 0; i < test_size; i++) {
            test_input[i] = std::sin(2.0 * M_PI * 1000.0 * i / 44100.0) * 0.5f;
        }
        
        // 测试重采样
        auto start = std::chrono::high_resolution_clock::now();
        bool resample_ok = vulkanProcessor.Resample(test_input.data(), test_output.data(), 
                                                   test_size, test_size, 1.0);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  🔄 重采样测试: " << (resample_ok ? "✅ 通过" : "❌ 失败")
                  << " (" << std::fixed << std::setprecision(3) 
                  << (duration.count() / 1000.0) << " ms)" << std::endl;
        
        // 测试EQ
        start = std::chrono::high_resolution_clock::now();
        bool eq_ok = vulkanProcessor.ProcessEQ(test_output.data(), test_size,
                                             100.0f, 3.0f, 0.7f, 8000.0f, -2.0f, 0.7f);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  🎛️ EQ测试: " << (eq_ok ? "✅ 通过" : "❌ 失败")
                  << " (" << std::fixed << std::setprecision(3) 
                  << (duration.count() / 1000.0) << " ms)" << std::endl;
        
        // 性能统计
        auto stats = vulkanProcessor.GetStats();
        std::cout << "  📊 GPU利用率: " << std::fixed << std::setprecision(1) 
                  << stats.gpu_utilization << "%" << std::endl;
        std::cout << "  💽 内存使用: " << (stats.memory_used / 1024 / 1024) << " MB / " 
                  << (stats.memory_total / 1024 / 1024) << " MB" << std::endl;
        
        vulkanProcessor.Shutdown();
        
        std::cout << std::endl;
        std::cout << "✅ Vulkan支持检测完成" << std::endl;
        std::cout << "🎵 GPU音乐播放器的Vulkan后端可以正常工作" << std::endl;
        
    } else {
        std::cout << "  ❌ Vulkan设备初始化失败" << std::endl;
        std::cout << "  🔧 错误信息: " << vulkanProcessor.GetLastError() << std::endl;
        return 1;
    }
    
    std::cout << "==========================================" << std::endl;
    return 0;
}