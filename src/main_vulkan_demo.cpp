#include "IGPUProcessor.h"
#include "gpu/VulkanProcessor.h"
#include <iostream>
#include <memory>
#include <iomanip>
#include <vector>

using namespace GPUPlayer;

// GPU信息展示
void ShowGPUInfo() {
    std::cout << "==========================================" << std::endl;
    std::cout << "     GPU音乐播放器 - GPU检测工具" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << std::endl;
    
    // 获取可用后端列表
    auto availableBackends = GPUProcessorFactory::GetAvailableBackends();
    
    std::cout << "📋 检测到的GPU后端:" << std::endl;
    for (auto backend : availableBackends) {
        std::cout << "  ✅ " << GPUProcessorFactory::GetBackendName(backend) << std::endl;
    }
    
    if (availableBackends.empty()) {
        std::cout << "  ❌ 未检测到支持的GPU后端" << std::endl;
        std::cout << "  💡 请检查GPU驱动和运行时库安装" << std::endl;
    }
    
    std::cout << std::endl;
    
    // 详细检测每个后端
    for (auto backend : availableBackends) {
        std::cout << "🔍 详细检测: " << GPUProcessorFactory::GetBackendName(backend) << std::endl;
        
        auto processor = GPUProcessorFactory::CreateProcessor(backend);
        if (processor && processor->Initialize()) {
            auto stats = processor->GetStats();
            
            std::cout << "  🎯 设备名称: " << processor->GetGPUName() << std::endl;
            std::cout << "  💾 总内存: " << (processor->GetGPUMemory() / 1024 / 1024) << " MB" << std::endl;
            std::cout << "  📊 GPU利用率: " << std::fixed << std::setprecision(1) << stats.gpu_utilization << "%" << std::endl;
            std::cout << "  ⏱️ 处理时间: " << std::setprecision(3) << stats.processing_time_ms << " ms" << std::endl;
            
            // Vulkan特有信息
            if (backend == GPUProcessorFactory::Backend::VULKAN) {
                VulkanProcessor* vulkanProcessor = dynamic_cast<VulkanProcessor*>(processor.get());
                if (vulkanProcessor) {
                    auto vulkanInfo = vulkanProcessor->GetDeviceInfo();
                    std::cout << "  🔧 设备类型: " << vulkanInfo.deviceType << std::endl;
                    std::cout << "  🏭 供应商ID: 0x" << std::hex << vulkanInfo.vendorID << std::dec << std::endl;
                    std::cout << "  🎯 设备ID: 0x" << std::hex << vulkanInfo.deviceID << std::dec << std::endl;
                    std::cout << "  🔢 API版本: " << vulkanInfo.apiVersion << std::endl;
                    std::cout << "  🚛 驱动版本: " << vulkanInfo.driverVersion << std::endl;
                    
                    std::cout << "  📐 最大计算工作组:" << std::endl;
                    std::cout << "    数量: [" << vulkanInfo.maxComputeWorkGroupCount[0] 
                              << ", " << vulkanInfo.maxComputeWorkGroupCount[1]
                              << ", " << vulkanInfo.maxComputeWorkGroupCount[2] << "]" << std::endl;
                    std::cout << "    大小: [" << vulkanInfo.maxComputeWorkGroupSize[0]
                              << ", " << vulkanInfo.maxComputeWorkGroupSize[1]
                              << ", " << vulkanInfo.maxComputeWorkGroupSize[2] << "]" << std::endl;
                }
            }
            
            processor->Shutdown();
        } else {
            std::cout << "  ❌ 无法初始化处理器" << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    // Vulkan特有检测
    std::cout << "🔍 Vulkan运行时检测:" << std::endl;
    VulkanSupportDetector::PrintVulkanInfo();
    std::cout << std::endl;
    
    // 性能测试
    std::cout << "⚡ 简单性能测试:" << std::endl;
    
    const int test_samples = 44100 * 2; // 2秒48kHz音频
    std::vector<float> test_input(test_samples);
    std::vector<float> test_output(test_samples);
    
    // 生成测试信号
    for (int i = 0; i < test_samples; i++) {
        test_input[i] = std::sin(2.0 * M_PI * 440.0 * i / 44100.0) * 0.5f;
    }
    
    for (auto backend : availableBackends) {
        auto processor = GPUProcessorFactory::CreateProcessor(backend);
        if (processor && processor->Initialize()) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // 测试重采样
            bool success = processor->Resample(test_input.data(), test_output.data(), 
                                             test_samples, test_samples, 1.0);
            
            // 测试EQ
            if (success) {
                success = processor->ProcessEQ(test_output.data(), test_samples,
                                             100.0f, 3.0f, 0.7f, 10000.0f, -2.0f, 0.7f);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            std::cout << "  " << GPUProcessorFactory::GetBackendName(backend) << ": ";
            if (success) {
                std::cout << std::fixed << std::setprecision(3) 
                         << (duration.count() / 1000.0) << " ms" << std::endl;
            } else {
                std::cout << "处理失败" << std::endl;
            }
            
            processor->Shutdown();
        }
    }
    
    std::cout << std::endl;
    std::cout << "==========================================" << std::endl;
}

// 后端对比测试
void CompareBackends() {
    std::cout << "🔥 后端性能对比测试:" << std::endl;
    std::cout << std::endl;
    
    const int test_sizes[] = {4410, 44100, 441000}; // 0.1s, 1s, 10s
    const char* test_names[] = {"0.1秒", "1秒", "10秒"};
    
    for (int test = 0; test < 3; test++) {
        int samples = test_sizes[test];
        std::cout << "📊 " << test_names[test] << "音频数据 (" << samples << " 采样):" << std::endl;
        
        std::vector<float> input(samples);
        std::vector<float> output(samples);
        
        // 生成测试信号
        for (int i = 0; i < samples; i++) {
            input[i] = std::sin(2.0 * M_PI * 1000.0 * i / 44100.0) * 0.5f;
        }
        
        auto availableBackends = GPUProcessorFactory::GetAvailableBackends();
        
        for (auto backend : availableBackends) {
            auto processor = GPUProcessorFactory::CreateProcessor(backend);
            if (processor && processor->Initialize()) {
                // 测试重采样
                auto start = std::chrono::high_resolution_clock::now();
                bool success = processor->Resample(input.data(), output.data(), samples, samples, 1.0);
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                
                std::cout << "  " << GPUProcessorFactory::GetBackendName(backend) << "重采样: ";
                if (success) {
                    std::cout << std::fixed << std::setprecision(3) 
                             << (duration.count() / 1000.0) << " ms";
                    
                    // 计算吞吐量
                    double throughput = (samples * sizeof(float)) / (duration.count() / 1000000.0);
                    std::cout << " (" << std::setprecision(1) << (throughput / 1024 / 1024) << " MB/s)";
                    std::cout << std::endl;
                } else {
                    std::cout << "失败" << std::endl;
                }
                
                // 测试EQ
                start = std::chrono::high_resolution_clock::now();
                success = processor->ProcessEQ(output.data(), samples, 100.0f, 6.0f, 0.7f, 8000.0f, -4.0f, 0.7f);
                end = std::chrono::high_resolution_clock::now();
                
                duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                
                std::cout << "  " << GPUProcessorFactory::GetBackendName(backend) << "EQ处理: ";
                if (success) {
                    std::cout << std::fixed << std::setprecision(3) 
                             << (duration.count() / 1000.0) << " ms" << std::endl;
                } else {
                    std::cout << "失败" << std::endl;
                }
                
                processor->Shutdown();
            }
        }
        
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "GPU音乐播放器 - GPU检测工具 v1.0" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << std::endl;
    
    // 解析命令行参数
    bool showComparison = false;
    bool showHelp = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--compare" || arg == "-c") {
            showComparison = true;
        } else if (arg == "--help" || arg == "-h") {
            showHelp = true;
        }
    }
    
    if (showHelp) {
        std::cout << "使用方法: " << argv[0] << " [选项]" << std::endl;
        std::cout << std::endl;
        std::cout << "选项:" << std::endl;
        std::cout << "  --compare, -c     显示后端性能对比测试" << std::endl;
        std::cout << "  --help, -h        显示帮助信息" << std::endl;
        std::cout << std::endl;
        std::cout << "示例:" << std::endl;
        std::cout << "  " << argv[0] << "           # 基本GPU检测" << std::endl;
        std::cout << "  " << argv[0] << " --compare  # 性能对比测试" << std::endl;
        return 0;
    }
    
    // 执行检测
    ShowGPUInfo();
    
    if (showComparison) {
        CompareBackends();
    }
    
    std::cout << "检测完成！" << std::endl;
    return 0;
}