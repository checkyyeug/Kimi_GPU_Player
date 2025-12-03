#include "IGPUProcessor.h"
#include "gpu/CudaProcessor.cuh"
#include "gpu/VulkanProcessor.h"
#include <memory>
#include <vector>
#include <iostream>

namespace GPUPlayer {

std::unique_ptr<IGPUProcessor> GPUProcessorFactory::CreateProcessor(Backend backend) {
    std::cout << "[FACTORY] 创建GPU处理器，后端: " << GetBackendName(backend) << std::endl;
    
    switch (backend) {
        case Backend::CUDA: {
            auto cudaProcessor = std::make_unique<CudaProcessor>();
            if (cudaProcessor->IsGPUSupported()) {
                std::cout << "[FACTORY] CUDA后端可用" << std::endl;
                return cudaProcessor;
            } else {
                std::cout << "[FACTORY] CUDA后端不可用，尝试其他后端" << std::endl;
            }
            break;
        }
        
        case Backend::VULKAN: {
            auto vulkanProcessor = std::make_unique<VulkanProcessor>();
            if (vulkanProcessor->DetectVulkanSupport()) {
                std::cout << "[FACTORY] Vulkan后端可用" << std::endl;
                return vulkanProcessor;
            } else {
                std::cout << "[FACTORY] Vulkan后端不可用" << std::endl;
            }
            break;
        }
        
        case Backend::OPENCL: {
            // OpenCL实现将在后续添加
            std::cout << "[FACTORY] OpenCL后端尚未实现" << std::endl;
            break;
        }
        
        case Backend::AUTO: {
            // 自动选择最佳后端
            std::cout << "[FACTORY] 自动选择最佳GPU后端" << std::endl;
            
            // 1. 首先尝试CUDA (性能最佳)
            auto cudaProcessor = std::make_unique<CudaProcessor>();
            if (cudaProcessor->IsGPUSupported()) {
                std::cout << "[FACTORY] 自动选择: CUDA后端" << std::endl;
                return cudaProcessor;
            }
            
            // 2. 然后尝试Vulkan (跨平台，性能好)
            auto vulkanProcessor = std::make_unique<VulkanProcessor>();
            if (vulkanProcessor->DetectVulkanSupport()) {
                std::cout << "[FACTORY] 自动选择: Vulkan后端" << std::endl;
                return vulkanProcessor;
            }
            
            // 3. 最后尝试OpenCL (最通用)
            // 这里将添加OpenCL检查
            std::cout << "[FACTORY] CUDA和Vulkan都不可用，需要OpenCL支持" << std::endl;
            
            break;
        }
    }
    
    // 如果指定后端失败，返回nullptr
    std::cout << "[FACTORY] 警告: 请求的GPU后端不可用" << std::endl;
    return nullptr;
}

std::vector<GPUProcessorFactory::Backend> GPUProcessorFactory::GetAvailableBackends() {
    std::vector<Backend> backends;
    
    // 检查CUDA
    auto cudaProcessor = std::make_unique<CudaProcessor>();
    if (cudaProcessor->IsGPUSupported()) {
        backends.push_back(Backend::CUDA);
    }
    
    // 检查Vulkan
    VulkanSupportDetector vulkanDetector;
    if (VulkanSupportDetector::IsVulkanAvailable()) {
        backends.push_back(Backend::VULKAN);
    }
    
    // 检查OpenCL (待实现)
    // backends.push_back(Backend::OPENCL);
    
    return backends;
}

std::string GPUProcessorFactory::GetBackendName(Backend backend) {
    switch (backend) {
        case Backend::CUDA:    return "CUDA";
        case Backend::OPENCL:  return "OpenCL";
        case Backend::VULKAN:  return "Vulkan";
        case Backend::AUTO:    return "自动选择";
        default:               return "未知";
    }
}

} // namespace GPUPlayer