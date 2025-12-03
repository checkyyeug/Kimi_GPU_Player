#pragma once

#include <memory>
#include <vector>

namespace GPUPlayer {

// GPU处理器接口，支持多后端实现
class IGPUProcessor {
public:
    virtual ~IGPUProcessor() = default;
    
    // 初始化和清理
    virtual bool Initialize() = 0;
    virtual void Shutdown() = 0;
    
    // 处理能力查询
    virtual bool IsGPUSupported() const = 0;
    virtual std::string GetGPUName() const = 0;
    virtual size_t GetGPUMemory() const = 0;
    
    // 重采样
    virtual bool Resample(const float* input, float* output, 
                         int input_samples, int output_samples,
                         double src_ratio) = 0;
    
    // 2段参数均衡器
    virtual bool ProcessEQ(float* data, int samples,
                          float freq1, float gain1, float q1,
                          float freq2, float gain2, float q2) = 0;
    
    // 数字滤波器
    virtual bool ProcessFilter(float* data, int samples,
                              const float* coefficients, int filter_order,
                              int filter_type) = 0;
    
    // PCM到DSD转换
    virtual bool ConvertPcmToDsd(const float* pcm_data, int pcm_samples,
                                unsigned char* dsd_data, int dsd_rate) = 0;
    
    // DSD到PCM转换
    virtual bool ConvertDsdToPcm(const unsigned char* dsd_data, int dsd_samples,
                                float* pcm_data, int dsd_rate) = 0;
    
    // DoP (DSD over PCM) 封装
    virtual bool EncodeDop(const unsigned char* dsd_data, int dsd_samples,
                          unsigned short* dop_data, int dop_samples) = 0;
    
    // DoP解码
    virtual bool DecodeDop(const unsigned short* dop_data, int dop_samples,
                          unsigned char* dsd_data) = 0;
    
    // 批量处理优化
    virtual bool ProcessBatch(std::vector<float*>& channels,
                             int samples_per_channel) = 0;
    
    // 性能统计
    struct GPUStats {
        double gpu_utilization;
        size_t memory_used;
        size_t memory_total;
        double processing_time_ms;
        int active_kernels;
    };
    virtual GPUStats GetStats() const = 0;
    
    // 错误处理
    virtual std::string GetLastError() const = 0;
};

// GPU处理器工厂
class GPUProcessorFactory {
public:
    enum class Backend {
        CUDA,      // NVIDIA CUDA
        OPENCL,    // OpenCL (跨平台)
        VULKAN,    // Vulkan Compute
        AUTO       // 自动选择最佳后端
    };
    
    static std::unique_ptr<IGPUProcessor> CreateProcessor(Backend backend = Backend::AUTO);
    static std::vector<Backend> GetAvailableBackends();
    static std::string GetBackendName(Backend backend);
};

} // namespace GPUPlayer