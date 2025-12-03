#pragma once

#include "IGPUProcessor.h"
#include <cuda_runtime.h>
#include <string>

namespace GPUPlayer {

class CudaProcessor : public IGPUProcessor {
public:
    CudaProcessor();
    ~CudaProcessor() override;
    
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
    
private:
    void InitFilterKernel();
    
    bool initialized_ = false;
    std::string device_name_;
    size_t memory_total_ = 0;
    std::string last_error_;
    
    // CUDA资源
    cudaStream_t stream_ = nullptr;
    float* d_filter_kernel = nullptr;
    int d_filter_size = 0;
};

} // namespace GPUPlayer