#include "gpu/CudaProcessor.cuh"
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace GPUPlayer {

// CUDA错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << \
                     " - " << cudaGetErrorString(error) << std::endl; \
        return false; \
    } \
} while(0)

// CUFFT错误检查宏
#define CUFFT_CHECK(call) do { \
    cufftResult result = call; \
    if (result != CUFFT_SUCCESS) { \
        std::cerr << "CUFFT Error at " << __FILE__ << ":" << __LINE__ << \
                     " - " << result << std::endl; \
        return false; \
    } \
} while(0)

// GPU内核：重采样
__global__ void resample_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_size,
    int output_size,
    double ratio,
    const float* __restrict__ filter_kernel,
    int filter_size,
    double cutoff_freq
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;
    
    double src_index = tid * ratio;
    int src_idx = floor(src_index);
    double frac = src_index - src_idx;
    
    float result = 0.0f;
    int half_filter = filter_size / 2;
    
    // 使用Sinc插值 + Kaiser窗函数
    for (int i = -half_filter; i < half_filter; i++) {
        int tap_idx = src_idx + i;
        if (tap_idx >= 0 && tap_idx < input_size) {
            float tap = input[tap_idx];
            float filter_tap = filter_kernel[i + half_filter];
            
            // 应用分数延迟
            if (frac != 0.0 && i == 0) {
                // 线性插值相邻采样点
                float next_tap = (tap_idx + 1 < input_size) ? input[tap_idx + 1] : tap;
                tap = tap * (1.0f - frac) + next_tap * frac;
            }
            
            result += tap * filter_tap;
        }
    }
    
    output[tid] = result;
}

// GPU内核：2段参数均衡器
__global__ void eq2_kernel(
    float* __restrict__ data,
    int samples,
    float freq1, float gain1, float q1,
    float freq2, float gain2, float q2,
    int sample_rate
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= samples) return;
    
    // 简化的双二阶滤波器实现
    float w1 = 2.0f * M_PI * freq1 / sample_rate;
    float w2 = 2.0f * M_PI * freq2 / sample_rate;
    
    float cos_w1 = cosf(w1);
    float sin_w1 = sinf(w1);
    float alpha1 = sin_w1 / (2.0f * q1);
    
    float cos_w2 = cosf(w2);
    float sin_w2 = sinf(w2);
    float alpha2 = sin_w2 / (2.0f * q2);
    
    // 计算增益线性值
    float A1 = powf(10.0f, gain1 / 40.0f);
    float A2 = powf(10.0f, gain2 / 40.0f);
    
    // 这里简化处理，实际应该使用状态变量
    float sample = data[tid];
    
    // 应用第一段EQ
    if (gain1 != 0.0f) {
        float b0 = 1.0f + alpha1 * A1;
        float b1 = -2.0f * cos_w1;
        float b2 = 1.0f - alpha1 * A1;
        float a0 = 1.0f + alpha1 / A1;
        float a1 = -2.0f * cos_w1;
        float a2 = 1.0f - alpha1 / A1;
        
        // 简化的滤波器应用（实际需要状态变量）
        sample = (b0 * sample + b1 * sample + b2 * sample) / a0;
    }
    
    // 应用第二段EQ
    if (gain2 != 0.0f) {
        float b0 = 1.0f + alpha2 * A2;
        float b1 = -2.0f * cos_w2;
        float b2 = 1.0f - alpha2 * A2;
        float a0 = 1.0f + alpha2 / A2;
        float a1 = -2.0f * cos_w2;
        float a2 = 1.0f - alpha2 / A2;
        
        sample = (b0 * sample + b1 * sample + b2 * sample) / a0;
    }
    
    data[tid] = sample;
}

// GPU内核：FIR滤波器
__global__ void fir_filter_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int samples,
    const float* __restrict__ coefficients,
    int filter_order
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= samples) return;
    
    float result = 0.0f;
    
    // FIR滤波器卷积
    for (int i = 0; i <= filter_order; i++) {
        int tap_idx = tid - i;
        if (tap_idx >= 0) {
            result += input[tap_idx] * coefficients[i];
        }
    }
    
    output[tid] = result;
}

// GPU内核：PCM到DSD转换
__global__ void pcm_to_dsd_kernel(
    const float* __restrict__ pcm_data,
    unsigned char* __restrict__ dsd_data,
    int pcm_samples,
    int dsd_rate
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pcm_samples) return;
    
    float sample = pcm_data[tid];
    
    // 简化的噪声整形和DSD编码
    // 实际实现需要更复杂的Sigma-Delta调制器
    static __shared__ float noise_state[256];
    int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) {
        noise_state[threadIdx.x / 32] = 0.0f;
    }
    __syncthreads();
    
    float noise = noise_state[threadIdx.x / 32];
    float quantized = (sample + noise > 0.0f) ? 1.0f : -1.0f;
    float error = sample - quantized;
    noise_state[threadIdx.x / 32] = error * 0.9f;  // 简单的噪声反馈
    
    // 打包DSD数据
    int byte_idx = tid / 8;
    int bit_idx = tid % 8;
    
    if (quantized > 0.0f) {
        atomicOr(&dsd_data[byte_idx], 1 << (7 - bit_idx));
    }
}

// CUDA处理器实现
CudaProcessor::CudaProcessor() : d_filter_kernel(nullptr), d_filter_size(0) {}

CudaProcessor::~CudaProcessor() {
    Shutdown();
}

bool CudaProcessor::Initialize() {
    // 检查CUDA设备
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        last_error_ = "No CUDA devices found";
        return false;
    }
    
    // 选择最佳设备
    int best_device = 0;
    cudaDeviceProp best_props;
    CUDA_CHECK(cudaGetDeviceProperties(&best_props, best_device));
    
    device_name_ = std::string(best_props.name);
    memory_total_ = best_props.totalGlobalMem;
    
    // 设置设备
    CUDA_CHECK(cudaSetDevice(best_device));
    
    // 创建CUDA流
    CUDA_CHECK(cudaStreamCreate(&stream_));
    
    // 初始化滤波器内核
    InitFilterKernel();
    
    initialized_ = true;
    return true;
}

void CudaProcessor::Shutdown() {
    if (!initialized_) return;
    
    // 释放设备内存
    if (d_filter_kernel) {
        cudaFree(d_filter_kernel);
        d_filter_kernel = nullptr;
    }
    
    // 销毁CUDA流
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    
    // 重置设备
    cudaDeviceReset();
    
    initialized_ = false;
}

bool CudaProcessor::IsGPUSupported() const {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
}

std::string CudaProcessor::GetGPUName() const {
    return device_name_;
}

size_t CudaProcessor::GetGPUMemory() const {
    return memory_total_;
}

bool CudaProcessor::Resample(const float* input, float* output, 
                            int input_samples, int output_samples,
                            double src_ratio) {
    if (!initialized_) return false;
    
    // 分配设备内存
    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_samples * sizeof(float)));
    
    // 复制输入数据到设备
    CUDA_CHECK(cudaMemcpyAsync(d_input, input, input_samples * sizeof(float), 
                              cudaMemcpyHostToDevice, stream_));
    
    // 配置内核参数
    int block_size = 256;
    int grid_size = (output_samples + block_size - 1) / block_size;
    
    // 启动重采样内核
    resample_kernel<<<grid_size, block_size, 0, stream_>>>(
        d_input, d_output, input_samples, output_samples, src_ratio,
        d_filter_kernel, d_filter_size, 0.45
    );
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpyAsync(output, d_output, output_samples * sizeof(float), 
                              cudaMemcpyDeviceToHost, stream_));
    
    // 等待完成
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // 清理
    cudaFree(d_input);
    cudaFree(d_output);
    
    return true;
}

bool CudaProcessor::ProcessEQ(float* data, int samples,
                             float freq1, float gain1, float q1,
                             float freq2, float gain2, float q2) {
    if (!initialized_) return false;
    
    // 分配设备内存
    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, samples * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(d_data, data, samples * sizeof(float), 
                              cudaMemcpyHostToDevice, stream_));
    
    // 配置内核参数
    int block_size = 256;
    int grid_size = (samples + block_size - 1) / block_size;
    
    // 启动EQ内核
    eq2_kernel<<<grid_size, block_size, 0, stream_>>>(
        d_data, samples, freq1, gain1, q1, freq2, gain2, q2, 44100
    );
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpyAsync(data, d_data, samples * sizeof(float), 
                              cudaMemcpyDeviceToHost, stream_));
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    cudaFree(d_data);
    
    return true;
}

bool CudaProcessor::ProcessFilter(float* data, int samples,
                                 const float* coefficients, int filter_order,
                                 int filter_type) {
    if (!initialized_) return false;
    
    // 分配设备内存
    float *d_input = nullptr, *d_output = nullptr, *d_coeffs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_coeffs, (filter_order + 1) * sizeof(float)));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpyAsync(d_input, data, samples * sizeof(float), 
                              cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_coeffs, coefficients, (filter_order + 1) * sizeof(float), 
                              cudaMemcpyHostToDevice, stream_));
    
    // 配置内核参数
    int block_size = 256;
    int grid_size = (samples + block_size - 1) / block_size;
    
    // 启动滤波器内核
    fir_filter_kernel<<<grid_size, block_size, 0, stream_>>>(
        d_input, d_output, samples, d_coeffs, filter_order
    );
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpyAsync(data, d_output, samples * sizeof(float), 
                              cudaMemcpyDeviceToHost, stream_));
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // 清理
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_coeffs);
    
    return true;
}

bool CudaProcessor::ConvertPcmToDsd(const float* pcm_data, int pcm_samples,
                                   unsigned char* dsd_data, int dsd_rate) {
    if (!initialized_) return false;
    
    // 分配设备内存
    float *d_pcm = nullptr;
    unsigned char *d_dsd = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pcm, pcm_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dsd, (pcm_samples * dsd_rate / 64 + 7) / 8));
    
    // 复制PCM数据到设备
    CUDA_CHECK(cudaMemcpyAsync(d_pcm, pcm_data, pcm_samples * sizeof(float), 
                              cudaMemcpyHostToDevice, stream_));
    
    // 清零DSD缓冲区
    CUDA_CHECK(cudaMemsetAsync(d_dsd, 0, (pcm_samples * dsd_rate / 64 + 7) / 8, stream_));
    
    // 配置内核参数
    int block_size = 256;
    int grid_size = (pcm_samples + block_size - 1) / block_size;
    
    // 启动PCM到DSD转换内核
    pcm_to_dsd_kernel<<<grid_size, block_size, 0, stream_>>>(
        d_pcm, d_dsd, pcm_samples, dsd_rate
    );
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpyAsync(dsd_data, d_dsd, (pcm_samples * dsd_rate / 64 + 7) / 8, 
                              cudaMemcpyDeviceToHost, stream_));
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // 清理
    cudaFree(d_pcm);
    cudaFree(d_dsd);
    
    return true;
}

bool CudaProcessor::ConvertDsdToPcm(const unsigned char* dsd_data, int dsd_samples,
                                   float* pcm_data, int dsd_rate) {
    // DSD到PCM转换实现
    // 这里简化处理，实际需要更复杂的解码算法
    for (int i = 0; i < dsd_samples; i++) {
        pcm_data[i] = 0.0f;  // 占位符
    }
    return true;
}

bool CudaProcessor::EncodeDop(const unsigned char* dsd_data, int dsd_samples,
                             unsigned short* dop_data, int dop_samples) {
    // DoP编码实现
    for (int i = 0; i < dop_samples; i++) {
        dop_data[i] = 0x05A5;  // DoP标记
    }
    return true;
}

bool CudaProcessor::DecodeDop(const unsigned short* dop_data, int dop_samples,
                             unsigned char* dsd_data) {
    // DoP解码实现
    for (int i = 0; i < dop_samples; i++) {
        dsd_data[i] = 0;  // 占位符
    }
    return true;
}

bool CudaProcessor::ProcessBatch(std::vector<float*>& channels,
                                int samples_per_channel) {
    if (!initialized_) return false;
    
    // 批量处理多个通道
    for (auto* channel : channels) {
        if (!ProcessEQ(channel, samples_per_channel, 100.0f, 0.0f, 0.7f, 10000.0f, 0.0f, 0.7f)) {
            return false;
        }
    }
    
    return true;
}

IGPUProcessor::GPUStats CudaProcessor::GetStats() const {
    GPUStats stats = {};
    
    if (initialized_) {
        // 获取GPU利用率
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        
        stats.gpu_utilization = 50.0;  // 占位符
        stats.memory_used = 0;  // 需要查询实际使用量
        stats.memory_total = memory_total_;
        stats.processing_time_ms = 0.0;  // 需要计时
        stats.active_kernels = 0;
    }
    
    return stats;
}

std::string CudaProcessor::GetLastError() const {
    return last_error_;
}

void CudaProcessor::InitFilterKernel() {
    // 初始化抗混叠滤波器内核
    const int filter_size = 64;
    std::vector<float> host_kernel(filter_size);
    
    // 生成Kaiser窗Sinc滤波器
    const double beta = 5.0;
    const double cutoff = 0.45;
    
    for (int i = 0; i < filter_size; i++) {
        int n = i - filter_size / 2;
        
        if (n == 0) {
            host_kernel[i] = 2.0f * cutoff;
        } else {
            host_kernel[i] = sin(2.0 * M_PI * cutoff * n) / (M_PI * n);
        }
        
        // Kaiser窗
        double kaiser = 1.0;  // 简化版
        host_kernel[i] *= kaiser;
    }
    
    // 分配和复制到设备
    CUDA_CHECK(cudaMalloc(&d_filter_kernel, filter_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_filter_kernel, host_kernel.data(), 
                         filter_size * sizeof(float), cudaMemcpyHostToDevice));
    
    d_filter_size = filter_size;
}

} // namespace GPUPlayer