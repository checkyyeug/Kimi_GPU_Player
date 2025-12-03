#pragma once

#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace GPUPlayer {

// 前向声明
class IAudioDecoder;
class IGPUProcessor;
class IAudioDevice;
class BufferManager;

// 音频格式枚举
enum class AudioFormat {
    PCM_16,
    PCM_24,
    PCM_32,
    PCM_32F,
    DSD_64,
    DSD_128,
    DSD_256
};

// 音频参数结构
struct AudioParams {
    int sample_rate;
    int channels;
    AudioFormat format;
    int bit_depth;
    
    AudioParams() : sample_rate(44100), channels(2), format(AudioFormat::PCM_16), bit_depth(16) {}
};

// 处理参数结构
struct ProcessingParams {
    // EQ参数
    struct EQParams {
        float freq1;      // 第一段频率
        float gain1;      // 第一段增益
        float q1;         // 第一段Q值
        float freq2;      // 第二段频率
        float gain2;      // 第二段增益
        float q2;         // 第二段Q值
        
        EQParams() : freq1(100.0f), gain1(0.0f), q1(0.7f), 
                     freq2(10000.0f), gain2(0.0f), q2(0.7f) {}
    } eq_params;
    
    // 滤波器参数
    struct FilterParams {
        enum Type { LOWPASS, HIGHPASS, BANDPASS, BANDSTOP };
        Type type;
        float freq;
        float q;
        int order;
        bool enabled;
        
        FilterParams() : type(LOWPASS), freq(1000.0f), q(0.7f), order(4), enabled(false) {}
    } filter_params;
    
    // 重采样参数
    struct ResampleParams {
        int target_sample_rate;
        bool enabled;
        
        ResampleParams() : target_sample_rate(44100), enabled(false) {}
    } resample_params;
    
    // 输出参数
    struct OutputParams {
        enum OutputFormat { PCM, DSD, DOP };
        OutputFormat format;
        int dsd_rate;  // DSD64, DSD128, DSD256
        
        OutputParams() : format(PCM), dsd_rate(64) {}
    } output_params;
};

// 音频引擎主类
class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();
    
    // 初始化和清理
    bool Initialize();
    void Shutdown();
    
    // 文件操作
    bool LoadFile(const std::string& filepath);
    void CloseFile();
    
    // 播放控制
    bool Play();
    bool Pause();
    bool Stop();
    bool Seek(double position_seconds);
    
    // 状态查询
    bool IsPlaying() const;
    bool IsPaused() const;
    double GetPosition() const;
    double GetDuration() const;
    
    // 参数设置
    void SetProcessingParams(const ProcessingParams& params);
    ProcessingParams GetProcessingParams() const;
    
    // 音频参数
    AudioParams GetInputAudioParams() const;
    AudioParams GetOutputAudioParams() const;
    
    // 回调函数
    using ErrorCallback = std::function<void(const std::string&)>;
    void SetErrorCallback(ErrorCallback callback);
    
    // 性能统计
    struct PerformanceStats {
        double cpu_usage;
        double gpu_usage;
        double latency_ms;
        size_t memory_usage;
        double dropout_rate;
    };
    PerformanceStats GetPerformanceStats() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace GPUPlayer