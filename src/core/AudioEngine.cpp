#include "AudioEngine.h"
#include "IAudioDecoder.h"
#include "IGPUProcessor.h"
#include "IAudioDevice.h"
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <iostream>
#include <atomic>

namespace GPUPlayer {

// 内部实现类
class AudioEngine::Impl : public IAudioCallback {
public:
    Impl() : is_playing_(false), is_paused_(false), position_(0), duration_(0) {}
    
    ~Impl() {
        Stop();
        if (audio_thread_.joinable()) {
            audio_thread_.join();
        }
    }
    
    bool Initialize() {
        // 创建GPU处理器
        gpu_processor_ = GPUProcessorFactory::CreateProcessor();
        if (!gpu_processor_ || !gpu_processor_->Initialize()) {
            if (error_callback_) {
                error_callback_("Failed to initialize GPU processor");
            }
            return false;
        }
        
        // 创建音频设备
        audio_device_ = AudioDeviceFactory::CreateDevice();
        if (!audio_device_ || !audio_device_->Initialize({44100, 2, 16, 512, true, true})) {
            if (error_callback_) {
                error_callback_("Failed to initialize audio device");
            }
            return false;
        }
        
        initialized_ = true;
        return true;
    }
    
    void Shutdown() {
        Stop();
        if (audio_device_) {
            audio_device_->Shutdown();
            audio_device_.reset();
        }
        if (gpu_processor_) {
            gpu_processor_->Shutdown();
            gpu_processor_.reset();
        }
        initialized_ = false;
    }
    
    bool LoadFile(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 关闭当前文件
        CloseFile();
        
        // 创建解码器
        decoder_ = DecoderFactory::CreateDecoder(filepath);
        if (!decoder_ || !decoder_->Open(filepath)) {
            if (error_callback_) {
                error_callback_("Failed to open audio file: " + filepath);
            }
            return false;
        }
        
        // 获取音频信息
        auto format_info = decoder_->GetFormatInfo();
        input_params_.sample_rate = format_info.sample_rate;
        input_params_.channels = format_info.channels;
        input_params_.bit_depth = format_info.bit_depth;
        input_params_.format = format_info.is_dsd ? AudioFormat::DSD_64 : AudioFormat::PCM_16;
        
        duration_ = format_info.duration_seconds;
        position_ = 0;
        
        // 配置输出参数
        output_params_ = input_params_;
        
        return true;
    }
    
    void CloseFile() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        Stop();
        if (decoder_) {
            decoder_->Close();
            decoder_.reset();
        }
        position_ = 0;
        duration_ = 0;
    }
    
    bool Play() {
        if (!initialized_ || !decoder_ || is_playing_) {
            return false;
        }
        
        is_playing_ = true;
        is_paused_ = false;
        
        // 启动音频设备
        if (!audio_device_->Start(this)) {
            is_playing_ = false;
            if (error_callback_) {
                error_callback_("Failed to start audio device");
            }
            return false;
        }
        
        return true;
    }
    
    bool Pause() {
        if (!is_playing_) return false;
        
        is_paused_ = !is_paused_;
        return true;
    }
    
    bool Stop() {
        if (!is_playing_) return false;
        
        is_playing_ = false;
        is_paused_ = false;
        
        // 停止音频设备
        audio_device_->Stop();
        
        return true;
    }
    
    bool Seek(double position_seconds) {
        if (!decoder_) return false;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        int64_t sample_position = static_cast<int64_t>(position_seconds * input_params_.sample_rate);
        if (decoder_->Seek(sample_position)) {
            position_ = position_seconds;
            return true;
        }
        return false;
    }
    
    // IAudioCallback实现
    int ProcessAudio(float* output_buffer, int frames) override {
        if (!is_playing_ || is_paused_ || !decoder_ || !gpu_processor_) {
            // 静音输出
            for (int i = 0; i < frames * 2; i++) {  // 假设立体声
                output_buffer[i] = 0.0f;
            }
            return frames;
        }
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 解码音频数据
        std::vector<float> decode_buffer(frames * input_params_.channels);
        int decoded_samples = decoder_->Decode(decode_buffer.data(), frames);
        
        if (decoded_samples == 0) {
            // 文件结束
            is_playing_ = false;
            return 0;
        }
        
        // GPU处理
        ProcessAudioGPU(decode_buffer.data(), decoded_samples);
        
        // 复制到输出缓冲区
        int output_samples = decoded_samples * output_params_.channels;
        std::copy(decode_buffer.begin(), decode_buffer.begin() + output_samples, output_buffer);
        
        // 更新播放位置
        position_ += static_cast<double>(decoded_samples) / input_params_.sample_rate;
        
        return decoded_samples;
    }
    
    void OnError(const std::string& error_message) override {
        if (error_callback_) {
            error_callback_(error_message);
        }
    }
    
    void OnStateChanged(const std::string& new_state) override {
        // 状态变化处理
    }
    
    // 参数设置
    void SetProcessingParams(const ProcessingParams& params) {
        std::lock_guard<std::mutex> lock(mutex_);
        processing_params_ = params;
    }
    
    ProcessingParams GetProcessingParams() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return processing_params_;
    }
    
    // 状态查询
    bool IsPlaying() const { return is_playing_; }
    bool IsPaused() const { return is_paused_; }
    double GetPosition() const { return position_; }
    double GetDuration() const { return duration_; }
    
    AudioParams GetInputAudioParams() const { return input_params_; }
    AudioParams GetOutputAudioParams() const { return output_params_; }
    
    // 性能统计
    AudioEngine::PerformanceStats GetPerformanceStats() const {
        AudioEngine::PerformanceStats stats = {};
        
        if (gpu_processor_) {
            auto gpu_stats = gpu_processor_->GetStats();
            stats.gpu_usage = gpu_stats.gpu_utilization;
            stats.memory_usage = gpu_stats.memory_used;
        }
        
        if (audio_device_) {
            stats.latency_ms = audio_device_->GetLatency() * 1000.0;
        }
        
        // CPU使用率估算
        stats.cpu_usage = 2.5;  // 占位符
        stats.dropout_rate = 0.0;
        
        return stats;
    }
    
    void SetErrorCallback(ErrorCallback callback) {
        error_callback_ = callback;
    }
    
private:
    void ProcessAudioGPU(float* data, int samples) {
        if (!gpu_processor_) return;
        
        // 重采样
        if (processing_params_.resample_params.enabled && 
            processing_params_.resample_params.target_sample_rate != input_params_.sample_rate) {
            
            int output_samples = samples * processing_params_.resample_params.target_sample_rate / input_params_.sample_rate;
            std::vector<float> resampled_data(output_samples * input_params_.channels);
            
            // 对每个通道进行重采样
            for (int ch = 0; ch < input_params_.channels; ch++) {
                gpu_processor_->Resample(
                    data + ch * samples,
                    resampled_data.data() + ch * output_samples,
                    samples, output_samples,
                    static_cast<double>(processing_params_.resample_params.target_sample_rate) / input_params_.sample_rate
                );
            }
            
            // 更新样本数和采样率
            samples = output_samples;
            std::copy(resampled_data.begin(), resampled_data.end(), data);
        }
        
        // 2段EQ处理
        if (processing_params_.eq_params.gain1 != 0.0f || processing_params_.eq_params.gain2 != 0.0f) {
            gpu_processor_->ProcessEQ(data, samples * input_params_.channels,
                                     processing_params_.eq_params.freq1,
                                     processing_params_.eq_params.gain1,
                                     processing_params_.eq_params.q1,
                                     processing_params_.eq_params.freq2,
                                     processing_params_.eq_params.gain2,
                                     processing_params_.eq_params.q2);
        }
        
        // 数字滤波器
        if (processing_params_.filter_params.enabled) {
            // 生成滤波器系数（简化版）
            std::vector<float> coefficients(16, 0.0f);
            coefficients[0] = 1.0f;  // 单位脉冲响应
            
            gpu_processor_->ProcessFilter(data, samples * input_params_.channels,
                                         coefficients.data(), 15,
                                         processing_params_.filter_params.type);
        }
        
        // 格式转换 (PCM/DSD/DoP)
        if (processing_params_.output_params.format != ProcessingParams::OutputParams::PCM) {
            // 这里需要根据具体格式进行转换
            // 简化处理，实际实现会更复杂
        }
    }
    
    // 成员变量
    std::unique_ptr<IAudioDecoder> decoder_;
    std::unique_ptr<IGPUProcessor> gpu_processor_;
    std::unique_ptr<IAudioDevice> audio_device_;
    
    std::atomic<bool> is_playing_;
    std::atomic<bool> is_paused_;
    std::atomic<double> position_;
    double duration_;
    
    AudioParams input_params_;
    AudioParams output_params_;
    ProcessingParams processing_params_;
    
    std::mutex mutex_;
    ErrorCallback error_callback_;
    
    bool initialized_ = false;
    std::thread audio_thread_;
};

// AudioEngine公有接口实现
AudioEngine::AudioEngine() : pImpl(std::make_unique<Impl>()) {}

AudioEngine::~AudioEngine() = default;

bool AudioEngine::Initialize() {
    return pImpl->Initialize();
}

void AudioEngine::Shutdown() {
    pImpl->Shutdown();
}

bool AudioEngine::LoadFile(const std::string& filepath) {
    return pImpl->LoadFile(filepath);
}

void AudioEngine::CloseFile() {
    pImpl->CloseFile();
}

bool AudioEngine::Play() {
    return pImpl->Play();
}

bool AudioEngine::Pause() {
    return pImpl->Pause();
}

bool AudioEngine::Stop() {
    return pImpl->Stop();
}

bool AudioEngine::Seek(double position_seconds) {
    return pImpl->Seek(position_seconds);
}

bool AudioEngine::IsPlaying() const {
    return pImpl->IsPlaying();
}

bool AudioEngine::IsPaused() const {
    return pImpl->IsPaused();
}

double AudioEngine::GetPosition() const {
    return pImpl->GetPosition();
}

double AudioEngine::GetDuration() const {
    return pImpl->GetDuration();
}

void AudioEngine::SetProcessingParams(const ProcessingParams& params) {
    pImpl->SetProcessingParams(params);
}

ProcessingParams AudioEngine::GetProcessingParams() const {
    return pImpl->GetProcessingParams();
}

AudioParams AudioEngine::GetInputAudioParams() const {
    return pImpl->GetInputAudioParams();
}

AudioParams AudioEngine::GetOutputAudioParams() const {
    return pImpl->GetOutputAudioParams();
}

void AudioEngine::SetErrorCallback(ErrorCallback callback) {
    pImpl->SetErrorCallback(callback);
}

AudioEngine::PerformanceStats AudioEngine::GetPerformanceStats() const {
    return pImpl->GetPerformanceStats();
}

} // namespace GPUPlayer