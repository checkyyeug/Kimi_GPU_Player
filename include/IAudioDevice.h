#pragma once

#include <string>
#include <vector>
#include <functional>

namespace GPUPlayer {

// 音频设备信息
struct AudioDeviceInfo {
    std::string id;
    std::string name;
    std::string api;  // WASAPI, CoreAudio, ALSA, etc.
    int max_channels;
    std::vector<int> supported_sample_rates;
    std::vector<int> supported_bit_depths;
    bool is_default;
    bool is_exclusive_mode_supported;
};

// 音频设备配置
struct AudioDeviceConfig {
    int sample_rate;
    int channels;
    int bit_depth;
    int buffer_size;  // in samples
    bool exclusive_mode;
    bool event_driven;  // for low latency
};

// 音频回调接口
class IAudioCallback {
public:
    virtual ~IAudioCallback() = default;
    
    // 音频处理回调
    virtual int ProcessAudio(float* output_buffer, int frames) = 0;
    
    // 错误回调
    virtual void OnError(const std::string& error_message) = 0;
    
    // 状态变化回调
    virtual void OnStateChanged(const std::string& new_state) = 0;
};

// 音频设备接口
class IAudioDevice {
public:
    virtual ~IAudioDevice() = default;
    
    // 设备枚举
    static std::vector<AudioDeviceInfo> EnumerateDevices();
    static AudioDeviceInfo GetDefaultDevice();
    
    // 初始化和配置
    virtual bool Initialize(const AudioDeviceConfig& config) = 0;
    virtual void Shutdown() = 0;
    virtual bool IsInitialized() const = 0;
    
    // 设备信息
    virtual AudioDeviceInfo GetDeviceInfo() const = 0;
    virtual AudioDeviceConfig GetConfig() const = 0;
    
    // 播放控制
    virtual bool Start(IAudioCallback* callback) = 0;
    virtual bool Stop() = 0;
    virtual bool IsPlaying() const = 0;
    
    // 音量控制
    virtual bool SetVolume(float volume) = 0;
    virtual float GetVolume() const = 0;
    
    // 延迟信息
    virtual double GetLatency() const = 0;
    virtual int GetBufferSize() const = 0;
    
    // 性能统计
    virtual int GetUnderrunCount() const = 0;
    virtual double GetCPUUsage() const = 0;
    
    // 错误信息
    virtual std::string GetLastError() const = 0;
};

// 音频设备工厂
class AudioDeviceFactory {
public:
    enum class API {
        AUTO,       // 自动选择最佳API
        WASAPI,     // Windows Audio Session API
        ASIO,       // Audio Stream Input/Output (Windows)
        CORE_AUDIO, // Core Audio (macOS)
        ALSA,       // Advanced Linux Sound Architecture
        JACK,       // JACK Audio Connection Kit
        PULSE_AUDIO // PulseAudio (Linux)
    };
    
    static std::unique_ptr<IAudioDevice> CreateDevice(API api = API::AUTO);
    static std::unique_ptr<IAudioDevice> CreateDevice(const std::string& device_id, API api = API::AUTO);
    static std::vector<API> GetAvailableAPIs();
    static std::string GetAPIName(API api);
};

// 平台特定的音频设备实现
#ifdef _WIN32
    #include "platform/windows/WasapiDevice.h"
    #include "platform/windows/AsioDevice.h"
#elif __APPLE__
    #include "platform/macos/CoreAudioDevice.h"
#elif __linux__
    #include "platform/linux/AlsaDevice.h"
    #include "platform/linux/JackDevice.h"
#endif

} // namespace GPUPlayer