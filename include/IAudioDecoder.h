#pragma once

#include <string>
#include <vector>
#include <memory>

namespace GPUPlayer {

// 音频格式信息
struct AudioFormatInfo {
    int sample_rate;
    int channels;
    int bit_depth;
    int64_t total_samples;
    double duration_seconds;
    std::string codec_name;
    std::string format_name;
    bool is_lossless;
    bool is_dsd;
    int dsd_rate;  // 64, 128, 256 (for DSD)
};

// 解码器接口
class IAudioDecoder {
public:
    virtual ~IAudioDecoder() = default;
    
    // 文件操作
    virtual bool Open(const std::string& filepath) = 0;
    virtual void Close() = 0;
    virtual bool IsOpen() const = 0;
    
    // 格式信息
    virtual AudioFormatInfo GetFormatInfo() const = 0;
    
    // 解码操作
    virtual int Decode(float* output_buffer, int max_samples) = 0;
    virtual bool Seek(int64_t sample_position) = 0;
    virtual int64_t GetPosition() const = 0;
    
    // 元数据
    virtual std::string GetMetadata(const std::string& key) const = 0;
    virtual std::vector<std::string> GetAvailableMetadataKeys() const = 0;
    
    // 错误信息
    virtual std::string GetLastError() const = 0;
    
    // 支持的格式
    static std::vector<std::string> GetSupportedFormats();
    static bool IsFormatSupported(const std::string& filepath);
};

// 解码器工厂
class DecoderFactory {
public:
    static std::unique_ptr<IAudioDecoder> CreateDecoder(const std::string& filepath);
    static std::unique_ptr<IAudioDecoder> CreateDecoderByFormat(const std::string& format);
    static std::vector<std::string> GetSupportedFormats();
};

// 支持的音频格式
enum class AudioFileFormat {
    // 有损格式
    MP3,
    AAC,
    OGG_VORBIS,
    OPUS,
    
    // 无损格式
    FLAC,
    ALAC,
    WAV,
    AIFF,
    APE,
    WavPack,
    
    // DSD格式
    DSF,
    DFF,
    
    // 未知格式
    UNKNOWN
};

AudioFileFormat GetAudioFileFormat(const std::string& filepath);
std::string AudioFormatToString(AudioFileFormat format);

} // namespace GPUPlayer