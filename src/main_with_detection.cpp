#include "AudioEngine.h"
#include "gpu/VulkanDetector.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

using namespace GPUPlayer;

// 增强版本的音频引擎，带GPU检测
class EnhancedAudioEngine {
public:
    EnhancedAudioEngine() : is_playing_(false), is_paused_(false), position_(0.0), duration_(0.0) {}
    
    bool Initialize() {
        std::cout << "[AUDIO] 初始化增强音频引擎..." << std::endl;
        
        // GPU检测
        std::cout << "[GPU] 检测可用GPU后端..." << std::endl;
        DetectGPUBackends();
        
        // 模拟初始化过程
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        std::cout << "[AUDIO] 音频引擎初始化成功" << std::endl;
        return true;
    }
    
    bool LoadFile(const std::string& filepath) {
        std::cout << "[AUDIO] 加载音频文件: " << filepath << std::endl;
        
        // 模拟文件加载
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        
        // 设置模拟的音频参数
        duration_ = 180.0; // 3分钟
        position_ = 0.0;
        
        std::cout << "[AUDIO] 文件加载成功 - 时长: " << duration_ << " 秒" << std::endl;
        return true;
    }
    
    bool Play() {
        if (is_playing_) return false;
        
        is_playing_ = true;
        is_paused_ = false;
        
        std::cout << "[AUDIO] 开始播放" << std::endl;
        
        // 启动播放线程
        play_thread_ = std::thread([this]() {
            this->PlaybackThread();
        });
        
        return true;
    }
    
    bool Pause() {
        if (!is_playing_) return false;
        
        is_paused_ = !is_paused_;
        std::cout << "[AUDIO] 播放 " << (is_paused_ ? "暂停" : "继续") << std::endl;
        return true;
    }
    
    bool Stop() {
        if (!is_playing_) return false;
        
        is_playing_ = false;
        is_paused_ = false;
        
        if (play_thread_.joinable()) {
            play_thread_.join();
        }
        
        std::cout << "[AUDIO] 播放停止" << std::endl;
        return true;
    }
    
    bool Seek(double position) {
        if (position < 0.0 || position > duration_) return false;
        
        position_ = position;
        std::cout << "[AUDIO] 跳转到: " << position << " 秒" << std::endl;
        return true;
    }
    
    bool IsPlaying() const { return is_playing_; }
    bool IsPaused() const { return is_paused_; }
    double GetPosition() const { return position_; }
    double GetDuration() const { return duration_; }
    
    void ShowStats() {
        std::cout << "\n===== 播放统计 =====" << std::endl;
        std::cout << "播放状态: " << (is_playing_ ? (is_paused_ ? "暂停" : "播放中") : "停止") << std::endl;
        std::cout << "播放位置: " << position_ << " / " << duration_ << " 秒" << std::endl;
        std::cout << "进度: " << std::fixed << std::setprecision(1) 
                  << (duration_ > 0 ? (position_ / duration_ * 100.0) : 0.0) << "%" << std::endl;
        
        // 模拟一些性能数据
        std::cout << "模拟CPU使用率: " << (is_playing_ ? "2.3%" : "0.1%") << std::endl;
        std::cout << "模拟内存使用: " << (is_playing_ ? "45.2 MB" : "12.8 MB") << std::endl;
        
        // 显示GPU信息
        ShowGPUStats();
        
        std::cout << "===================" << std::endl;
    }
    
    void ShowGPUStats() {
        std::cout << "\n--- GPU信息 ---" << std::endl;
        
        // 获取可用后端
        auto backends = GPUProcessorFactory::GetAvailableBackends();
        if (!backends.empty()) {
            std::cout << "可用GPU后端: ";
            for (size_t i = 0; i < backends.size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << GPUProcessorFactory::GetBackendName(backends[i]);
            }
            std::cout << std::endl;
            
            // 显示第一个可用后端的详细信息
            auto processor = GPUProcessorFactory::CreateProcessor(backends[0]);
            if (processor && processor->Initialize()) {
                std::cout << "当前GPU: " << processor->GetGPUName() << std::endl;
                std::cout << "GPU内存: " << (processor->GetGPUMemory() / 1024 / 1024) << " MB" << std::endl;
                
                auto stats = processor->GetStats();
                std::cout << "GPU利用率: " << std::fixed << std::setprecision(1) << stats.gpu_utilization << "%" << std::endl;
                
                processor->Shutdown();
            }
        } else {
            std::cout << "⚠️ 未检测到GPU加速支持" << std::endl;
        }
    }
    
    void DetectGPUBackends() {
        std::cout << "\n🔍 GPU后端检测:" << std::endl;
        
        // 使用通用GPU检测器
        GPUDetector::PrintGPUReport();
        
        // 详细检测每个后端
        auto backends = GPUProcessorFactory::GetAvailableBackends();
        
        if (!backends.empty()) {
            std::cout << "\n🔧 详细测试:" << std::endl;
            
            for (auto backend : backends) {
                std::cout << "测试 " << GPUProcessorFactory::GetBackendName(backend) << "..." << std::endl;
                
                auto processor = GPUProcessorFactory::CreateProcessor(backend);
                if (processor && processor->Initialize()) {
                    // 简单功能测试
                    const int test_samples = 4410; // 0.1秒
                    std::vector<float> test_input(test_samples, 0.5f);
                    std::vector<float> test_output(test_samples);
                    
                    bool test_passed = processor->Resample(test_input.data(), test_output.data(),
                                                          test_samples, test_samples, 1.0);
                    
                    if (test_passed) {
                        std::cout << "  ✅ " << GPUProcessorFactory::GetBackendName(backend) << " 测试通过" << std::endl;
                    } else {
                        std::cout << "  ❌ " << GPUProcessorFactory::GetBackendName(backend) << " 测试失败" << std::endl;
                    }
                    
                    processor->Shutdown();
                }
            }
        }
    }
    
    ~EnhancedAudioEngine() {
        Stop();
    }

private:
    void PlaybackThread() {
        const double update_interval = 0.1; // 100ms更新一次
        
        while (is_playing_ && position_ < duration_) {
            if (!is_paused_) {
                position_ += update_interval;
                
                // 模拟音频处理
                SimulateAudioProcessing();
            }
            
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(update_interval * 1000))
            );
        }
        
        if (position_ >= duration_) {
            std::cout << "[AUDIO] 播放完成" << std::endl;
            is_playing_ = false;
        }
    }
    
    void SimulateAudioProcessing() {
        // 模拟简单的音频数据生成
        const int samples_per_update = 4410; // 44.1kHz * 0.1s
        
        // 生成模拟音频数据（正弦波）
        for (int i = 0; i < samples_per_update; ++i) {
            double time = (position_ * 44100.0 + i) / 44100.0;
            double sample = std::sin(2.0 * M_PI * 440.0 * time) * 0.5; // 440Hz正弦波
            
            // 这里可以添加音频处理算法
            // 例如：音量调节、简单滤波等
            
            (void)sample; // 避免未使用变量警告
        }
    }
    
    bool is_playing_;
    bool is_paused_;
    double position_;
    double duration_;
    std::thread play_thread_;
};

// 控制台界面
class EnhancedConsoleUI {
public:
    EnhancedConsoleUI(EnhancedAudioEngine& engine) : engine_(engine), running_(true) {}
    
    void Run() {
        std::cout << "==========================================" << std::endl;
        std::cout << "     GPU 音乐播放器 (增强版) v1.0.0" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "命令: play <文件> | pause | stop | seek <秒> | stats | gpu | quit" << std::endl;
        std::cout << "增强: gpu 命令显示详细的GPU信息" << std::endl;
        std::cout << "==========================================" << std::endl;
        
        while (running_) {
            std::cout << "> ";
            std::string command;
            std::getline(std::cin, command);
            
            if (!ProcessCommand(command)) {
                break;
            }
        }
    }
    
private:
    bool ProcessCommand(const std::string& command) {
        std::vector<std::string> tokens;
        std::istringstream iss(command);
        std::string token;
        
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        if (tokens.empty()) return true;
        
        const std::string& cmd = tokens[0];
        
        if (cmd == "play") {
            if (tokens.size() < 2) {
                std::cout << "错误: 请指定音频文件路径" << std::endl;
                return true;
            }
            
            std::string filepath = tokens[1];
            for (size_t i = 2; i < tokens.size(); i++) {
                filepath += " " + tokens[i];
            }
            
            if (engine_.LoadFile(filepath)) {
                if (engine_.Play()) {
                    std::cout << "正在播放: " << filepath << std::endl;
                } else {
                    std::cout << "错误: 无法开始播放" << std::endl;
                }
            } else {
                std::cout << "错误: 无法加载音频文件" << std::endl;
            }
            
        } else if (cmd == "pause") {
            if (engine_.Pause()) {
                std::cout << "播放已暂停/继续" << std::endl;
            } else {
                std::cout << "错误: 无法暂停播放" << std::endl;
            }
            
        } else if (cmd == "stop") {
            if (engine_.Stop()) {
                std::cout << "播放已停止" << std::endl;
            } else {
                std::cout << "错误: 无法停止播放" << std::endl;
            }
            
        } else if (cmd == "seek") {
            if (tokens.size() < 2) {
                std::cout << "错误: 请指定时间位置(秒)" << std::endl;
                return true;
            }
            
            try {
                double position = std::stod(tokens[1]);
                if (engine_.Seek(position)) {
                    std::cout << "已跳转到: " << position << " 秒" << std::endl;
                } else {
                    std::cout << "错误: 无法跳转到指定位置" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "错误: 无效的时间格式" << std::endl;
            }
            
        } else if (cmd == "stats") {
            engine_.ShowStats();
            
        } else if (cmd == "gpu") {
            std::cout << "\n🔍 详细GPU检测:" << std::endl;
            GPUDetector::PrintGPUReport();
            
        } else if (cmd == "quit" || cmd == "exit") {
            std::cout << "正在退出..." << std::endl;
            running_ = false;
            return false;
            
        } else {
            std::cout << "未知命令: " << cmd << std::endl;
            std::cout << "可用命令: play, pause, stop, seek, stats, gpu, quit" << std::endl;
        }
        
        return true;
    }
    
    EnhancedAudioEngine& engine_;
    bool running_;
};

int main(int argc, char* argv[]) {
    std::cout << "GPU音乐播放器 (增强版) 启动中..." << std::endl;
    std::cout << "包含GPU检测功能 🔍" << std::endl;
    
    // 创建音频引擎
    EnhancedAudioEngine engine;
    
    // 初始化引擎（包含GPU检测）
    if (!engine.Initialize()) {
        std::cerr << "无法初始化音频引擎" << std::endl;
        return 1;
    }
    
    std::cout << "音频引擎初始化成功" << std::endl;
    std::cout << "注意: 这是演示版本，包含GPU检测功能" << std::endl;
    
    // 如果有命令行参数，尝试加载文件
    if (argc > 1) {
        std::string filepath = argv[1];
        std::cout << "正在加载文件: " << filepath << std::endl;
        
        if (engine.LoadFile(filepath)) {
            if (engine.Play()) {
                std::cout << "开始播放，使用控制台命令控制播放..." << std::endl;
            } else {
                std::cerr << "无法开始播放" << std::endl;
            }
        } else {
            std::cerr << "无法加载音频文件" << std::endl;
        }
    }
    
    // 运行控制台界面
    EnhancedConsoleUI ui(engine);
    ui.Run();
    
    // 清理
    engine.Stop();
    
    std::cout << "GPU音乐播放器已退出" << std::endl;
    return 0;
}