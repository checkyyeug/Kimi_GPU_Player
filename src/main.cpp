#include "AudioEngine.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>

#ifdef _WIN32
#include <windows.h>
#elif __linux__
#include <unistd.h>
#elif __APPLE__
#include <unistd.h>
#endif

using namespace GPUPlayer;

// 简单的控制台界面
class ConsoleUI {
public:
    ConsoleUI(AudioEngine& engine) : engine_(engine), running_(true) {}
    
    void Run() {
        std::cout << "==========================================\n";
        std::cout << "     GPU 音乐播放器 v1.0.0\n";
        std::cout << "==========================================\n";
        std::cout << "命令: play <文件> | pause | stop | seek <秒> | eq <freq1> <gain1> <freq2> <gain2> | stats | quit\n";
        std::cout << "==========================================\n";
        
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
                std::cout << "错误: 请指定音频文件路径\n";
                return true;
            }
            
            std::string filepath = tokens[1];
            // 处理带空格的文件路径
            for (size_t i = 2; i < tokens.size(); i++) {
                filepath += " " + tokens[i];
            }
            
            if (engine_.LoadFile(filepath)) {
                if (engine_.Play()) {
                    std::cout << "正在播放: " << filepath << "\n";
                } else {
                    std::cout << "错误: 无法开始播放\n";
                }
            } else {
                std::cout << "错误: 无法加载音频文件\n";
            }
            
        } else if (cmd == "pause") {
            if (engine_.Pause()) {
                std::cout << "播放已暂停/继续\n";
            } else {
                std::cout << "错误: 无法暂停播放\n";
            }
            
        } else if (cmd == "stop") {
            if (engine_.Stop()) {
                std::cout << "播放已停止\n";
            } else {
                std::cout << "错误: 无法停止播放\n";
            }
            
        } else if (cmd == "seek") {
            if (tokens.size() < 2) {
                std::cout << "错误: 请指定时间位置(秒)\n";
                return true;
            }
            
            try {
                double position = std::stod(tokens[1]);
                if (engine_.Seek(position)) {
                    std::cout << "已跳转到: " << position << " 秒\n";
                } else {
                    std::cout << "错误: 无法跳转到指定位置\n";
                }
            } catch (const std::exception& e) {
                std::cout << "错误: 无效的时间格式\n";
            }
            
        } else if (cmd == "eq") {
            if (tokens.size() < 5) {
                std::cout << "错误: 参数格式: eq <freq1> <gain1> <freq2> <gain2>\n";
                return true;
            }
            
            try {
                auto params = engine_.GetProcessingParams();
                params.eq_params.freq1 = std::stof(tokens[1]);
                params.eq_params.gain1 = std::stof(tokens[2]);
                params.eq_params.freq2 = std::stof(tokens[3]);
                params.eq_params.gain2 = std::stof(tokens[4]);
                
                engine_.SetProcessingParams(params);
                std::cout << "EQ设置已更新:\n";
                std::cout << "  频段1: " << params.eq_params.freq1 << "Hz, " 
                         << params.eq_params.gain1 << "dB\n";
                std::cout << "  频段2: " << params.eq_params.freq2 << "Hz, " 
                         << params.eq_params.gain2 << "dB\n";
                         
            } catch (const std::exception& e) {
                std::cout << "错误: 无效的参数格式\n";
            }
            
        } else if (cmd == "stats") {
            ShowStats();
            
        } else if (cmd == "quit" || cmd == "exit") {
            std::cout << "正在退出...\n";
            running_ = false;
            return false;
            
        } else {
            std::cout << "未知命令: " << cmd << "\n";
            std::cout << "可用命令: play, pause, stop, seek, eq, stats, quit\n";
        }
        
        return true;
    }
    
    void ShowStats() {
        auto stats = engine_.GetPerformanceStats();
        auto input_params = engine_.GetInputAudioParams();
        auto output_params = engine_.GetOutputAudioParams();
        
        std::cout << "\n===== 性能统计 =====\n";
        std::cout << "CPU使用率: " << std::fixed << std::setprecision(1) 
                 << stats.cpu_usage << "%\n";
        std::cout << "GPU使用率: " << stats.gpu_usage << "%\n";
        std::cout << "内存使用: " << (stats.memory_usage / 1024 / 1024) << " MB\n";
        std::cout << "延迟: " << std::setprecision(2) << stats.latency_ms << " ms\n";
        std::cout << "丢帧率: " << std::setprecision(3) << stats.dropout_rate << "%\n";
        
        std::cout << "\n===== 音频信息 =====\n";
        std::cout << "输入格式: " << input_params.sample_rate << "Hz, "
                 << input_params.channels << "ch, "
                 << input_params.bit_depth << "-bit\n";
        std::cout << "输出格式: " << output_params.sample_rate << "Hz, "
                 << output_params.channels << "ch, "
                 << output_params.bit_depth << "-bit\n";
        
        if (engine_.IsPlaying()) {
            std::cout << "播放状态: ";
            if (engine_.IsPaused()) {
                std::cout << "暂停\n";
            } else {
                std::cout << "播放中\n";
            }
            std::cout << "位置: " << std::setprecision(1) << engine_.GetPosition() 
                     << " / " << engine_.GetDuration() << " 秒\n";
        } else {
            std::cout << "播放状态: 停止\n";
        }
        std::cout << "===================\n\n";
    }
    
    AudioEngine& engine_;
    bool running_;
};

// 错误处理回调
void ErrorHandler(const std::string& error) {
    std::cerr << "错误: " << error << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "GPU音乐播放器启动中...\n";
    
    // 创建音频引擎
    AudioEngine engine;
    
    // 设置错误回调
    engine.SetErrorCallback(ErrorHandler);
    
    // 初始化引擎
    if (!engine.Initialize()) {
        std::cerr << "无法初始化音频引擎\n";
        return 1;
    }
    
    std::cout << "音频引擎初始化成功\n";
    
    // 显示GPU信息
    auto processor = GPUProcessorFactory::CreateProcessor();
    if (processor && processor->IsGPUSupported()) {
        std::cout << "GPU: " << processor->GetGPUName() << "\n";
        std::cout << "GPU内存: " << (processor->GetGPUMemory() / 1024 / 1024) << " MB\n";
    } else {
        std::cout << "警告: 未检测到支持的GPU，将使用CPU处理\n";
    }
    
    // 设置默认处理参数
    ProcessingParams default_params;
    default_params.eq_params.freq1 = 100.0f;    // 低频
    default_params.eq_params.gain1 = 0.0f;      // 低频增益
    default_params.eq_params.q1 = 0.7f;         // 低频Q值
    default_params.eq_params.freq2 = 10000.0f;  // 高频
    default_params.eq_params.gain2 = 0.0f;      // 高频增益
    default_params.eq_params.q2 = 0.7f;         // 高频Q值
    
    default_params.filter_params.enabled = false;
    default_params.resample_params.enabled = false;
    default_params.output_params.format = ProcessingParams::OutputParams::PCM;
    
    engine.SetProcessingParams(default_params);
    
    // 如果有命令行参数，直接播放文件
    if (argc > 1) {
        std::string filepath = argv[1];
        std::cout << "正在加载文件: " << filepath << "\n";
        
        if (engine.LoadFile(filepath)) {
            if (engine.Play()) {
                std::cout << "开始播放，使用控制台命令控制播放...\n";
            } else {
                std::cerr << "无法开始播放\n";
            }
        } else {
            std::cerr << "无法加载音频文件\n";
        }
    }
    
    // 运行控制台界面
    ConsoleUI ui(engine);
    ui.Run();
    
    // 清理
    engine.Stop();
    engine.CloseFile();
    engine.Shutdown();
    
    std::cout << "GPU音乐播放器已退出\n";
    return 0;
}