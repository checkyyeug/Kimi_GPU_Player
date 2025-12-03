# GPU音乐播放器 - 构建和运行指南

## 🎯 概述

本项目提供了两种构建方式：
1. **简化版本** - 无需外部依赖，用于演示基本功能
2. **完整版本** - 包含所有功能，需要完整的依赖库

## 🔧 简化版本构建 (推荐)

### 快速开始

```bash
# 1. 清理并构建
make clean
make

# 2. 运行演示
./gpu_player_simple

# 3. 或使用自动演示
./demo.sh
```

### 功能特点
- ✅ 无需外部依赖
- ✅ 跨平台支持 (Linux/macOS/Windows)
- ✅ 完整的控制台界面
- ✅ 模拟音频播放流程
- ⚠️ 无实际音频输出
- ⚠️ 无GPU加速

### 使用命令

启动程序后，可以使用以下命令：

```
play <文件路径>    # 加载并播放音频文件
pause              # 暂停/继续播放
stop               # 停止播放
seek <秒数>        # 跳转到指定位置
stats              # 显示播放统计
quit               # 退出程序
```

### 示例会话

```
> play music.mp3
[AUDIO] 加载音频文件: music.mp3
[AUDIO] 文件加载成功 - 时长: 180.0 秒
正在播放: music.mp3

> stats
===== 播放统计 =====
播放状态: 播放中
播放位置: 15.2 / 180.0 秒
进度: 8.4%
模拟CPU使用率: 2.3%
模拟内存使用: 45.2 MB
===================

> pause
[AUDIO] 播放已暂停/继续

> quit
正在退出...
```

## 🚀 完整版本构建

### 系统依赖

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libasound2-dev libjack-dev
sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswresample-dev
sudo apt install -y qt6-base-dev qt6-charts-dev
sudo apt install -y nvidia-cuda-toolkit  # 如果使用NVIDIA GPU
```

#### CentOS/RHEL/Fedora
```bash
sudo yum install -y gcc gcc-c++ cmake pkgconfig
sudo yum install -y alsa-lib-devel jack-audio-connection-kit-devel
sudo yum install -y ffmpeg-devel
sudo yum install -y cuda-toolkit  # 如果使用NVIDIA GPU
```

#### macOS
```bash
brew install cmake pkg-config
brew install ffmpeg
brew install qt@6
brew install jack
```

#### Windows
1. 安装 Visual Studio 2019+ 和 CMake
2. 安装 vcpkg 包管理器
3. 使用 vcpkg 安装依赖：
```powershell
vcpkg install ffmpeg:x64-windows
vcpkg install qt6:x64-windows
vcpkg install cuda:x64-windows  # 如果使用CUDA
```

### 构建步骤

```bash
# 1. 创建构建目录
mkdir build
cd build

# 2. 配置项目
cmake .. -DENABLE_CUDA=ON -DENABLE_OPENCL=ON -DBUILD_TESTS=ON

# 3. 编译
make -j$(nproc)

# 4. 运行测试
ctest --output-on-failure

# 5. 运行程序
./gpu_player
```

### CMake 配置选项

```bash
cmake .. -DENABLE_CUDA=ON          # 启用CUDA支持
cmake .. -DENABLE_OPENCL=ON        # 启用OpenCL支持  
cmake .. -DENABLE_VULKAN=ON        # 启用Vulkan支持
cmake .. -DBUILD_TESTS=ON          # 构建测试
cmake .. -DBUILD_EXAMPLES=ON       # 构建示例
cmake .. -DCMAKE_BUILD_TYPE=Debug  # 调试模式
```

## 💻 运行程序

### 命令行参数

```bash
# 直接运行
./gpu_player

# 加载音频文件
./gpu_player music.mp3

# 使用特定GPU后端
./gpu_player --backend=cuda music.mp3

# 显示帮助
./gpu_player --help
```

### 控制台命令 (完整版本)

```
play <文件>                    # 播放音频文件
pause                          # 暂停/继续播放
stop                           # 停止播放
seek <秒数>                    # 跳转到指定位置
eq <f1> <g1> <f2> <g2>        # 设置EQ参数
stats                          # 显示性能统计
volume <0-100>                # 调节音量
device <设备ID>                # 选择音频设备
backend <cuda/opencl/vulkan>   # 选择GPU后端
quit                           # 退出程序
```

## 🔍 故障排除

### 常见问题

#### 1. 构建失败 - 缺少依赖
```bash
# 检查已安装的包
dpkg -l | grep -E "(libasound|libjack|ffmpeg|cuda)"

# 安装缺失的依赖
sudo apt install -y libasound2-dev libjack-dev ffmpeg libavcodec-dev
```

#### 2. CUDA 相关问题
```bash
# 检查CUDA安装
nvidia-smi
nvcc --version

# 检查CUDA库
ldconfig -p | grep cuda
```

#### 3. 音频设备问题
```bash
# 检查音频设备
aplay -l                    # Linux
system_profiler SPAudioDataType  # macOS
```

#### 4. 权限问题
```bash
# 音频设备权限 (Linux)
sudo usermod -a -G audio $USER
# 重新登录生效
```

### 调试模式

```bash
# 构建调试版本
cmake .. -DCMAKE_BUILD_TYPE=Debug
make

# 运行调试
gdb ./gpu_player
# 或
valgrind ./gpu_player
```

## 🎯 性能优化

### 系统调优 (Linux)

```bash
# 提高音频优先级
sudo chrt -f 99 ./gpu_player

# 禁用CPU频率调节
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 增加音频缓冲区
sudo sysctl -w vm.swappiness=10
```

### GPU优化

```bash
# 检查GPU状态
nvidia-smi                    # NVIDIA
clinfo                        # OpenCL
vulkan-tools                  # Vulkan

# 设置GPU性能模式
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 2505,1177
```

## 📊 验证安装

### 功能测试

```bash
# 测试基本播放
./gpu_player_simple test.mp3

# 测试GPU检测
./gpu_player_simple --check-gpu

# 测试音频设备
./gpu_player_simple --list-devices

# 性能测试
./gpu_player_simple --benchmark
```

### 成功指标

- ✅ 程序正常启动
- ✅ 能加载音频文件
- ✅ 播放控制响应
- ✅ 统计显示正常
- ✅ 无崩溃或内存泄漏

## 🚀 下一步

1. **体验简化版本** - 理解基本功能
2. **安装完整依赖** - 构建完整版本  
3. **测试GPU加速** - 体验高性能音频处理
4. **探索高级功能** - EQ、滤波器、DSD支持

## 📞 支持

如遇到问题，请检查：
1. 系统依赖是否完整安装
2. 音频设备是否正常工作
3. GPU驱动是否为最新版本
4. 查看程序输出的错误信息

祝使用愉快！ 🎵