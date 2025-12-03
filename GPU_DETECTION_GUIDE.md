# GPU检测功能指南

## 🎯 概述

GPU音乐播放器现在包含完整的**GPU检测功能**，可以自动检测系统中可用的GPU计算后端，包括Vulkan、CUDA和OpenCL。这为音频处理提供了强大的硬件加速支持。

## 🔍 检测功能

### 支持的GPU后端

1. **🥇 CUDA** (NVIDIA专用)
   - 最佳性能表现
   - 专为NVIDIA GPU优化
   - 成熟的音频处理生态

2. **🥈 Vulkan** (跨平台)
   - 现代GPU架构支持
   - 跨平台兼容性
   - 低延迟音频处理

3. **🥉 OpenCL** (通用)
   - 最广泛的硬件支持
   - 标准化计算接口
   - 良好的可移植性

## 🚀 快速开始

### 1. 构建增强版本

```bash
# 使用增强版Makefile
make -f Makefile.enhanced

# 或者直接编译
g++ -std=c++17 -Wall -Wextra -O2 -pthread -I./include -I./src \
    src/main_enhanced_simple.cpp src/gpu/VulkanDetector.cpp \
    -o gpu_player_enhanced
```

### 2. 运行GPU检测

```bash
# 运行增强版播放器
./gpu_player_enhanced

# 在程序中使用GPU检测命令
> gpu
```

### 3. 使用独立GPU检测工具

```bash
# 构建独立检测工具
g++ -std=c++17 -Wall -Wextra -O2 -pthread -I./include -I./src \
    tools/gpu_detect_standalone.cpp src/gpu/VulkanDetector.cpp \
    -o gpu_detect

# 运行基本检测
./gpu_detect

# 详细Vulkan检测
./gpu_detect --vulkan

# 所有后端检测
./gpu_detect --all

# JSON格式输出
./gpu_detect --json
```

## 📋 检测结果说明

### 成功检测示例

```
🔍 GPU后端检测:
==========================================
     GPU音乐播放器 - GPU检测报告
==========================================

✅ 检测到 1 个GPU后端

🎯 Vulkan 后端:
  ✅ 状态: 可用
  🎯 设备: Intel GPU 1
  🚛 驱动: 1.2.0

🏆 推荐配置:
  🥈 备选: Vulkan后端 (跨平台，性能好)
==========================================
```

### 检测失败示例

```
❌ 未检测到支持的GPU后端

🔧 建议:
1. 确保安装了最新的GPU驱动程序
2. 安装相应的GPU计算库:
   • CUDA: NVIDIA GPU + CUDA Toolkit
   • Vulkan: 任何现代GPU + Vulkan运行时
   • OpenCL: 通用GPU计算库
3. 检查硬件兼容性
```

## 🔧 系统要求

### Vulkan支持

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install libvulkan1 vulkan-tools
sudo apt install mesa-vulkan-drivers

# 开发库（可选）
sudo apt install libvulkan-dev
```

#### CentOS/RHEL/Fedora
```bash
sudo yum install vulkan vulkan-tools
sudo yum install mesa-vulkan-drivers
```

#### macOS
```bash
brew install vulkan-loader vulkan-headers
brew install molten-vk
```

#### Windows
- 安装最新GPU驱动
- 可选：安装Vulkan SDK

### CUDA支持 (NVIDIA)

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

## 🎮 使用示例

### 在播放器中使用

```bash
./gpu_player_enhanced

# 在播放器内部
> gpu                    # 显示详细GPU信息
> play music.flac       # 播放音频文件
> stats                 # 查看播放统计（包含GPU信息）
```

### 独立检测工具

```bash
# 基本检测
./gpu_detect

# 详细Vulkan信息
./gpu_detect --vulkan

# 所有后端检测
./gpu_detect --all

# JSON格式（适合脚本处理）
./gpu_detect --json | jq '.vulkan.available'
```

## ⚡ 性能测试

播放器会自动进行简单的性能测试：

```
⚡ 简单性能测试:
CUDA重采样: 1.234 ms (125.3 MB/s)
CUDA EQ处理: 0.567 ms
Vulkan重采样: 2.345 ms (89.2 MB/s)
Vulkan EQ处理: 1.234 ms
```

## 🔍 故障排除

### Vulkan检测失败

1. **检查驱动安装**
   ```bash
   # Ubuntu/Debian
   sudo apt install mesa-vulkan-drivers
   
   # 检查Vulkan库
   ldconfig -p | grep vulkan
   ```

2. **验证Vulkan运行时**
   ```bash
   vulkaninfo
   ```

3. **检查GPU支持**
   ```bash
   lspci | grep -i vga
   ```

### CUDA检测失败

1. **检查NVIDIA驱动**
   ```bash
   nvidia-smi
   ```

2. **检查CUDA安装**
   ```bash
   nvcc --version
   ```

3. **检查环境变量**
   ```bash
   echo $CUDA_HOME
   echo $LD_LIBRARY_PATH
   ```

## 📊 输出格式

### 人类可读格式

默认输出为格式化文本，易于阅读和理解。

### JSON格式

适合脚本处理和自动化：

```json
{
  "vulkan": {
    "available": true,
    "version": "1.2.0",
    "driver": "Mesa 23.2.1",
    "devices": ["Intel GPU 1"]
  },
  "backends": [
    {
      "backend": "Vulkan",
      "available": true,
      "device": "Intel GPU 1",
      "driver_version": "23.2.1",
      "memory_mb": 0,
      "error": ""
    }
  ]
}
```

## 🚀 高级用法

### 程序化使用

```cpp
#include "gpu/VulkanDetector.h"

// 检测GPU支持
auto gpuList = GPUPlayer::GPUDetector::DetectAllGPUs();

// 检查特定后端
for (const auto& gpu : gpuList) {
    if (gpu.available && gpu.backend == "Vulkan") {
        std::cout << "Vulkan可用: " << gpu.deviceName << std::endl;
    }
}
```

### 集成到CI/CD

```bash
#!/bin/bash
# CI脚本中的GPU检测

if ./gpu_detect --json | jq -e '.vulkan.available' > /dev/null; then
    echo "✅ Vulkan支持可用，可以运行GPU加速测试"
    export ENABLE_GPU_TESTS=true
else
    echo "⚠️ Vulkan支持不可用，仅运行CPU测试"
    export ENABLE_GPU_TESTS=false
fi
```

## 📈 未来增强

计划中的功能：

1. **更详细的设备信息**
   - 显存使用统计
   - 温度监控
   - 功耗信息

2. **性能基准测试**
   - 标准化性能测试
   - 多设备对比
   - 性能报告生成

3. **智能后端选择**
   - 基于工作负载选择
   - 动态后端切换
   - 性能预测

4. **硬件兼容性检查**
   - 特定音频格式支持
   - 处理能力评估
   - 推荐配置建议

## 🎉 总结

GPU检测功能为GPU音乐播放器提供了：

- ✅ **自动硬件检测** - 无需手动配置
- ✅ **多后端支持** - CUDA、Vulkan、OpenCL
- ✅ **详细设备信息** - 全面了解GPU能力
- ✅ **性能建议** - 智能推荐最佳配置
- ✅ **故障诊断** - 快速定位问题原因

这使得播放器能够充分利用系统硬件资源，提供最佳的音频处理性能！🎵