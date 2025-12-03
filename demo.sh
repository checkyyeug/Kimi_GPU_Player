#!/bin/bash

echo "=========================================="
echo "     GPU音乐播放器自动演示"
echo "=========================================="
echo ""

# 检查程序是否存在
if [ ! -f "./gpu_player_simple" ]; then
    echo "错误: 请先运行 'make' 构建程序"
    exit 1
fi

echo "启动GPU音乐播放器演示..."
echo "这将演示播放器的基本功能"
echo ""

# 创建演示用的音频文件路径
DEMO_FILE="demo_song.mp3"

# 使用expect或简单输入重定向进行演示
cat << 'EOF' | timeout 30 ./gpu_player_simple
play demo_song.mp3
stats
seek 30
stats
pause
stats
seek 60
pause
stats
stop
stats
quit
EOF

echo ""
echo "演示完成！"
echo ""
echo "你可以手动运行 './gpu_player_simple' 来体验完整功能"