#!/bin/bash

echo "=========================================="
echo "     GPU音乐播放器增强版演示"
echo "=========================================="
echo ""

# 检查程序是否存在
if [ ! -f "./gpu_player_enhanced" ]; then
    echo "错误: 请先运行 'make -f Makefile.enhanced' 构建程序"
    exit 1
fi

echo "启动GPU音乐播放器增强版..."
echo "演示将展示GPU检测和播放控制功能"
echo ""

# 创建演示命令序列
cat << 'EOF' | timeout 30 ./gpu_player_enhanced
gpu
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
quit
EOF

echo ""
echo "演示完成！"
echo ""
echo "你可以手动运行 './gpu_player_enhanced' 来体验完整功能"
echo "使用 'gpu' 命令查看详细的GPU信息"