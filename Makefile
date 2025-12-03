# GPU音乐播放器简化版 Makefile

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -pthread
INCLUDES = -I./include -I./src
TARGET = gpu_player_simple
SOURCES = src/main_simple.cpp

# 默认目标
all: $(TARGET)

# 构建目标
$(TARGET): $(SOURCES)
	@echo "Building GPU Music Player (Simplified Version)..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET)
	@echo "Build complete: $(TARGET)"

# 清理
clean:
	@echo "Cleaning build files..."
	rm -f $(TARGET)
	rm -rf build/
	@echo "Clean complete"

# 运行
run: $(TARGET)
	@echo "Running GPU Music Player..."
	./$(TARGET)

# 构建完整版本（需要所有依赖）
full:
	@echo "Attempting full build with CMake..."
	@mkdir -p build
	@cd build && cmake .. 2>/dev/null && make -j4 2>/dev/null || echo "Full build failed - missing dependencies. Use 'make' for simplified version."

# 帮助
help:
	@echo "GPU Music Player Build System"
	@echo "============================="
	@echo "Available targets:"
	@echo "  make         - Build simplified version (recommended)"
	@echo "  make run     - Build and run simplified version"
	@echo "  make full    - Attempt full build (requires all dependencies)"
	@echo "  make clean   - Clean build files"
	@echo "  make help    - Show this help"
	@echo ""
	@echo "Usage examples:"
	@echo "  make run                    # Build and run"
	@echo "  ./gpu_player_simple         # Run the player"
	@echo "  ./gpu_player_simple song.mp3 # Load and play a file"

.PHONY: all clean run full help