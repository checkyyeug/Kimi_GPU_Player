cmake_minimum_required(VERSION 3.18)
project(GPU_Player VERSION 1.0.0 LANGUAGES CXX C)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 选项
option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_OPENCL "Enable OpenCL support" ON)
option(ENABLE_VULKAN "Enable Vulkan support" OFF)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# 平台检测
if(WIN32)
    set(PLATFORM_NAME "windows")
    add_definitions(-DPLATFORM_WINDOWS)
elseif(APPLE)
    set(PLATFORM_NAME "macos")
    add_definitions(-DPLATFORM_MACOS)
elseif(UNIX)
    set(PLATFORM_NAME "linux")
    add_definitions(-DPLATFORM_LINUX)
endif()

# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# 查找依赖包
find_package(Threads REQUIRED)

# FFmpeg
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED 
    libavcodec>=58.0.0
    libavformat>=58.0.0
    libavutil>=56.0.0
    libswresample>=3.0.0
)

# Qt6 (用于GUI)
find_package(Qt6 COMPONENTS Core Widgets Charts)
if(Qt6_FOUND)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)
    add_definitions(-DHAS_QT6)
endif()

# CUDA
if(ENABLE_CUDA)
    find_package(CUDA 11.0)
    if(CUDA_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        add_definitions(-DHAS_CUDA)
        include_directories(${CUDA_INCLUDE_DIRS})
    endif()
endif()

# OpenCL
if(ENABLE_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        add_definitions(-DHAS_OPENCL)
        include_directories(${OpenCL_INCLUDE_DIRS})
    endif()
endif()

# Vulkan
if(ENABLE_VULKAN)
    find_package(Vulkan)
    if(Vulkan_FOUND)
        add_definitions(-DHAS_VULKAN)
        include_directories(${Vulkan_INCLUDE_DIRS})
        message(STATUS "  Vulkan库: ${Vulkan_LIBRARIES}")
        message(STATUS "  Vulkan包含目录: ${Vulkan_INCLUDE_DIRS}")
    else()
        message(STATUS "  Vulkan未找到")
    endif()
endif()

# 平台特定库
if(WIN32)
    # Windows音频API
    set(PLATFORM_LIBS 
        winmm
        mmdevapi
        ${CMAKE_SOURCE_DIR}/third_party/asio/asio.lib
    )
elseif(APPLE)
    # macOS框架
    find_library(COREAUDIO_FRAMEWORK CoreAudio)
    find_library(AUDIOUNIT_FRAMEWORK AudioUnit)
    find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
    set(PLATFORM_LIBS
        ${COREAUDIO_FRAMEWORK}
        ${AUDIOUNIT_FRAMEWORK}
        ${COREFOUNDATION_FRAMEWORK}
    )
elseif(UNIX)
    # Linux音频库
    find_package(ALSA REQUIRED)
    find_package(Jack)
    set(PLATFORM_LIBS
        ${ALSA_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
    )
    if(Jack_FOUND)
        list(APPEND PLATFORM_LIBS ${JACK_LIBRARIES})
        add_definitions(-DHAS_JACK)
    endif()
endif()

# 源文件收集
file(GLOB_RECURSE SOURCES 
    src/*.cpp
    src/*.c
)

file(GLOB_RECURSE HEADERS
    include/*.h
    include/*.hpp
)

# CUDA源文件
if(CUDA_FOUND)
    file(GLOB_RECURSE CUDA_SOURCES
        src/*.cu
        src/*.cuh
    )
    list(APPEND SOURCES ${CUDA_SOURCES})
endif()

# 创建主库
add_library(gpu_player_core STATIC ${SOURCES} ${HEADERS})

target_link_libraries(gpu_player_core
    ${CMAKE_THREAD_LIBS_INIT}
    ${FFMPEG_LIBRARIES}
    ${PLATFORM_LIBS}
)

if(CUDA_FOUND)
    target_link_libraries(gpu_player_core ${CUDA_LIBRARIES})
endif()

if(OpenCL_FOUND)
    target_link_libraries(gpu_player_core ${OpenCL_LIBRARIES})
endif()

if(Vulkan_FOUND)
    target_link_libraries(gpu_player_core ${Vulkan_LIBRARIES})
    message(STATUS "  链接Vulkan库: ${Vulkan_LIBRARIES}")
endif()

if(Qt6_FOUND)
    target_link_libraries(gpu_player_core Qt6::Core Qt6::Widgets Qt6::Charts)
endif()

# 可执行文件
add_executable(gpu_player src/main.cpp)
target_link_libraries(gpu_player gpu_player_core)

# Vulkan检测工具
if(Vulkan_FOUND)
    add_executable(vulkan_detect tools/vulkan_detect.cpp)
    target_link_libraries(vulkan_detect gpu_player_core)
endif()

# 测试
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 示例
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# 安装
install(TARGETS gpu_player gpu_player_core
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# 打包
set(CPACK_PACKAGE_NAME "GPU_Player")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GPU-accelerated Music Player")
set(CPACK_PACKAGE_VENDOR "GPU_Player_Team")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "GPU_Player")

if(WIN32)
    set(CPACK_GENERATOR "NSIS")
elseif(APPLE)
    set(CPACK_GENERATOR "DragNDrop")
elseif(UNIX)
    set(CPACK_GENERATOR "DEB")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libasound2, libjack0")
endif()

include(CPack)

# 打印配置信息
message(STATUS "=====================================")
message(STATUS "GPU Player Configuration:")
message(STATUS "  Platform: ${PLATFORM_NAME}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Support: ${CUDA_FOUND}")
message(STATUS "  OpenCL Support: ${OpenCL_FOUND}")
message(STATUS "  Vulkan Support: ${Vulkan_FOUND}")
message(STATUS "  Qt6 Support: ${Qt6_FOUND}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "=====================================")