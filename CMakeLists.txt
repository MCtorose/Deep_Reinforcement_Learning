cmake_minimum_required(VERSION 3.16.3)
project(flyBallBuild)
set(CMAKE_CXX_STANDARD 14)

add_subdirectory(./extern/pybind11)

# 添加源文件
file(GLOB srcfile ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cpp)
#// 指定Python版本以及安装路径

pybind11_add_module(flyBall ${srcfile})



