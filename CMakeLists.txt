cmake_minimum_required(VERSION 3.15)

project(inferGPT VERSION 1.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

include_directories(include)


set(SOURCES
    main.cpp
    src/model.cpp
    src/bpe.cpp
    src/load_gpt2.cpp
    src/ops.cpp
    src/quantize.cpp
    src/benchmark.cpp
)


add_executable(inferGPT ${SOURCES})
