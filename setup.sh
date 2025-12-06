#!/bin/bash

if [ -d "build" ]; then
    rm -rf build
    echo "removing existing build/ directory."
fi

mkdir build
cd build
cmake ..
make -j
cd .. && ./build/inferGPT