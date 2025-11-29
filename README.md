## inferGPT

A high performance C/C++ inference engine that runs on CPU.


### Setup

1. install the virtual environment.

```
uv sync
```

2. first convert the weights from safetensors.

```
python3 scripts/convert_weight.py
```

3. move the weights and index.json to `/model`.

4. create the build directory

```
mkdir build
cd build
```

5. configure with Cmake

```
cmake ..
```

6. build the project

```
make -j
```

7. run the executable

```
cd .. && ./build/inferGPT
```

Roadmap
- [ ] Operator fusion + multithreading
- [ ] Implement SIMD instructions
- [ ] Add quantization algorithms with performance benchmarking
- [ ] Support GPU operations via CUDA C++


### References

1. https://github.com/a1k0n/a1gpt/
2. https://github.com/karpathy/llama2.c
3. https://github.com/ggml-org/llama.cpp