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

4. You can either use the `setup.sh` or continue from step 5

```
bash setup.sh
```

5. create the build directory

```
mkdir build
cd build
```

6. configure with Cmake

```
cmake ..
```

7. build the project

```
make -j
```

8. run the executable

```
cd .. && ./build/inferGPT
```

### Current Benchmarks

| Vectorization | Sampling Strategy | Performance | Speedup |
|---|---|---|---|
| no SIMD | Temperature | 20 toks/sec | 1.0x |
| **NEON SIMD** (dot product) | Temperature | **57.27 toks/sec** | **2.9x** |


Roadmap
( would flash attention make sense on a CPU? flash attention style blocked matrices?)
- [ ] Add conditional compilation for metal archs
- [x] Operator fusion
- [x] Implement SIMD instructions
- [ ] Add quantization algorithms with performance benchmarking
- [ ] Support GPU operations via CUDA C++


### References

1. https://github.com/a1k0n/a1gpt/
2. https://github.com/karpathy/llama2.c
3. https://github.com/ggml-org/llama.cpp