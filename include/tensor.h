//TO-DO use vector and get rid of manual alloc for with arrays
//TO-DO find a nicer way to merge the Tensors together

#pragma once

#include <iostream>
#include <assert.h>

template<int N>
class Tensor{
    
    public:
        int shape[N];
        float* data;
        float* alloc;

        Tensor() : data(nullptr), alloc(nullptr){};

        Tensor(float* _data, int i){
            assert(N == 1);
            shape[0] = i;
            data = _data;
            alloc = nullptr;
        }

        Tensor(float* _data, int i, int j){
            assert(N == 2);
            shape[0] = i;
            shape[1] = j;
            data = _data;
            alloc = nullptr;
        }

        void _alloc(size_t nfloats){
            //aligned float array
            alloc = new float[nfloats + 7];
            data = (float *)(((uintptr_t)alloc + 31) & ~31);
        }

        Tensor(int i){
            assert(N == 1);
            shape[0] = i;
            _alloc(i);
        }

        Tensor(int i, int j) {
            assert(N == 2);
            shape[0] = i;
            shape[1] = j;
            _alloc(i*j);
        }

        Tensor(int i, int j, int k) {
            assert(N == 3);
            shape[0] = i;
            shape[1] = j;
            shape[2] = k;
            _alloc(i*j*k);
        }

        //to index
        float& operator[](int i) const {
            if (N != 1) {
            fprintf(stderr, "Tensor: operator[]: expected 1 dimension, got %d\n", N);
            abort();
            }
            if (i >= shape[0]) {
            fprintf(stderr, "Tensor: out of bounds: %d >= %d\n", i, shape[N-1]);
            abort();
            }
            return data[i];
        }

        float& operator()(int i, int j) const {
            if (N != 2) {
            fprintf(stderr, "Tensor: operator[]: expected 2 dimensions, got %d\n", N);
            abort();
            }
            if (i >= shape[0]) {
            fprintf(stderr, "Tensor: out of bounds: %d >= %d\n", i, shape[N-2]);
            abort();
            }
            if (j >= shape[1]) {
            fprintf(stderr, "Tensor: out of bounds: %d >= %d\n", j, shape[N-1]);
            abort();
            }
            return data[i * shape[1] + j];
        }

        //only along the first dimension, wont work column wise
        Tensor<N-1> slice(int i) const {
            if (N <= 1) {
                fprintf(stderr, "Tensor: row: expected >1 dimensions, got %d\n", N);
                abort();
            }
            if (i >= shape[0]) {
                fprintf(stderr, "Tensor: out of bounds: %d >= %d\n", i, shape[0]);
                abort();
            }
            // return new tensor with no alloc, so it won't destroy the underlying array
            // when it goes out of scope
            Tensor<N-1> out;
            int stride = 1;
            for (int j = 0; j < N-1; j++) {
            out.shape[j] = shape[j+1];
            stride *= shape[j+1];
            }
            if (data != NULL) {
            out.data = data + i * stride;
            }
            out.alloc = NULL;
            return out;
        }
};


template<int N>
class Tensor_Quant{
    
    public:
        int shape[N];
        int8_t* data;
        int8_t* alloc;
        float scale;

        Tensor_Quant() : data(nullptr), alloc(nullptr), scale(0){};

        Tensor_Quant(int8_t* _data, int i, float _scale){
            assert(N == 1);
            shape[0] = i;
            data = _data;
            scale = _scale;
            alloc = nullptr;
        }

        Tensor_Quant(int8_t* _data, int i, int j, float _scale){
            assert(N == 2);
            shape[0] = i;
            shape[1] = j;
            data = _data;
            scale = _scale;
            alloc = nullptr;
        }

        void _alloc(size_t nfloats){
            alloc = new int8_t[nfloats + 7];
            data = (int8_t *)(((uintptr_t)alloc + 31) & ~31);
        }

        Tensor_Quant(int i){
            assert(N == 1);
            shape[0] = i;
            _alloc(i);
        }

        Tensor_Quant(int i, int j) {
            assert(N == 2);
            shape[0] = i;
            shape[1] = j;
            _alloc(i*j);
        }

        Tensor_Quant(int i, int j, int k) {
            assert(N == 3);
            shape[0] = i;
            shape[1] = j;
            shape[2] = k;
            _alloc(i*j*k);
        }

        int8_t& operator[](int i) const {
            if (N != 1) {
            fprintf(stderr, "Tensor_Quant: operator[]: expected 1 dimension, got %d\n", N);
            abort();
            }
            if (i >= shape[0]) {
            fprintf(stderr, "Tensor_Quant: out of bounds: %d >= %d\n", i, shape[N-1]);
            abort();
            }
            return data[i];
        }

        int8_t& operator()(int i, int j) const {
            if (N != 2) {
            fprintf(stderr, "Tensor_Quant: operator[]: expected 2 dimensions, got %d\n", N);
            abort();
            }
            if (i >= shape[0]) {
            fprintf(stderr, "Tensor_Quant: out of bounds: %d >= %d\n", i, shape[N-2]);
            abort();
            }
            if (j >= shape[1]) {
            fprintf(stderr, "Tensor_Quant: out of bounds: %d >= %d\n", j, shape[N-1]);
            abort();
            }
            return data[i * shape[1] + j];
        }

        Tensor_Quant<N-1> slice(int i) const {
            if (N <= 1) {
                fprintf(stderr, "Tensor: row: expected >1 dimensions, got %d\n", N);
                abort();
            }
            if (i >= shape[0]) {
                fprintf(stderr, "Tensor: out of bounds: %d >= %d\n", i, shape[0]);
                abort();
            }
            Tensor_Quant<N-1> out;
            int stride = 1;
            for (int j = 0; j < N-1; j++) {
            out.shape[j] = shape[j+1];
            stride *= shape[j+1];
            }
            if (data != NULL) {
            out.data = data + i * stride;
            }
            out.alloc = NULL;
            return out;
        }

        Tensor<N> dequantize() const {
            Tensor<N> out;
            for (int i = 0; i < N; i++) {
                out.shape[i] = shape[i];
            }
            size_t total_size = 1;
            for (int i = 0; i < N; i++) {
                total_size *= shape[i];
            }
            out._alloc(total_size);

            for (size_t i = 0; i < total_size; i++) {
                out.data[i] = ((float)data[i]) / scale;
            }
            return out;
        }

        void dequantize_to(Tensor<N>& out) const {
            for (int i = 0; i < N; i++) {
                assert(out.shape[i] == shape[i]);
            }
            size_t total_size = 1;
            for (int i = 0; i < N; i++) {
                total_size *= shape[i];
            }

            for (size_t i = 0; i < total_size; i++) {
                out.data[i] = ((float)data[i]) / scale;
            }
        }

        float get_dequantized(int i) const {
            assert(N == 1);
            assert(i < shape[0]);
            return ((float)data[i]) / scale;
        }

        float get_dequantized(int i, int j) const {
            assert(N == 2);
            assert(i < shape[0] && j < shape[1]);
            return ((float)data[i * shape[1] + j]) / scale;
        }
};
