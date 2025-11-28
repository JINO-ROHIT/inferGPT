//TO-DO use vector and get rid of manual alloc for with arrays

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

