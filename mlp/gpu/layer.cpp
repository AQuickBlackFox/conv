#include<hip/hip_runtime.h>
#include<iostream>

#undef hipLaunchKernel
#define hipLaunchKernel hipLaunchKernelGGL

#define X1_WIDTH 128
#define X1_HEIGHT 128
#define W1_HEIGHT 4
#define W1_WIDTH X1_HEIGHT
#define Y1_HEIGHT W1_HEIGHT
#define Y1_WIDTH X1_WIDTH

template<typename T, int height, int width>
struct Matrix{
  int h, w;
  T *ptr;
  Matrix(T *ptr) : ptr(ptr), h(height), w(width){}
  Matrix() = delete;
  __device__ __host__ inline T& operator[](int id){
    return ptr[id];
  }
  __device__ __host__ inline const T& operator[](int id) const {
    return ptr[id];
  }
  ~Matrix(){}
};

template<typename T, int w_height, int w_width, int x_height, int x_width>
__global__ void Dot(Matrix<T, w_height, x_width> Y,
                    Matrix<T, w_height, w_width> W,
                    Matrix<T, x_height, x_width> X)
{
  int tx = hipThreadIdx_x;
  int ty = hipThreadIdx_y;
  X[tx] = X[tx] + 1.0f;
}

int main(){
  float *x1d, *w1d, *y1d;
  hipMalloc(&x1d, sizeof(float)*X1_WIDTH*X1_HEIGHT);
  hipMalloc(&w1d, sizeof(float)*W1_WIDTH*W1_HEIGHT);
  hipMalloc(&y1d, sizeof(float)*Y1_WIDTH*Y1_HEIGHT);
  Matrix<float, X1_HEIGHT, X1_WIDTH> X1(x1d);
  Matrix<float, W1_HEIGHT, W1_WIDTH> W1(w1d);
  Matrix<float, Y1_HEIGHT, Y1_WIDTH> Y1(y1d);
  hipLaunchKernel((Dot<float, W1_HEIGHT, W1_WIDTH, X1_HEIGHT, X1_WIDTH>), dim3(1,1,1), dim3(X1_WIDTH, 4,1), 0, 0, Y1, W1, X1);
}
