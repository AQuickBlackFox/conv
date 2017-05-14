#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include<iostream>

#undef hipLaunchKernel 
#define hipLaunchKernel hipLaunchKernelGGL

#define X1_WIDTH 256
#define X1_HEIGHT 256
#define W1_HEIGHT 16
#define W1_WIDTH X1_HEIGHT
#define Y1_HEIGHT W1_HEIGHT
#define Y1_WIDTH X1_WIDTH
#define TILE_X 16
#define TILE_Y 16

template<typename T, int height, int width>
struct Matrix {
	int h, w;
	T *ptr;
	__device__ __host__ Matrix(T *ptr) : ptr(ptr), h(height), w(width) {}
	Matrix() = delete;
	__device__ __host__ inline T& operator[](int idx) {
		return ptr[idx];
	}
	__device__ __host__ inline const T& operator[](int idx) const {
		return ptr[idx];
	}
	__device__ __host__ inline T& operator()(int y, int x) {
		return ptr[x + y * width];
	}
	__device__ __host__ inline const T& operator()(int y, int x) const {
		return ptr[x + y * width];
	}
	__device__ __host__ ~Matrix() {}
};

template<typename T, int height, int width>
void PrintMatrix(Matrix<T, height, width> M) {
	std::cout << "Height x Width: " << height << " x " << width << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			std::cout << M.ptr[i + j*width] << " ";
		}
		std::cout << std::endl;
	}
}

template<typename T, int w_height, int w_width, int x_height, int x_width>
__global__ void Dot(Matrix<T, w_height, x_width> Y,
	Matrix<T, w_height, w_width> W,
	Matrix<T, x_height, x_width> X)
{
	int tx = hipThreadIdx_x;
	int ty = hipThreadIdx_y;

	int bx = hipBlockIdx_x;
	int by = hipBlockIdx_y;

	int row = ty + by * TILE_X;
	int col = tx + bx * TILE_X;

	__shared__ T sW[TILE_Y][TILE_X];
	__shared__ T sX[TILE_Y][TILE_X];
	T C = 0;

	for (int j = 0; j < w_width / TILE_Y; j++) {
		sW[ty][tx] = W[row*w_width + (j*TILE_X + tx)];
		sX[ty][tx] = X[col + (j*TILE_X+ty)*x_width];
		__syncthreads();
		for (int i = 0; i < TILE_Y; i++) {
			C = C + sW[ty][i] * sX[i][tx];
		}
		__syncthreads();
		Y[row*x_width+col] = C;
	}
}

int main() {
	float *x1, *w1, *y1;
	x1 = new float[X1_HEIGHT*X1_WIDTH];
	w1 = new float[W1_HEIGHT*W1_WIDTH];
	y1 = new float[Y1_HEIGHT*Y1_WIDTH];
	for (int j = 0; j < X1_HEIGHT; j++) {
		for (int i = 0; i < X1_WIDTH; i++) {
			x1[i + j*X1_WIDTH] = 1.0f;
		}
	}
	for (int j = 0; j < W1_HEIGHT; j++) {
		for (int i = 0; i < W1_WIDTH; i++) {
			w1[i + j*W1_WIDTH] = 0.5f;
		}
	}
	for (int j = 0; j < Y1_HEIGHT; j++) {
		for (int i = 0; i < Y1_WIDTH; i++) {
			y1[i + j*Y1_WIDTH] = 0.0f;
		}
	}

	float *x1d, *w1d, *y1d;
	hipMalloc(&x1d, sizeof(float)*X1_WIDTH*X1_HEIGHT);
	hipMalloc(&w1d, sizeof(float)*W1_WIDTH*W1_HEIGHT);
	hipMalloc(&y1d, sizeof(float)*Y1_WIDTH*Y1_HEIGHT);

	hipMemcpy(x1d, x1, sizeof(float)*X1_WIDTH*X1_HEIGHT, hipMemcpyHostToDevice);
	hipMemcpy(w1d, w1, sizeof(float)*W1_WIDTH*W1_HEIGHT, hipMemcpyHostToDevice);
	hipMemcpy(y1d, y1, sizeof(float)*Y1_WIDTH*Y1_HEIGHT, hipMemcpyHostToDevice);

	Matrix<float, X1_HEIGHT, X1_WIDTH> X1(x1d);
	Matrix<float, W1_HEIGHT, W1_WIDTH> W1(w1d);
	Matrix<float, Y1_HEIGHT, Y1_WIDTH> Y1(y1d);
	dim3 dimGrid(X1_WIDTH / TILE_X, W1_HEIGHT / TILE_X);
	dim3 dimBlock(TILE_Y, TILE_X);
	hipLaunchKernel((Dot<float, W1_HEIGHT, W1_WIDTH, X1_HEIGHT, X1_WIDTH>),dimGrid, dimBlock, 0, 0, Y1, W1, X1);
	std::cout << dimGrid.x << " " << dimGrid.y << std::endl;
	std::cout << dimBlock.x << " " << dimBlock.y << std::endl;
	hipDeviceSynchronize();
	hipMemcpy(y1, y1d, Y1_HEIGHT*Y1_WIDTH * sizeof(float), hipMemcpyDeviceToHost);
	Matrix<float, Y1_HEIGHT, Y1_WIDTH> Y(y1);
	PrintMatrix(Y);
	std::cout << std::endl;
}
