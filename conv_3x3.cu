#include<cuda.h>
#include<cuda_runtime_api.h>
#include<iostream>
#include<sm_30_intrinsics.h>


#define IN_X 10
#define IN_Y 10
#define IN_N IN_X*IN_Y

#define W_X 3
#define W_Y 3
#define W_N W_X*W_Y

__global__ void Conv2(float *weights_d, float *input, float *output) {
	int tid = threadIdx.x;

	float w00, w01, w02, \
		w10, w11, w12, \
		w20, w21, w22;

	float var00, var01, var02, \
		var10, var11, var12, \
		var20, var21, var22;

	float tmp1, tmp2;

	float *weights = weights_d;

	w00 = weights[0];
	w01 = weights[1];
	w02 = weights[2];

	w10 = weights[5];
	w11 = weights[6];
	w12 = weights[7];

	w20 = weights[10];
	w21 = weights[11];
	w22 = weights[12];

	var01 = input[tid];
	var00 = __shfl_up(var01, 1, IN_X);
	var02 = __shfl_down(var01, 1, IN_X);


	var11 = input[tid + IN_X];
	var10 = __shfl_up(var11, 1, IN_X);
	var12 = __shfl_down(var11, 1, IN_X);

	var21 = input[tid + 2 * IN_X];
	var20 = __shfl_up(var21, 1, IN_X);
	var22 = __shfl_down(var21, 1, IN_X);

	tmp1 = w00 * var00;
	tmp2 = w01 * var01;
	tmp1 = w02 * var02 + tmp1;

	tmp2 = w10 * var10 + tmp2;
	tmp1 = w11 * var11 + tmp1;
	tmp2 = w12 * var12 + tmp2;

	tmp1 = w20 * var20 + tmp1;
	tmp2 = w21 * var21 + tmp2;
	tmp1 = w22 * var22 + tmp1;

//	output[tid] = tmp1 + tmp2;
	output[tid] = var00 * w00 + var01 * w01 + var02 * w02 + \
		var10 * w10 + var11 * w11 + var12 * w12 + \
		var20 * w20 + var21 * w21 + var22 * w22;
//	printf("Value at block: %d and thread: %d and index: %d is %f, %f, %f, %f, %f, %f, %f, %f, %f : %f\n", blockIdx.x, threadIdx.x, tid, var00, var01, var02, var10, var11, var12, var20, var21, var22, output[tid]);

	for (unsigned i = 1; i<IN_Y; i++) {

		var00 = var10;
		var01 = var11;
		var02 = var12;
	
		var10 = var20;
		var11 = var21;
		var12 = var22;
	
	
		var21 = input[tid + (i + 2)*IN_X];
		var20 = __shfl_up(var21, 1, IN_X);
		var22 = __shfl_up(var21, 1, IN_X);

		output[tid + i * IN_X] = var00 * w00 + var01 * w01 + var02 * w02 + \
								var10 * w10 + var11 * w11 + var12 * w12 + \
								var20 * w20 + var21 * w21 + var22 * w22;
//		printf("Value at block: %d and thread: %d and index: %d is %f, %f, %f, %f, %f, %f, %f, %f, %f : %f\n", blockIdx.x, threadIdx.x, tid+i*IN_X, var00, var01, var02, var10, var11, var12, var20, var21, var22, output[tid+i*IN_X]);
	}

}

__global__ void Conv(float *weights, float *input, float *output) {
	unsigned tx = threadIdx.x;
	unsigned ty = threadIdx.y;
	unsigned Tx = blockDim.x;
	unsigned Ty = blockDim.y;
	float w00, w01, w02, \
		w10, w11, w12, \
		w20, w21, w22;
	
	float var00, var01, var02, \
		var10, var11, var12, \
		var20, var21, var22;
	
	w00 = weights[0];
	w01 = weights[1];
	w02 = weights[2];
	w10 = weights[3];
	w11 = weights[4];
	w12 = weights[5];
	w20 = weights[6];
	w21 = weights[7];
	w22 = weights[8];

	var01 = input[tx + ty * Tx];
	var00 = __shfl_up(var01, 1, 32);
	var02 = __shfl_down(var01, 1, 32);

	var11 = input[tx + (ty+1) *Tx];
	var10 = __shfl_up(var11, 1, 32);
	var12 = __shfl_down(var11, 1, 32);

	var21 = input[tx + (ty + 2)*Tx];
	var20 = __shfl_up(var21, 1, 32);
	var22 = __shfl_down(var21, 1, 32);

	if (tx > 0 && tx < IN_X-1 && ty < IN_Y-2) {
		output[(tx-1) + ty* (Tx-2)] = w00 * var00 + w01 * var01 + w02 * var02 + \
			w10 * var10 + w11 * var11 + w12 * var12 + \
			w20 * var20 + w21 * var21 + w22 * var22;
//		printf("Value at block: %d and thread X: %d and thread Y: %d and index: %d is %f, %f, %f, %f, %f, %f, %f, %f, %f : %f\n", blockIdx.x, threadIdx.x, threadIdx.y, tx + ty*Tx, var00, var01, var02, var10, var11, var12, var20, var21, var22, output[(tx-1) + ty*Tx]);
	}

	//printf("Value at block: %d and thread X: %d and thread Y: %d and index: %d is %f, %f, %f, %f, %f, %f, %f, %f, %f : %f\n", blockIdx.x, threadIdx.x, threadIdx.y, tx + ty*Tx, var00, var01, var02, var10, var11, var12, var20, var21, var22, output[tx + ty*Tx]);
}

int main() {
	float *input_h, *output_h, *weights_h, *output_h2;
	input_h = new float[IN_N];
	output_h = new float[IN_N];
	weights_h = new float[W_N];
	output_h2 = new float[IN_N];
	for (unsigned i = 0;i < IN_N;i++) {
		input_h[i] = i*1.0f;
		output_h[i] = 0.0f;
		output_h2[i] = 0.0f;
	}
	for (unsigned i = 0;i < W_N;i++) {
		weights_h[i] = 1.0f;
	}
	float *output_d, *input_d, *weights_d, *output_d2;
	size_t size = sizeof(float)*IN_N;
	cudaMalloc((void**)&input_d, size);
	cudaMalloc((void**)&output_d, size);
	cudaMalloc((void**)&weights_d, sizeof(float)*W_N);
	cudaMalloc((void**)&output_d2, size);

	cudaMemcpy(output_d, output_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(weights_d, weights_h, sizeof(float)*W_N, cudaMemcpyHostToDevice);
	cudaMemcpy(output_d2, output_h2, size, cudaMemcpyHostToDevice);

	Conv << <dim3(1, 1, 1), dim3(IN_X, IN_Y, 1) >> > (weights_d, input_d, output_d);
	Conv2 << <dim3(1, 1, 1), dim3(IN_X, 1, 1) >> > (weights_d, input_d, output_d2);

	cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(output_h2, output_d2, size, cudaMemcpyDeviceToHost);
	for (unsigned i = 0;i < IN_N;i++) {
		if (output_h2[i] != output_h[i]) {
			std::cout << "OUput invalid at: " << i <<" "<<input_h[i]<<" "<<output_h[i]<<" "<<output_h2[i]<<std::endl;
		}
	}
	std::cout << "Bye!" << std::endl;
}
