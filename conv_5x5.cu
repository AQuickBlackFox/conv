/*
Copyright (c) 2016 Aditya Atluri. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<iostream>
#include<time.h>
#include<sm_30_intrinsics.h>

#define FIL_X 5
#define FIL_Y 5

#define IMG_N 200
#define IMG_C 3

#define IMG_IN_X 20
#define IMG_IN_Y 20

#define IMG_OUT_X 16
#define IMG_OUT_Y 16

#define IMG_IN_NUM IMG_IN_X * IMG_IN_Y
#define IMG_OUT_NUM IMG_OUT_X * IMG_OUT_Y

#define FIL_NUM FIL_X * FIL_Y
#define FIL_N 64

/*
Weights access pattern

for(unsigned i=0;i<FIL_N;i++){
weights = weights_d  + (bx%IMG_C) * FIL_NUM + i * (IMG_C*FIL_NUM);
}

*/

__device__ void convolve(float *weights_d, float *input, float *output) {
	int tid = threadIdx.x;

	float w00, w01, w02, w03, w04, \
		w10, w11, w12, w13, w14, \
		w20, w21, w22, w23, w24, \
		w30, w31, w32, w33, w34, \
		w40, w41, w42, w43, w44;

	float var00, var01, var02, var03, var04, \
		var10, var11, var12, var13, var14, \
		var20, var21, var22, var23, var24, \
		var30, var31, var32, var33, var34, \
		var40, var41, var42, var43, var44;

	float tmp1, tmp2;

	float *weights = weights_d;

	w00 = weights[0];
	w01 = weights[1];
	w02 = weights[2];
	w03 = weights[3];
	w04 = weights[4];

	w10 = weights[5];
	w11 = weights[6];
	w12 = weights[7];
	w13 = weights[8];
	w14 = weights[9];

	w20 = weights[10];
	w21 = weights[11];
	w22 = weights[12];
	w23 = weights[13];
	w24 = weights[14];

	w30 = weights[15];
	w31 = weights[16];
	w32 = weights[17];
	w33 = weights[18];
	w34 = weights[19];

	w40 = weights[20];
	w41 = weights[21];
	w42 = weights[22];
	w43 = weights[23];
	w44 = weights[24];

	var02 = input[tid];
	var00 = __shfl_up(var02, 2, 64);
	var01 = __shfl_up(var02, 1, 64);
	var03 = __shfl_down(var02, 1, 64);
	var04 = __shfl_down(var02, 2, 64);


	var12 = input[tid + IMG_IN_X];
	var10 = __shfl_up(var12, 2, 64);
	var11 = __shfl_up(var12, 1, 64);
	var13 = __shfl_down(var12, 1, 64);
	var14 = __shfl_down(var12, 2, 64);


	var22 = input[tid + 2 * IMG_IN_X];
	var20 = __shfl_up(var22, 2, 64);
	var21 = __shfl_up(var22, 1, 64);
	var23 = __shfl_down(var22, 1, 64);
	var24 = __shfl_down(var22, 2, 64);


	var32 = input[tid + 3 * IMG_IN_X];
	var30 = __shfl_up(var32, 2, 64);
	var31 = __shfl_up(var32, 1, 64);
	var33 = __shfl_down(var32, 1, 64);
	var34 = __shfl_down(var32, 2, 64);


	var42 = input[tid + 4 * IMG_IN_X];
	var40 = __shfl_up(var42, 2, 64);
	var41 = __shfl_up(var42, 1, 64);
	var43 = __shfl_down(var42, 1, 64);
	var44 = __shfl_down(var42, 2, 64);


	tmp1 = w00 * var00;
	tmp2 = w01 * var01;
	tmp1 = w02 * var02 + tmp1;
	tmp2 = w03 * var03 + tmp2;
	tmp1 = w04 * var04 + tmp1;

	tmp2 = w10 * var10 + tmp2;
	tmp1 = w11 * var11 + tmp1;
	tmp2 = w12 * var12 + tmp2;
	tmp1 = w13 * var13 + tmp1;
	tmp2 = w14 * var14 + tmp2;

	tmp1 = w20 * var20 + tmp1;
	tmp2 = w21 * var21 + tmp2;
	tmp1 = w22 * var22 + tmp1;
	tmp2 = w23 * var23 + tmp2;
	tmp1 = w24 * var24 + tmp1;

	tmp2 = w30 * var30 + tmp2;
	tmp1 = w31 * var31 + tmp1;
	tmp2 = w32 * var32 + tmp2;
	tmp1 = w33 * var33 + tmp1;
	tmp2 = w34 * var34 + tmp2;

	tmp1 = w40 * var40 + tmp1;
	tmp2 = w41 * var41 + tmp2;
	tmp1 = w42 * var42 + tmp1;
	tmp2 = w43 * var43 + tmp2;
	tmp1 = w44 * var44 + tmp1;

	if (tid > 1 && tid < IMG_IN_X - 2) {
		output[tid - 2] = tmp1 + tmp2;
		//		printf("Value %f calculated at thread: %d in block: %d at row: %d\n", output[tid-2], threadIdx.x, blockIdx.x, tid - 2);
	}


	for (unsigned i = 1; i<IMG_OUT_Y; i++) {

		var00 = var10;
		var01 = var11;
		var02 = var12;
		var03 = var13;
		var04 = var14;

		var10 = var20;
		var11 = var21;
		var12 = var22;
		var13 = var23;
		var14 = var24;

		var20 = var30;
		var21 = var31;
		var22 = var32;
		var23 = var33;
		var24 = var34;

		var30 = var40;
		var31 = var41;
		var32 = var42;
		var33 = var43;
		var34 = var44;

		var42 = input[tid + (i + 4)*IMG_IN_X];
		var40 = __shfl_up(var42, 2, 64);
		var41 = __shfl_up(var42, 1, 64);
		var43 = __shfl_down(var42, 1, 64);
		var44 = __shfl_down(var42, 2, 64);

		tmp1 = w00 * var00;
		tmp2 = w01 * var01;
		tmp1 = w02 * var02 + tmp1;
		tmp2 = w03 * var03 + tmp2;
		tmp1 = w04 * var04 + tmp1;

		tmp2 = w10 * var10 + tmp2;
		tmp1 = w11 * var11 + tmp1;
		tmp2 = w12 * var12 + tmp2;
		tmp1 = w13 * var13 + tmp1;
		tmp2 = w14 * var14 + tmp2;

		tmp1 = w20 * var20 + tmp1;
		tmp2 = w21 * var21 + tmp2;
		tmp1 = w22 * var22 + tmp1;
		tmp2 = w23 * var23 + tmp2;
		tmp1 = w24 * var24 + tmp1;

		tmp2 = w30 * var30 + tmp2;
		tmp1 = w31 * var31 + tmp1;
		tmp2 = w32 * var32 + tmp2;
		tmp1 = w33 * var33 + tmp1;
		tmp2 = w34 * var34 + tmp2;

		tmp1 = w40 * var40 + tmp1;
		tmp2 = w41 * var41 + tmp2;
		tmp1 = w42 * var42 + tmp1;
		tmp2 = w43 * var43 + tmp2;
		tmp1 = w44 * var44 + tmp1;

		if (tid > 1 && tid < IMG_IN_X - 2) {
			output[tid - 2 + i * IMG_OUT_X] = tmp1 + tmp2;
		}

	}


}

__global__ void conv_5x5(float *weights_d, float *input_d, float *output_d) {
	unsigned bx = blockIdx.x;
	
	float *input = input_d + (bx%IMG_C) * IMG_IN_NUM + (bx / (IMG_C*FIL_N))*(IMG_C*IMG_IN_NUM);
	float *weights = weights_d + (bx % (IMG_C*FIL_N))*FIL_NUM;
	float *output = output_d + bx * IMG_OUT_NUM;
	convolve(weights, input, output);
}


/*

__global__ void conv_5x5(float *weights_d, float *input_d, float *output_d)
{
	unsigned bx = blockIdx.x;
	
	extern __shared__ float weights_lds[];
	if (threadIdx.x < FIL_X) {
		for (unsigned k = 0;k < FIL_N; k++) {
			for (unsigned j = 0;j < FIL_Y;j++) {
				weights_lds[threadIdx.x + j*FIL_X + k*FIL_NUM*IMG_C] = weights_d[threadIdx.x + j*FIL_X + (bx % IMG_C) * FIL_NUM + k*FIL_NUM*IMG_C];
			}
		}
	}
	
	float *input, *weights, *output;

	unsigned quo = bx / IMG_C;
	unsigned rem = bx % IMG_C;
	output = output_d + bx * IMG_OUT_NUM;
	input = input_d + bx * IMG_IN_NUM;
	for (unsigned j = 0;j < FIL_N;j++) {
		weights = weights_lds;//weights_d + rem * FIL_NUM + j * IMG_C * FIL_NUM;
		output = output_d + quo * IMG_C * IMG_OUT_NUM * FIL_N + rem * IMG_OUT_NUM + j * IMG_C * IMG_OUT_NUM;
		
		input = input_d + bx * IMG_IN_NUM;
		for(unsigned i=0;i<FIL_N;i++){
		weights = weights_d + rem * FIL_NUM + i * IMG_C * FIL_NUM;
		output = output_d + quo * FIL_N * IMG_C * IMG_OUT_NUM ;
		convolve(weights, input, output);
		}
		
		convolve(weights, input, output);
	}

	if (threadIdx.x == 0) {
		//		printf("blocks: %d at rem: %d\n", bx, (bx%IMG_C)*FIL_NUM);
	}

	if (threadIdx.x == 0) {
		for (unsigned i = 0;i < IMG_OUT_NUM;i++) {
			//			printf("%f of output at block: %d at id: %d \n", output[i], bx, i);
		}
		for (unsigned i = 0;i < IMG_IN_NUM;i++) {
			//			printf("%f of output at block: %d at id: %d \n", input[i], bx, i);
		}
		//		printf("%ld Pointer %ld\n", (unsigned long)(input), (unsigned long)(output));
		for (unsigned i = 0;i < IMG_OUT_NUM * IMG_N;i++) {
			//			printf("Value %f is present at %d th index\n", output_d[i], i);
		}
	}

	//  }
}
*/
void genCPU(float *ih_h, float *wh_h, float *oh_h)
{
	for (unsigned n = 0;n<IMG_N*IMG_C;n++) {
		float* ih = ih_h + n * IMG_IN_NUM;
		float *wh = wh_h;
		float *oh = oh_h + n*IMG_OUT_NUM;
		for (unsigned l = 0;l<IMG_OUT_Y;l++)
		{
			for (unsigned k = 0;k<IMG_OUT_X;k++) {
				unsigned id = k + l *IMG_OUT_X;
				unsigned acc_id = (k + 2) + (l + 2) * IMG_IN_X;
				oh[id] = wh[0] * ih[acc_id - (IMG_IN_X * 2 + 2)] + \
					wh[1] * ih[acc_id - (IMG_IN_X * 2 + 1)] + \
					wh[2] * ih[acc_id - (IMG_IN_X * 2 + 0)] + \
					wh[3] * ih[acc_id - (IMG_IN_X * 2 - 1)] + \
					wh[4] * ih[acc_id - (IMG_IN_X * 2 - 2)] + \
					wh[5] * ih[acc_id - (IMG_IN_X * 1 + 2)] + \
					wh[6] * ih[acc_id - (IMG_IN_X * 1 + 1)] + \
					wh[7] * ih[acc_id - (IMG_IN_X * 1 + 0)] + \
					wh[8] * ih[acc_id - (IMG_IN_X * 1 - 1)] + \
					wh[9] * ih[acc_id - (IMG_IN_X * 1 - 2)] + \
					wh[10] * ih[acc_id - (2)] + \
					wh[11] * ih[acc_id - (1)] + \
					wh[12] * ih[acc_id - (0)] + \
					wh[13] * ih[acc_id + (1)] + \
					wh[14] * ih[acc_id + (2)] + \
					wh[15] * ih[acc_id + (IMG_IN_X - 2)] + \
					wh[16] * ih[acc_id + (IMG_IN_X - 1)] + \
					wh[17] * ih[acc_id + (IMG_IN_X + 0)] + \
					wh[18] * ih[acc_id + (IMG_IN_X + 1)] + \
					wh[19] * ih[acc_id + (IMG_IN_X + 2)] + \
					wh[20] * ih[acc_id + (IMG_IN_X * 2 - 2)] + \
					wh[21] * ih[acc_id + (IMG_IN_X * 2 - 1)] + \
					wh[22] * ih[acc_id + (IMG_IN_X * 2 + 0)] + \
					wh[23] * ih[acc_id + (IMG_IN_X * 2 + 1)] + \
					wh[24] * ih[acc_id + (IMG_IN_X * 2 + 2)];
			}
		}
	}


}

void verifyCPU(float *in1, float *in2) {
	for (unsigned i = 0;i<IMG_OUT_X*IMG_OUT_Y*IMG_N;i++) {
		if (in1[i] != 12.5f) {
			std::cout << "Bad output at: " << i << " GPU: " << in1[i] << " CPU: " << in2[i] << std::endl;
			return;
		}
	}
}

int main() {
	float *wh, *ih, *oh, *oh2;
	float *wd, *id, *od;
	wh = new float[FIL_NUM*IMG_C*FIL_N];
	ih = new float[IMG_IN_NUM*IMG_N*IMG_C];
	oh = new float[IMG_OUT_NUM*IMG_N*IMG_C*FIL_N];
	oh2 = new float[IMG_OUT_NUM*IMG_N*IMG_C*FIL_N];

	for (unsigned i = 0;i<IMG_IN_NUM*IMG_N*IMG_C;i++) {
		ih[i] = 1.0f;
	}
	for (unsigned i = 0;i<IMG_OUT_NUM*IMG_N*IMG_C*FIL_N;i++) {
		oh[i] = 0.0f;
		oh2[i] = 0.0f;
	}
	for (unsigned i = 0;i<FIL_NUM*IMG_C*FIL_N;i++) {
		wh[i] = 0.5f;
	}
	cudaMalloc((void**)&od, sizeof(float)*IMG_OUT_NUM*IMG_N*IMG_C*FIL_N);
	cudaMalloc((void**)&id, sizeof(float)*IMG_IN_NUM*IMG_N*IMG_C);
	cudaMalloc((void**)&wd, sizeof(float)*FIL_NUM*FIL_N*IMG_C);
	cudaMemcpy(od, oh, sizeof(float)*IMG_OUT_NUM*IMG_N*IMG_C*FIL_N, cudaMemcpyHostToDevice);
	cudaMemcpy(id, ih, sizeof(float)*IMG_IN_NUM*IMG_N*IMG_C, cudaMemcpyHostToDevice);
	cudaMemcpy(wd, wh, sizeof(float)*FIL_NUM*FIL_N*IMG_C, cudaMemcpyHostToDevice);

	clock_t start, stop;
	start = clock();
	cudaEvent_t st, et;
	cudaEventCreate(&st);
	cudaEventCreate(&et);
	cudaEventRecord(st, 0);
	conv_5x5 << <dim3(IMG_N*IMG_C*FIL_N, 1, 1), dim3(IMG_IN_X, 1, 1) >> >(wd, id, od);
	cudaDeviceSynchronize();
	cudaEventRecord(et, 0);
	float t = 0;
	cudaEventElapsedTime(&t, st, et);
	stop = clock();
	std::cout << (double)(stop - start) / CLOCKS_PER_SEC << std::endl;;
	std::cout << t << std::endl;
	cudaMemcpy(oh, od, sizeof(float)*IMG_OUT_NUM*IMG_N*IMG_C*FIL_N, cudaMemcpyDeviceToHost);

	for (unsigned i = 0;i < IMG_OUT_NUM*IMG_N*IMG_C*FIL_N;i++) {
		if (oh[i] != 12.5f) {
			std::cout << "Damn! went wrong at: " << i << " returned: " << oh[i] << std::endl;
		}
	}

	genCPU(ih, wh, oh2);
	verifyCPU(oh, oh2);
	std::cout << oh[0] << " " << ih[0] << std::endl;
	std::cout << oh[2] << " " << ih[2] << std::endl;
	std::cout << oh[10] << " " << ih[10] << std::endl;
}
