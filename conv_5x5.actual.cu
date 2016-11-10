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


#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>
#include"sm_30_intrinsics.h"

#define IN_X 36
#define IN_Y 36
#define IN_NUM IN_X * IN_Y
#define IN_N 100

#define OUT_X 32
#define OUT_Y 32
#define OUT_NUM OUT_X*OUT_Y

#define FIL_N 3
#define IMG_C 4

#define FIL_X 5
#define FIL_Y 5


__global__ void print() {
	printf("HAHA\n");
}

__global__ void conv_gpu(float *input_d, float *weights_d, float *output_d,
	unsigned in_x, unsigned in_y, unsigned in_num, unsigned in_n,
	unsigned img_c, unsigned fil_n, unsigned fil_num,
	unsigned out_x, unsigned out_y, unsigned out_num, unsigned out_n) {
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	float *input = input_d + (bx / img_c)*(img_c*in_num);
	float *weights = weights_d + (bx % img_c)*(img_c*fil_num);
	float *output = output_d + bx * out_num;

	float w00, w01, w02, w03, w04;
	float w10, w11, w12, w13, w14;
	float w20, w21, w22, w23, w24;
	float w30, w31, w32, w33, w34;
	float w40, w41, w42, w43, w44;

	float var00, var01, var02, var03, var04;
	float var10, var11, var12, var13, var14;
	float var20, var21, var22, var23, var24;
	float var30, var31, var32, var33, var34;
	float var40, var41, var42, var43, var44;

	float tmp = 0;
	for (unsigned j = 0;j < out_y;j++) {
		tmp = 0;
		for (unsigned i = 0;i < img_c;i++) {
			w00 = weights[i + j*fil_num];
			w01 = weights[i + j*fil_num + 1];
			w02 = weights[i + j*fil_num + 2];
			w03 = weights[i + j*fil_num + 3];
			w04 = weights[i + j*fil_num + 4];
			w10 = weights[i + j*fil_num + 5];
			w11 = weights[i + j*fil_num + 6];
			w12 = weights[i + j*fil_num + 7];
			w13 = weights[i + j*fil_num + 8];
			w14 = weights[i + j*fil_num + 9];
			w20 = weights[i + j*fil_num + 10];
			w21 = weights[i + j*fil_num + 11];
			w22 = weights[i + j*fil_num + 12];
			w23 = weights[i + j*fil_num + 13];
			w24 = weights[i + j*fil_num + 14];
			w30 = weights[i + j*fil_num + 15];
			w31 = weights[i + j*fil_num + 16];
			w32 = weights[i + j*fil_num + 17];
			w33 = weights[i + j*fil_num + 18];
			w34 = weights[i + j*fil_num + 19];
			w40 = weights[i + j*fil_num + 20];
			w41 = weights[i + j*fil_num + 21];
			w42 = weights[i + j*fil_num + 22];
			w43 = weights[i + j*fil_num + 23];
			w44 = weights[i + j*fil_num + 24];

			var02 = input[tx + i*in_num + j*in_x];
			var00 = __shfl_up(var02, 2, 32);
			var01 = __shfl_up(var02, 1, 32);
			var03 = __shfl_down(var02, 1, 32);
			var04 = __shfl_down(var02, 2, 32);

			var12 = input[tx + i*in_num + (j + 1)*in_x];
			var10 = __shfl_up(var12, 2, 32);
			var11 = __shfl_up(var12, 1, 32);
			var13 = __shfl_down(var12, 1, 32);
			var14 = __shfl_down(var12, 2, 32);

			var22 = input[tx + i*in_num + (j + 2)*in_x];
			var20 = __shfl_up(var22, 2, 32);
			var21 = __shfl_up(var22, 1, 32);
			var23 = __shfl_down(var22, 1, 32);
			var24 = __shfl_down(var22, 2, 32);

			var32 = input[tx + i*in_num + (j + 3)*in_x];
			var30 = __shfl_up(var32, 2, 32);
			var31 = __shfl_up(var32, 1, 32);
			var33 = __shfl_down(var32, 1, 32);
			var34 = __shfl_down(var32, 2, 32);

			float tmp1, tmp2;

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

			tmp += tmp1 + tmp2;
		}
		if (tx > 1 && tx < in_x - 2) {
			output[tx + j*out_x] = tmp;
		}
	}
}

void conv_cpu(float *input_h, float *weights_h, float *output_h, 
			unsigned in_x, unsigned in_y, unsigned in_num, unsigned in_n, 
			unsigned img_c, unsigned fil_n, unsigned fil_num,
			unsigned out_x, unsigned out_y, unsigned out_num, unsigned out_n) {
	float tmp;
	for (unsigned k = 0;k < in_n;k++) {
		float *input_c = input_h + k * in_num*img_c;
		float *output_c = output_h + k * out_num*fil_n;
		for (unsigned j = 0;j < fil_n; j++) {
			float* weights_c = weights_h + j * fil_num*img_c;
			float* output = output_c + j * out_num;
			for (unsigned i = 0;i < img_c;i++) {
				float *input = input_c + i * in_num;
				float *weights = weights_c + i * fil_num;
				for (unsigned iter_y = 0;iter_y < out_y;iter_y++) {
					for (unsigned iter_x = 0;iter_x < out_x; iter_x++) {

						tmp = 
							weights[0] * input[iter_x+ iter_y*IN_X] + \
							weights[1] * input[iter_x + iter_y*IN_X + 1] + \
							weights[2] * input[iter_x + iter_y*IN_X + 2] + \
							weights[3] * input[iter_x + iter_y*IN_X + 3] + \
							weights[4] * input[iter_x + iter_y*IN_X + 4] + \
							weights[5] * input[iter_x + iter_y*IN_X + IN_X] + \
							weights[6] * input[iter_x + iter_y*IN_X + IN_X + 1] + \
							weights[7] * input[iter_x + iter_y*IN_X + IN_X + 2] + \
							weights[8] * input[iter_x + iter_y*IN_X + IN_X + 3] + \
							weights[9] * input[iter_x + iter_y*IN_X + IN_X + 4] + \
							weights[10]	* input[iter_x + iter_y*IN_X + 2*IN_X] + \
							weights[11] * input[iter_x + iter_y*IN_X + 2*IN_X + 1] + \
							weights[12] * input[iter_x + iter_y*IN_X + 2*IN_X + 2] + \
							weights[13] * input[iter_x + iter_y*IN_X + 2*IN_X + 3] + \
							weights[14] * input[iter_x + iter_y*IN_X + 2*IN_X + 4] + \
							weights[15] * input[iter_x + iter_y*IN_X + 3*IN_X] + \
							weights[16] * input[iter_x + iter_y*IN_X + 3*IN_X+1] + \
							weights[17] * input[iter_x + iter_y*IN_X + 3*IN_X+2] + \
							weights[18] * input[iter_x + iter_y*IN_X + 3*IN_X+3] + \
							weights[19] * input[iter_x + iter_y*IN_X + 3*IN_X+4] + \
							weights[20] * input[iter_x + iter_y*IN_X + 4*IN_X] + \
							weights[21] * input[iter_x + iter_y*IN_X + 4*IN_X + 1] + \
							weights[22] * input[iter_x + iter_y*IN_X + 4*IN_X + 2] + \
							weights[23] * input[iter_x + iter_y*IN_X + 4*IN_X + 3] + \
							weights[24] * input[iter_x + iter_y*IN_X + 4*IN_X + 4];

						output[iter_x + iter_y * out_x] = tmp + output[iter_x + iter_y * out_x];
					}
				}
			}
		}
	 }
}

int main() {
	float *input_h = new float[IN_NUM * IMG_C * IN_N];
	float *fil_h = new float[25 * FIL_N*IMG_C];
	float *output_h = new float[OUT_NUM*FIL_N*IN_N];

	for (unsigned k = 0;k < IN_N;k++) {
		for (unsigned j = 0;j < IMG_C;j++) {
			for (unsigned i = 0;i < IN_NUM;i++) {
				input_h[i+j*IN_NUM+k*IN_NUM*IMG_C] = 1.0f + j*1.0f;
			}
		}
	}

	for (unsigned i = 0;i < 25 * FIL_N*IMG_C;i++) {
		fil_h[i] = 1.0f;
	}
	for (unsigned i = 0;i < OUT_NUM*FIL_N*IN_N;i++) {
		output_h[i] = 0.0f;
	}

	conv_cpu(input_h, fil_h, output_h,
		IN_X, IN_Y, IN_NUM, IN_N, IMG_C, FIL_N, 25,
		OUT_X, OUT_Y, OUT_NUM, IN_N);


	for (unsigned i = 0;i < OUT_NUM*FIL_N;i++) {
		if (output_h[i] != 250.0f) {
			std::cout << "Bad output at: "<<i<< " " <<output_h[i] << std::endl;
			return;
		}
	}

	float *input_d, *weights_d, *output_d;
	cudaMalloc((void**)&input_d, sizeof(float)*IN_NUM*IMG_C*IN_N);
	cudaMalloc((void**)&weights_d, sizeof(float)*FIL_N*IMG_C * 25);
	cudaMalloc((void**)&output_d, sizeof(float)*OUT_NUM*FIL_N*IN_N);

	cudaMemcpy(input_d, input_h, sizeof(float)*IN_NUM*IMG_C*IN_N, cudaMemcpyHostToDevice);
	cudaMemcpy(weights_d, fil_h, sizeof(float) * 25 * FIL_N*IMG_C, cudaMemcpyHostToDevice);
	cudaMemcpy(output_d, output_h, sizeof(float)*OUT_NUM*FIL_N*IN_N, cudaMemcpyHostToDevice);

	print << <dim3(1, 1, 1), dim3(1, 1, 1) >> > ();

	conv_gpu<<<dim3(IN_N*FIL_N,1,1), dim3(IN_X,1,1)>>>(input_d, weights_d, output_d,
		IN_X, IN_Y, IN_NUM, IN_N, IMG_C, FIL_N, 25,
		OUT_X, OUT_Y, OUT_NUM, IN_N);

	cudaMemcpy(output_h, output_d, sizeof(float)*OUT_NUM*FIL_N*IN_N, cudaMemcpyDeviceToHost);
	for (unsigned i = 0;i < OUT_NUM*FIL_N;i++) {
		if (output_h[i] != 250.0f) {
			std::cout << "Bad output at: " << i << " " << output_h[i] << std::endl;
			return;
		}
	}
}
