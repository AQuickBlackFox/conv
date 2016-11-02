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


#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include<iostream>

#define FIL_X 3
#define FIL_Y 3

#define IMG_X 128
#define IMG_Y 128

#define IMG_NUM IMG_X * IMG_Y
#define FIL_NUM FIL_X * FIL_Y

#define IMG_N 10

__global__ void conv_3x3(hipLaunchParm lp, float *weights, float *input, float *output)
{
  int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  float w00 = weights[0];
  float w01 = weights[1];
  float w02 = weights[2];
  float w10 = weights[3];
  float w11 = weights[4];
  float w12 = weights[5];
  float w20 = weights[6];
  float w21 = weights[7];
  float w22 = weights[8];
  for(unsigned i=0;i<IMG_Y;i++){
    if(tid < IMG_X){

      float var01 = input[tid - IMG_X + i*IMG_X];
      float var00 = __shfl_up(var01, 1, 64);
      float var02 = __shfl_down(var01, 1, 64);
      float var11 = input[tid + i*IMG_X];
      float var10 = __shfl_up(var11, 1, 64);
      float var12 = __shfl_down(var11, 1, 64);
      float var21 = input[tid + IMG_X + i*IMG_X];
      float var20 = __shfl_up(var21, 1, 64);
      float var22 = __shfl_down(var21, 1, 64);

      float tmp1 = w00 * var00;
      float tmp2 = w01 * var01;
      tmp1 = w02 * var02 + tmp1;
      tmp2 = w10 * var10 + tmp2;
      tmp1 = w11 * var11 + tmp1;
      tmp2 = w12 * var12 + tmp2;
      tmp1 = w20 * var20 + tmp1;
      tmp2 = w21 * var21 + tmp2;
      tmp1 = w22 * var22 + tmp1;
      output[tid + i*IMG_X] = tmp1 + tmp2;
/*    output[tid] = w00 * var00 + w01 * var01 + w02 * var02 + \
                  w10 * var10 + w11 * var11 + w12 * var12 + \
                  w21 * var20 + w22 * var21 + w22 * var22;
*/
  }
}
}

int main() {
    float *wh, *ih, *oh;
    float *wd, *id, *od;
    wh = new float[FIL_NUM];
    ih = new float[IMG_NUM];
    oh = new float[IMG_NUM];
    for(unsigned i=0;i<IMG_NUM;i++) {
        ih[i] = i * 1.0f;
        oh[i] = 0.0f;
    }
    for(unsigned i=0;i<FIL_NUM;i++) {
        wh[i] = 0.5f;
    }
    hipMalloc((void**)&od, sizeof(float)*IMG_NUM);
    hipMalloc((void**)&id, sizeof(float)*IMG_NUM);
    hipMalloc((void**)&wd, sizeof(float)*FIL_NUM);
    hipMemcpy(od, oh, sizeof(float)*IMG_NUM, hipMemcpyHostToDevice);
    hipMemcpy(id, ih, sizeof(float)*IMG_NUM, hipMemcpyHostToDevice);
    hipMemcpy(wd, wh, sizeof(float)*FIL_NUM, hipMemcpyHostToDevice);

    hipLaunchKernel(conv_3x3, dim3(IMG_Y,1,1), dim3(IMG_X,1,1), 0, 0, wd, id, od);

    hipMemcpy(oh, od, sizeof(float)*IMG_NUM, hipMemcpyDeviceToHost);

    std::cout<<oh[0]<<" "<<ih[0]<<std::endl;
    std::cout<<oh[2]<<" "<<ih[2]<<std::endl;
    std::cout<<oh[10]<<" "<<ih[10]<<std::endl;
}
