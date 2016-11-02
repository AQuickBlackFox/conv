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

  float var01 = input[tid];
  float var00 = __shfl_up(var01, 1, 64);
  float var10 = __shfl_down(var01, 1, 64);

  output[tid] = var00 + var01 + var10;

}

int main() {
    float *wh, *ih, *oh;
    float *wd, *id, *od;
    wh = new float[FIL_NUM];
    ih = new float[IMG_NUM];
    oh = new float[IMG_NUM];
    for(unsigned i=0;i<IMG_NUM;i++) {
        ih[i] = 1.0f;
        oh[i] = 0.0f;
    }
    hipMalloc((void**)&od, sizeof(float)*IMG_NUM);
    hipMalloc((void**)&id, sizeof(float)*IMG_NUM);
    hipMalloc((void**)&wd, sizeof(float)*FIL_NUM);
    hipMemcpy(od, oh, sizeof(float)*IMG_NUM, hipMemcpyHostToDevice);
    hipMemcpy(id, ih, sizeof(float)*IMG_NUM, hipMemcpyHostToDevice);
    hipMemcpy(wd, wh, sizeof(float)*FIL_NUM, hipMemcpyHostToDevice);

    hipLaunchKernel(conv_3x3, dim3(IMG_Y,1,1), dim3(IMG_X,1,1), 0, 0, wd, id, od);

    hipMemcpy(oh, od, sizeof(float)*IMG_NUM, hipMemcpyDeviceToHost);

    std::cout<<oh[10]<<" "<<ih[10]<<std::endl;
}
