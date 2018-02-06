#include <cafe_constants.h>
#include <stream_to_streams.h>

__global__ void stream_to_streams(float2 *in, float2 *out)
{
  int id = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
           threadIdx.x;
  out[threadIdx.x * gridDim.x * blockDim.y + blockIdx.x * blockDim.y +
      threadIdx.y] = in[id];
}

__global__ void streams_to_stream(float2 *in, float2 *out)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  out[threadIdx.x * gridDim.x + blockIdx.x] = in[id];
}
