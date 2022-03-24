#ifdef __NVCC__
#include "Base256uMathCUDA.cuh"
#else
#include "Base256uMathCUDA.h"
#endif

/*
__device__
std::size_t bitwise_get_element(
	int blockIdxy,
	int blockIdxx,
	int threadIdxy,
	int threadIdxx,
	int gridDimx,
	int gridDimy,
	int blockDimx,
	int blockDimy
) {
	return (
		(blockIdxx + gridDimy * blockIdxy)
		+ (gridDimx * gridDimy)
		* (blockDimx * threadIdxx + threadIdxy)
		) + (blockDimx * blockDimy) * (gridDimx * gridDimy); //*gpu_thread_id;
}

__global__
int Base256uMath::CUDA::bitwise_and(
	const void* const left,
	std::size_t left_n,
	const void* const right,
	std::size_t right_n,
	void* const dst,
	std::size_t dst_n
) {



}*/
