#ifndef __BASE256UMATHCUDA_H__
#define __BASE256UMATHCUDA_H__

#ifndef __NVCC__
#include "Base256uMath.h"
#define __global__
#define __host__ 
#define __device__
#define __shared__
#define __constant__
#define __managed__
#else
#include "Base256uMath.cuh"
#endif

namespace Base256uMath {


	namespace CUDA {
		// gonna make some kernel variants of some functions

		__global__ int bitwise_and(
			const void* const left,
			std::size_t left_n,
			const void* const right,
			std::size_t right_n,
			void* const dst,
			std::size_t dst_n
		);

		__global__ int bitwise_or(
			const void* const left,
			std::size_t left_n,
			const void* const right,
			std::size_t right_n,
			void* const dst,
			std::size_t dst_n
		);

		__global__ int bitwise_xor(
			const void* const left,
			std::size_t left_n,
			const void* const right,
			std::size_t right_n,
			void* const dst,
			std::size_t dst_n
		);

		__global__ int bitwise_not(
			void* const src,
			std::size_t src_n
		);

	};
};

#endif // __BASE256UMATHCUDA_H__