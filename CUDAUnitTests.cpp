#include "CUDAUnitTests.h"
#include "Base256uMath.h"

#ifndef __NVCC__
#define KERNEL_CALL(func_name, code_ptr)
#define cudaMalloc(ptr, size)
#define cudaMemcpy(dst_ptr, src_ptr, size, designation)
#define cudaFree(ptr)
#define cudaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost
#define __global__
#include <cstring>
#else
#define KERNEL_CALL(func_name, code_ptr) func_name<<<1,1>>>(code_ptr)
#endif
#include <cassert>

void Base256uMathTests::CUDA::test_unit_tests() {
	is_zero::test();
	compare::test();
	max::test();
	min::test();
	bitwise_and::test();
	bitwise_or::test();
	bitwise_xor::test();
	bitwise_not::test();
	byte_shift_left::test();
	byte_shift_right::test();
	bit_shift_left::test();
	bit_shift_right::test();
	increment::test();
	decrement::test();
	add::test();
	subtract::test();
	log2::test();
	log256::test();
	multiply::test();
	divide::test();
	divide_no_mod::test();
	mod::test();
}

void Base256uMathTests::CUDA::is_zero::test() {
	ideal_case();
	big_ideal_case();
	not_zero();
	big_not_zero();
	src_n_zero();
}
__global__
void is_zero_ideal_case_kernel(int* code) {
	*code = 0;
	std::size_t num = 0;
	if (!Base256uMath::is_zero(&num, sizeof(num))) {
		*code = 1;
	}
}
void Base256uMathTests::CUDA::is_zero::ideal_case() {
	int code = -1;
	int* d_code;
	cudaMalloc(&d_code, sizeof(int));

	KERNEL_CALL(is_zero_ideal_case_kernel, d_code);
	cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_code);
	
	assert(code == 0);
}
__global__
void is_zero_big_ideal_case_kernel(int* code) {
	*code = 0;
	unsigned char num[20];
	memset(num, 0, 20);
	if (!Base256uMath::is_zero(num, sizeof(num))) {
		*code = 1;
	}
}
void Base256uMathTests::CUDA::is_zero::big_ideal_case() {
	int code = -1;
	int* d_code;
	cudaMalloc(&d_code, sizeof(int));

	KERNEL_CALL(is_zero_big_ideal_case_kernel, d_code);
	cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_code);
	
	assert(code == 0);
}
__global__
void is_zero_not_zero_kernel(int* code) {
	*code = 0;
	std::size_t num = 1 << 17;
	if (Base256uMath::is_zero(&num, sizeof(num))) {
		*code = 1;
	}
}
void Base256uMathTests::CUDA::is_zero::not_zero() {
	int code = -1;
	int* d_code;
	cudaMalloc(&d_code, sizeof(int));

	KERNEL_CALL(is_zero_not_zero_kernel, d_code);
	cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_code);

	assert(code == 0);
}
__global__
void is_zero_big_not_zero_kernel(int* code) {
	*code = 0;
	unsigned char num[20];
	memset(num, 0, sizeof(num));
	num[15] = 8;
	if (Base256uMath::is_zero(num, sizeof(num))) {
		*code = 1;
	}
}
void Base256uMathTests::CUDA::is_zero::big_not_zero() {
	int code = -1;
	int* d_code;
	cudaMalloc(&d_code, sizeof(int));

	KERNEL_CALL(is_zero_big_not_zero_kernel, d_code);
	cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_code);

	assert(code == 0);
}
__global__
void is_zero_src_n_zero_kernel(int* code) {
	// if src_n is zero, then it is assumed to be zero.

	*code = 0;
	unsigned int num = 1337420;
	if (!Base256uMath::is_zero(&num, 0)) {
		*code = 1;
	}
}
void Base256uMathTests::CUDA::is_zero::src_n_zero() {
	int code = -1;
	int* d_code;
	cudaMalloc(&d_code, sizeof(int));

	KERNEL_CALL(is_zero_src_n_zero_kernel, d_code);
	cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_code);

	assert(code == 0);
}

void Base256uMathTests::CUDA::compare::test() {}
void Base256uMathTests::CUDA::max::test() {}
void Base256uMathTests::CUDA::min::test() {}
void Base256uMathTests::CUDA::bitwise_and::test() {}
void Base256uMathTests::CUDA::bitwise_or::test() {}
void Base256uMathTests::CUDA::bitwise_xor::test() {}
void Base256uMathTests::CUDA::bitwise_not::test() {}
void Base256uMathTests::CUDA::byte_shift_left::test() {}
void Base256uMathTests::CUDA::byte_shift_right::test() {}
void Base256uMathTests::CUDA::bit_shift_left::test() {}
void Base256uMathTests::CUDA::bit_shift_right::test() {}
void Base256uMathTests::CUDA::increment::test() {}
void Base256uMathTests::CUDA::decrement::test() {}
void Base256uMathTests::CUDA::add::test() {}
void Base256uMathTests::CUDA::subtract::test() {}
void Base256uMathTests::CUDA::log2::test() {}
void Base256uMathTests::CUDA::log256::test() {}
void Base256uMathTests::CUDA::multiply::test() {}
void Base256uMathTests::CUDA::divide::test() {}
void Base256uMathTests::CUDA::divide_no_mod::test() {}
void Base256uMathTests::CUDA::mod::test() {}
