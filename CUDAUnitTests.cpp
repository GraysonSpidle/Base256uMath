#ifdef __NVCC__
#include "CUDAUnitTests.h"
#include "Base256uMath.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define KERNEL_CALL(func_name, code_ptr) func_name<<<1,1>>>(code_ptr)
#define KERNEL_CALL2(func_name, code_ptr, ptr1) func_name<<<1,1>>>(code_ptr, ptr1)
#include <cassert>
#include <iostream>

// ===================================================================================

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

// ===================================================================================

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
__global__
void is_zero_big_ideal_case_kernel(int* code) {
	*code = 0;
	unsigned char num[20];
	memset(num, 0, 20);
	if (!Base256uMath::is_zero(num, sizeof(num))) {
		*code = 1;
	}
}
__global__
void is_zero_not_zero_kernel(int* code) {
	*code = 0;
	std::size_t num = 1 << 17;
	if (Base256uMath::is_zero(&num, sizeof(num))) {
		*code = 1;
	}
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
__global__
void is_zero_src_n_zero_kernel(int* code) {
	// if src_n is zero, then it is assumed to be zero.

	*code = 0;
	unsigned int num = 1337420;
	if (!Base256uMath::is_zero(&num, 0)) {
		*code = 1;
	}
}

void Base256uMathTests::CUDA::is_zero::ideal_case() {
	int code = -1;
	int* d_code;
	auto err = cudaMalloc(&d_code, sizeof(int));
	if (err != cudaSuccess) {
		std::cout << "Error in cudaMalloc: " << err << std::endl;
		assert(err == cudaSuccess);
	}

	KERNEL_CALL(is_zero_ideal_case_kernel, d_code);
	err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cout << "Error in cudaMemcpy: " << err << std::endl;
		assert(err == cudaSuccess);
	}
	cudaFree(d_code);

	assert(code == 0);
}
void Base256uMathTests::CUDA::is_zero::big_ideal_case() {
	int code = -1;
	int* d_code;
	auto err = cudaMalloc(&d_code, sizeof(int));
	if (err != cudaSuccess) {
		std::cout << "Error in cudaMalloc: " << err << std::endl;
		assert(err == cudaSuccess);
	}

	KERNEL_CALL(is_zero_big_ideal_case_kernel, d_code);
	err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cout << "Error in cudaMemcpy: " << err << std::endl;
		assert(err == cudaSuccess);
	}
	cudaFree(d_code);

	assert(code == 0);
}
void Base256uMathTests::CUDA::is_zero::not_zero() {
	int code = -1;
	int* d_code;
	auto err = cudaMalloc(&d_code, sizeof(int));
	if (err != cudaSuccess) {
		std::cout << "Error in cudaMalloc: " << err << std::endl;
		assert(err == cudaSuccess);
	}

	KERNEL_CALL(is_zero_not_zero_kernel, d_code);
	err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cout << "Error in cudaMemcpy: " << err << std::endl;
		assert(err == cudaSuccess);
	}
	cudaFree(d_code);

	assert(code == 0);
}
void Base256uMathTests::CUDA::is_zero::big_not_zero() {
	int code = -1;
	int* d_code;
	auto err = cudaMalloc(&d_code, sizeof(int));
	if (err != cudaSuccess) {
		std::cout << "Error in cudaMalloc: " << err << std::endl;
		assert(err == cudaSuccess);
	}

	KERNEL_CALL(is_zero_big_not_zero_kernel, d_code);
	err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cout << "Error in cudaMemcpy: " << err << std::endl;
		assert(err == cudaSuccess);
	}
	cudaFree(d_code);

	assert(code == 0);
}
void Base256uMathTests::CUDA::is_zero::src_n_zero() {
	int code = -1;
	int* d_code;
	auto err = cudaMalloc(&d_code, sizeof(int));
	if (err != cudaSuccess) {
		std::cout << "Error in cudaMalloc: " << err << std::endl;
		assert(err == cudaSuccess);
	}

	KERNEL_CALL(is_zero_src_n_zero_kernel, d_code);
	err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cout << "Error in cudaMemcpy: " << err << std::endl;
		assert(err == cudaSuccess);
	}
	cudaFree(d_code);

	assert(code == 0);
}

// ===================================================================================

// I'll probably regret making this a macro at a later date. I'm just too lazy.
#define compare_test_macro(kernel_func)\
int code = -1; \
int* d_code; \
int* d_cmp; \
auto err = cudaMalloc(&d_code, sizeof(int)); \
if (err != cudaSuccess) { \
	std::cout << "CUDA malloc error: " << err << std::endl; \
	assert(err == cudaSuccess); \
} \
err = cudaMalloc(&d_cmp, sizeof(int)); \
if (err != cudaSuccess) { \
	std::cout << "CUDA malloc error: " << err << std::endl; \
	assert(err == cudaSuccess); \
} \
int cmp = -1; \
KERNEL_CALL2(kernel_func, d_code, d_cmp); \
err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost); \
if (err != cudaSuccess) { \
	std::cout << "CUDA memcpy error: " << err << std::endl; \
	assert(err == cudaSuccess); \
} \
err = cudaMemcpy(&cmp, d_cmp, sizeof(int), cudaMemcpyDeviceToHost); \
if (err != cudaSuccess) { \
	std::cout << "CUDA memcpy error: " << err << std::endl; \
	assert(err == cudaSuccess); \
} \
assert(code == 0); \
cudaFree(d_code); \
cudaFree(d_cmp)

// I'm sorry for your eyes

void Base256uMathTests::CUDA::compare::test() {
	ideal_case_equal();
	ideal_case_greater();
	ideal_case_less();

	big_ideal_case_equal();
	big_ideal_case_greater();
	big_ideal_case_less();

	l_bigger_equal();
	l_smaller_equal();
	big_l_bigger_equal();
	big_l_smaller_equal();

	l_bigger_greater();
	l_smaller_greater();
	big_l_bigger_greater();
	big_l_smaller_greater();

	l_bigger_less();
	l_smaller_less();
	big_l_bigger_less();
	big_l_smaller_less();

	left_n_zero();
	right_n_zero();
}
__global__
void ideal_case_equal_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned int l = 156;
	unsigned int r = 156;
	*cmp = Base256uMath::compare(&l, sizeof(l), &r, sizeof(r));
	if (*cmp != 0) {
		*code = 1;
	}
}
__global__
void ideal_case_greater_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned int l = 420,
		r = 69;
	*cmp = Base256uMath::compare(&l, sizeof(l), &r, sizeof(r));
	if (*cmp <= 0) {
		*code = 1;
	}
}
__global__
void ideal_case_less_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned int l = 69,
		r = 420;
	*cmp = Base256uMath::compare(&l, sizeof(l), &r, sizeof(r));
	if (*cmp >= 0) {
		*code = 1;
	}
}
__global__
void big_ideal_case_equal_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned char l[] = { 183, 79, 180, 87, 57, 45, 214, 45, 189 };
	unsigned char r[] = { 183, 79, 180, 87, 57, 45, 214, 45, 189 };
	*cmp = Base256uMath::compare(l, sizeof(l), r, sizeof(r));
	if (*cmp != 0) {
		*code = 1;
	}
}
__global__
void big_ideal_case_greater_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned char l[] = { 186, 153, 248, 144, 124, 225, 100, 21, 186 };
	unsigned char r[] = { 125, 225, 204, 133, 182, 137, 171, 180, 105 };
	*cmp = Base256uMath::compare(l, sizeof(l), r, sizeof(r));
	if (*cmp <= 0) {
		*code = 1;
	}
}
__global__
void big_ideal_case_less_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned char l[] = { 15, 100, 121, 37, 114, 241, 99, 246, 155 };
	unsigned char r[] = { 97, 197, 235, 80, 143, 160, 4, 88, 188 };
	*cmp = Base256uMath::compare(l, sizeof(l), r, sizeof(r));
	if (*cmp >= 0) {
		*code = 1;
	}
}
__global__
void l_bigger_equal_kernel(int* code, int* cmp) {
	*code = 0;
	std::size_t l = 8008135;
	unsigned int r = 8008135;
	*cmp = Base256uMath::compare(&l, sizeof(l), &r, sizeof(r));
	if (*cmp != 0) {
		*code = 1;
	}
}
__global__
void l_smaller_equal_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned int l = 8008135;
	std::size_t r = 8008135;
	*cmp = Base256uMath::compare(&l, sizeof(l), &r, sizeof(r));
	if (*cmp != 0) {
		*code = 1;
	}
}
__global__
void big_l_bigger_equal_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned char l[] = { 202, 14, 146, 155, 72, 7, 240, 198, 40, 0, 0 };
	unsigned char r[] = { 202, 14, 146, 155, 72, 7, 240, 198, 40 };
	*cmp = Base256uMath::compare(l, sizeof(l), r, sizeof(r));
	if (*cmp != 0) {
		*code = 1;
	}
}
__global__
void big_l_smaller_equal_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned char l[] = { 252, 95, 221, 19, 91, 22, 144, 72, 99 };
	unsigned char r[] = { 252, 95, 221, 19, 91, 22, 144, 72, 99, 0, 0 };
	*cmp = Base256uMath::compare(l, sizeof(l), r, sizeof(r));
	if (*cmp != 0) {
		*code = 1;
	}

}
__global__
void l_bigger_greater_kernel(int* code, int* cmp) {
	*code = 0;
	std::size_t l = 144;
	unsigned int r = 25;
	*cmp = Base256uMath::compare(&l, sizeof(l), &r, sizeof(r));
	if (*cmp <= 0) {
		*code = 1;
	}
}
__global__
void l_smaller_greater_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned int l = 1026;
	std::size_t r = 55;
	*cmp = Base256uMath::compare(&l, sizeof(l), &r, sizeof(r));
	if (*cmp <= 0) {
		*code = 1;
	}
}
__global__
void big_l_bigger_greater_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned char l[] = { 147, 199, 111, 216, 79, 139, 236, 53, 116, 0, 0 };
	unsigned char r[] = { 142, 99, 1, 230, 35, 170, 69, 133, 22 };
	*cmp = Base256uMath::compare(l, sizeof(l), r, sizeof(r));
	if (*cmp <= 0) {
		*code = 1;
	}
}
__global__
void big_l_smaller_greater_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned char l[] = { 245, 206, 105, 71, 234, 204, 105, 6, 220 };
	unsigned char r[] = { 172, 253, 57, 29, 149, 255, 208, 108, 3, 0, 0 };
	*cmp = Base256uMath::compare(
		l, sizeof(l),
		r, sizeof(r)
	);
	if (*cmp <= 0) {
		*code = 1;
	}
}
__global__
void l_bigger_less_kernel(int* code, int* cmp) {
	*code = 0;
	std::size_t l = 55;
	unsigned int r = 98;
	*cmp = Base256uMath::compare(
		&l, sizeof(l),
		&r, sizeof(r)
	);
	if (*cmp >= 0) {
		*code = 1;
	}
}
__global__
void l_smaller_less_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned int l = 18;
	std::size_t r = 2173;
	*cmp = Base256uMath::compare(
		&l, sizeof(l),
		&r, sizeof(r)
	);
	if (*cmp >= 0) {
		*code = 1;
	}
}
__global__
void big_l_bigger_less_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned char l[] = { 170, 30, 170, 121, 65, 171, 74, 245, 197, 0, 0 };
	unsigned char r[] = { 172, 253, 57, 29, 149, 255, 208, 108, 220 };
	*cmp = Base256uMath::compare(
		l, sizeof(l),
		r, sizeof(r)
	);
	if (*cmp >= 0) {
		*code = 1;
	}
}
__global__
void big_l_smaller_less_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned char l[] = { 8, 14, 171, 56, 247, 85, 145, 105, 219 };
	unsigned char r[] = { 35, 47, 187, 90, 199, 73, 141, 94, 241, 0, 0 };
	*cmp = Base256uMath::compare(
		l, sizeof(l),
		r, sizeof(r)
	);
	if (*cmp >= 0) {
		*code = 1;
	}
}
__global__
void left_n_zero_kernel(int* code, int* cmp) {
	*code = 0;
	std::size_t left = 5839010;
	std::size_t right = 199931;
	
	*cmp = Base256uMath::compare(&left, 0, &right, sizeof(right));
	if (*cmp >= 0) {
		*code = 1;
		return;
	}
	*cmp = Base256uMath::compare(&right, 0, &left, sizeof(left));
	if (*cmp >= 0) {
		*code = 2;
		return;
	}
	*cmp = Base256uMath::compare(&left, 0, &left, sizeof(left));
	if (*cmp >= 0) {
		*code = 3;
		return;
	}
	right = 0;
	*cmp = Base256uMath::compare(&left, 0, &right, sizeof(right));
	if (*cmp != 0) {
		*code = 4;
		return;
	}
}
__global__
void right_n_zero_kernel(int* code, int* cmp) {
	*code = 0;
	std::size_t left = 199931;
	std::size_t right = 5839010;

	*cmp = Base256uMath::compare(&left, 0, &right, sizeof(right));
	if (*cmp >= 0) {
		*code = 1;
		return;
	}
	*cmp = Base256uMath::compare(&right, 0, &left, sizeof(left));
	if (*cmp >= 0) {
		*code = 2;
		return;
	}
	*cmp = Base256uMath::compare(&left, 0, &left, sizeof(left));
	if (*cmp >= 0) {
		*code = 3;
		return;
	}
	right = 0;
	*cmp = Base256uMath::compare(&left, 0, &right, sizeof(right));
	if (*cmp != 0) {
		*code = 4;
		return;
	}
}

void Base256uMathTests::CUDA::compare::ideal_case_equal() {
	compare_test_macro(ideal_case_equal_kernel);
}
void Base256uMathTests::CUDA::compare::ideal_case_greater() {
	compare_test_macro(ideal_case_greater_kernel);
}
void Base256uMathTests::CUDA::compare::ideal_case_less() {
	compare_test_macro(ideal_case_less_kernel);
}
void Base256uMathTests::CUDA::compare::big_ideal_case_equal() {
	compare_test_macro(big_ideal_case_equal_kernel);
}
void Base256uMathTests::CUDA::compare::big_ideal_case_greater() {
	compare_test_macro(big_ideal_case_greater_kernel);
}
void Base256uMathTests::CUDA::compare::big_ideal_case_less() {
	compare_test_macro(big_ideal_case_less_kernel);
}
void Base256uMathTests::CUDA::compare::l_bigger_equal() {
	compare_test_macro(l_bigger_equal_kernel);
}
void Base256uMathTests::CUDA::compare::l_smaller_equal() {
	compare_test_macro(l_smaller_equal_kernel);
}
void Base256uMathTests::CUDA::compare::big_l_bigger_equal() {
	compare_test_macro(big_l_bigger_equal_kernel);
}
void Base256uMathTests::CUDA::compare::big_l_smaller_equal() {
	compare_test_macro(big_l_smaller_equal_kernel);
}
void Base256uMathTests::CUDA::compare::l_bigger_greater() {
	compare_test_macro(l_bigger_greater_kernel);
}
void Base256uMathTests::CUDA::compare::l_smaller_greater() {
	compare_test_macro(l_smaller_greater_kernel);
}
void Base256uMathTests::CUDA::compare::big_l_bigger_greater() {
	compare_test_macro(big_l_bigger_greater_kernel);
}
void Base256uMathTests::CUDA::compare::big_l_smaller_greater() {
	compare_test_macro(big_l_smaller_greater_kernel);
}
void Base256uMathTests::CUDA::compare::l_bigger_less() {
	compare_test_macro(l_bigger_less_kernel);
}
void Base256uMathTests::CUDA::compare::l_smaller_less() {
	compare_test_macro(l_smaller_less_kernel);
}
void Base256uMathTests::CUDA::compare::big_l_bigger_less() {
	compare_test_macro(big_l_bigger_less_kernel);
}
void Base256uMathTests::CUDA::compare::big_l_smaller_less() {
	compare_test_macro(big_l_smaller_less_kernel);
}
void Base256uMathTests::CUDA::compare::left_n_zero() {
	compare_test_macro(left_n_zero_kernel);
}
void Base256uMathTests::CUDA::compare::right_n_zero() {
	compare_test_macro(right_n_zero_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::max::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::min::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::bitwise_and::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::bitwise_or::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::bitwise_xor::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::bitwise_not::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::byte_shift_left::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::byte_shift_right::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::bit_shift_left::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::bit_shift_right::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::increment::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::decrement::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::add::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::subtract::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::log2::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::log256::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::multiply::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::divide::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::divide_no_mod::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::mod::test() {}

#endif