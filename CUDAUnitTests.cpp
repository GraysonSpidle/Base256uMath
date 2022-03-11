#ifdef __NVCC__
#include "CUDAUnitTests.h"
#include "Base256uMath.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define KERNEL_CALL(func_name, code_ptr) func_name<<<1,1>>>(code_ptr); cudaDeviceSynchronize()
#define KERNEL_CALL2(func_name, code_ptr, ptr1) func_name<<<1,1>>>(code_ptr, ptr1); cudaDeviceSynchronize()
#define KERNEL_CALL3(func_name, code_ptr, ptr1, ptr2) func_name<<<1,1>>>(code_ptr, ptr1, ptr2); cudaDeviceSynchronize()
#define KERNEL_CALL4(func_name, code_ptr, ptr1, ptr2, ptr3) func_name<<<1,1>>>(code_ptr, ptr1, ptr2, ptr3); cudaDeviceSynchronize()
#include <cassert>
#include <iostream>

#define cudaMemcpy_check_macro(err_code_name) \
if (err_code_name != cudaSuccess) { \
	std::cout << "cuda memcpy error: "; \
	switch (err_code_name) { \
	case cudaErrorInvalidValue: \
		std::cout << "cudaErrorInvalidValue" << std::endl; \
		break; \
	case cudaErrorInvalidMemcpyDirection: \
		std::cout << "the memcpy kind (ie cudaMemcpyDeviceToHost) is not valid" << std::endl; \
		break; \
	default: \
		std::cout << "Unknown error: " << std::to_string(err_code_name) << std::endl; \
	} \
	assert(err_code_name == cudaSuccess); \
}

#define cudaMalloc_check_macro(err_code_name) \
if (err_code_name != cudaSuccess) { \
	std::cout << "cuda malloc error: "; \
	switch (err_code_name) { \
	case cudaErrorInvalidValue: \
		std::cout << "cudaErrorInvalidValue" << std::endl; \
		break; \
	case cudaErrorMemoryAllocation: \
		std::cout << "couldn't allocate enough memory" << std::endl; \
		break; \
	default: \
		std::cout << "Unknown error: " << std::to_string(err_code_name) << std::endl; \
	} \
	assert(err_code_name == cudaSuccess); \
}

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
	cudaMalloc_check_macro(err);
	KERNEL_CALL(is_zero_ideal_case_kernel, d_code);
	err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy_check_macro(err);
	cudaFree(d_code);
	assert(code == 0);
}
void Base256uMathTests::CUDA::is_zero::big_ideal_case() {
	int code = -1;
	int* d_code;
	auto err = cudaMalloc(&d_code, sizeof(int));
	cudaMalloc_check_macro(err);
	KERNEL_CALL(is_zero_big_ideal_case_kernel, d_code);
	err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy_check_macro(err);
	cudaFree(d_code);
	assert(code == 0);
}
void Base256uMathTests::CUDA::is_zero::not_zero() {
	int code = -1;
	int* d_code;
	auto err = cudaMalloc(&d_code, sizeof(int));
	cudaMalloc_check_macro(err);
	KERNEL_CALL(is_zero_not_zero_kernel, d_code);
	err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy_check_macro(err);
	cudaFree(d_code);
	assert(code == 0);
}
void Base256uMathTests::CUDA::is_zero::big_not_zero() {
	int code = -1;
	int* d_code;
	auto err = cudaMalloc(&d_code, sizeof(int));
	cudaMalloc_check_macro(err);
	KERNEL_CALL(is_zero_big_not_zero_kernel, d_code);
	err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy_check_macro(err);
	cudaFree(d_code);
	assert(code == 0);
}
void Base256uMathTests::CUDA::is_zero::src_n_zero() {
	int code = -1;
	int* d_code;
	auto err = cudaMalloc(&d_code, sizeof(int));
	cudaMalloc_check_macro(err);
	KERNEL_CALL(is_zero_src_n_zero_kernel, d_code);
	err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy_check_macro(err);
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
cudaMalloc_check_macro(err); \
err = cudaMalloc(&d_cmp, sizeof(int)); \
cudaMalloc_check_macro(err); \
int cmp = -1; \
KERNEL_CALL2(kernel_func, d_code, d_cmp); \
err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost); \
cudaMemcpy_check_macro(err); \
err = cudaMemcpy(&cmp, d_cmp, sizeof(int), cudaMemcpyDeviceToHost); \
cudaMemcpy_check_macro(err); \
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
void compare_ideal_case_equal_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned int l = 156;
	unsigned int r = 156;
	*cmp = Base256uMath::compare(&l, sizeof(l), &r, sizeof(r));
	if (*cmp != 0) {
		*code = 1;
	}
}
__global__
void compare_ideal_case_greater_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned int l = 420,
		r = 69;
	*cmp = Base256uMath::compare(&l, sizeof(l), &r, sizeof(r));
	if (*cmp <= 0) {
		*code = 1;
	}
}
__global__
void compare_ideal_case_less_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned int l = 69,
		r = 420;
	*cmp = Base256uMath::compare(&l, sizeof(l), &r, sizeof(r));
	if (*cmp >= 0) {
		*code = 1;
	}
}
__global__
void compare_big_ideal_case_equal_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned char l[] = { 183, 79, 180, 87, 57, 45, 214, 45, 189 };
	unsigned char r[] = { 183, 79, 180, 87, 57, 45, 214, 45, 189 };
	*cmp = Base256uMath::compare(l, sizeof(l), r, sizeof(r));
	if (*cmp != 0) {
		*code = 1;
	}
}
__global__
void compare_big_ideal_case_greater_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned char l[] = { 186, 153, 248, 144, 124, 225, 100, 21, 186 };
	unsigned char r[] = { 125, 225, 204, 133, 182, 137, 171, 180, 105 };
	*cmp = Base256uMath::compare(l, sizeof(l), r, sizeof(r));
	if (*cmp <= 0) {
		*code = 1;
	}
}
__global__
void compare_big_ideal_case_less_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned char l[] = { 15, 100, 121, 37, 114, 241, 99, 246, 155 };
	unsigned char r[] = { 97, 197, 235, 80, 143, 160, 4, 88, 188 };
	*cmp = Base256uMath::compare(l, sizeof(l), r, sizeof(r));
	if (*cmp >= 0) {
		*code = 1;
	}
}
__global__
void compare_l_bigger_equal_kernel(int* code, int* cmp) {
	*code = 0;
	std::size_t l = 8008135;
	unsigned int r = 8008135;
	*cmp = Base256uMath::compare(&l, sizeof(l), &r, sizeof(r));
	if (*cmp != 0) {
		*code = 1;
	}
}
__global__
void compare_l_smaller_equal_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned int l = 8008135;
	std::size_t r = 8008135;
	*cmp = Base256uMath::compare(&l, sizeof(l), &r, sizeof(r));
	if (*cmp != 0) {
		*code = 1;
	}
}
__global__
void compare_big_l_bigger_equal_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned char l[] = { 202, 14, 146, 155, 72, 7, 240, 198, 40, 0, 0 };
	unsigned char r[] = { 202, 14, 146, 155, 72, 7, 240, 198, 40 };
	*cmp = Base256uMath::compare(l, sizeof(l), r, sizeof(r));
	if (*cmp != 0) {
		*code = 1;
	}
}
__global__
void compare_big_l_smaller_equal_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned char l[] = { 252, 95, 221, 19, 91, 22, 144, 72, 99 };
	unsigned char r[] = { 252, 95, 221, 19, 91, 22, 144, 72, 99, 0, 0 };
	*cmp = Base256uMath::compare(l, sizeof(l), r, sizeof(r));
	if (*cmp != 0) {
		*code = 1;
	}

}
__global__
void compare_l_bigger_greater_kernel(int* code, int* cmp) {
	*code = 0;
	std::size_t l = 144;
	unsigned int r = 25;
	*cmp = Base256uMath::compare(&l, sizeof(l), &r, sizeof(r));
	if (*cmp <= 0) {
		*code = 1;
	}
}
__global__
void compare_l_smaller_greater_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned int l = 1026;
	std::size_t r = 55;
	*cmp = Base256uMath::compare(&l, sizeof(l), &r, sizeof(r));
	if (*cmp <= 0) {
		*code = 1;
	}
}
__global__
void compare_big_l_bigger_greater_kernel(int* code, int* cmp) {
	*code = 0;
	unsigned char l[] = { 147, 199, 111, 216, 79, 139, 236, 53, 116, 0, 0 };
	unsigned char r[] = { 142, 99, 1, 230, 35, 170, 69, 133, 22 };
	*cmp = Base256uMath::compare(l, sizeof(l), r, sizeof(r));
	if (*cmp <= 0) {
		*code = 1;
	}
}
__global__
void compare_big_l_smaller_greater_kernel(int* code, int* cmp) {
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
void compare_l_bigger_less_kernel(int* code, int* cmp) {
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
void compare_l_smaller_less_kernel(int* code, int* cmp) {
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
void compare_big_l_bigger_less_kernel(int* code, int* cmp) {
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
void compare_big_l_smaller_less_kernel(int* code, int* cmp) {
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
void compare_left_n_zero_kernel(int* code, int* cmp) {
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
void compare_right_n_zero_kernel(int* code, int* cmp) {
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
	compare_test_macro(compare_ideal_case_equal_kernel);
}
void Base256uMathTests::CUDA::compare::ideal_case_greater() {
	compare_test_macro(compare_ideal_case_greater_kernel);
}
void Base256uMathTests::CUDA::compare::ideal_case_less() {
	compare_test_macro(compare_ideal_case_less_kernel);
}
void Base256uMathTests::CUDA::compare::big_ideal_case_equal() {
	compare_test_macro(compare_big_ideal_case_equal_kernel);
}
void Base256uMathTests::CUDA::compare::big_ideal_case_greater() {
	compare_test_macro(compare_big_ideal_case_greater_kernel);
}
void Base256uMathTests::CUDA::compare::big_ideal_case_less() {
	compare_test_macro(compare_big_ideal_case_less_kernel);
}
void Base256uMathTests::CUDA::compare::l_bigger_equal() {
	compare_test_macro(compare_l_bigger_equal_kernel);
}
void Base256uMathTests::CUDA::compare::l_smaller_equal() {
	compare_test_macro(compare_l_smaller_equal_kernel);
}
void Base256uMathTests::CUDA::compare::big_l_bigger_equal() {
	compare_test_macro(compare_big_l_bigger_equal_kernel);
}
void Base256uMathTests::CUDA::compare::big_l_smaller_equal() {
	compare_test_macro(compare_big_l_smaller_equal_kernel);
}
void Base256uMathTests::CUDA::compare::l_bigger_greater() {
	compare_test_macro(compare_l_bigger_greater_kernel);
}
void Base256uMathTests::CUDA::compare::l_smaller_greater() {
	compare_test_macro(compare_l_smaller_greater_kernel);
}
void Base256uMathTests::CUDA::compare::big_l_bigger_greater() {
	compare_test_macro(compare_big_l_bigger_greater_kernel);
}
void Base256uMathTests::CUDA::compare::big_l_smaller_greater() {
	compare_test_macro(compare_big_l_smaller_greater_kernel);
}
void Base256uMathTests::CUDA::compare::l_bigger_less() {
	compare_test_macro(compare_l_bigger_less_kernel);
}
void Base256uMathTests::CUDA::compare::l_smaller_less() {
	compare_test_macro(compare_l_smaller_less_kernel);
}
void Base256uMathTests::CUDA::compare::big_l_bigger_less() {
	compare_test_macro(compare_big_l_bigger_less_kernel);
}
void Base256uMathTests::CUDA::compare::big_l_smaller_less() {
	compare_test_macro(compare_big_l_smaller_less_kernel);
}
void Base256uMathTests::CUDA::compare::left_n_zero() {
	compare_test_macro(compare_left_n_zero_kernel);
}
void Base256uMathTests::CUDA::compare::right_n_zero() {
	compare_test_macro(compare_right_n_zero_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::max::test() {
	ideal_case_left();
	ideal_case_right();
	big_ideal_case_left();
	big_ideal_case_right();

	left_bigger_left();
	left_smaller_left();
	left_bigger_right();
	left_smaller_right();

	big_left_bigger_left();
	big_left_smaller_left();
	big_left_bigger_right();
	big_left_smaller_right();
}

__global__
void max_ideal_case_left_kernel(int* code) {
	*code = 0;
	unsigned int left = 500,
		right = 320;
	auto ptr = reinterpret_cast<const unsigned int*>(
		Base256uMath::max(&left, sizeof(left), &right, sizeof(right))
		);
	if (ptr != &left) {
		*code = 1;
	}
}
__global__
void max_ideal_case_right_kernel(int* code) {
	*code = 0;
	unsigned int left = 13,
		right = 1337;
	auto ptr = reinterpret_cast<const unsigned int*>(Base256uMath::max(
		&left, sizeof(left), &right, sizeof(right))
		);
	if (ptr != &right) {
		*code = 1;
	}
}
__global__
void max_big_ideal_case_left_kernel(int* code) {
	*code = 0;
	const unsigned char left[] = { 156, 247, 183, 55, 60, 119, 65, 37, 175 };
	const unsigned char right[] = { 239, 55, 236, 133, 175, 168, 253, 237, 57 };
	auto ptr = reinterpret_cast<const unsigned char*>(
		Base256uMath::max(left, sizeof(left), right, sizeof(right))
		);
	if (ptr != left) {
		*code = 1;
	}
}
__global__
void max_big_ideal_case_right_kernel(int* code) {
	*code = 0;
	const unsigned char left[] = { 220, 165, 118, 130, 251, 82, 50, 81, 178 };
	const unsigned char right[] = { 177, 246, 145, 224, 167, 216, 180, 173, 186 };
	auto ptr = reinterpret_cast<const unsigned char*>(
		Base256uMath::max(left, sizeof(left), right, sizeof(right))
		);
	if (ptr != right) {
		*code = 1;
	}
}
__global__
void max_left_bigger_left_kernel(int* code) {
	*code = 0;
	std::size_t left = 69696969696;
	unsigned int right = 21360;
	auto ptr = Base256uMath::max(&left, sizeof(left), &right, sizeof(right));
	if (ptr != &left) {
		*code = 1;
	}
}
__global__
void max_left_smaller_left_kernel(int* code) {
	*code = 0;
	unsigned int left = 35459;
	std::size_t right = 3819;
	auto ptr = Base256uMath::max(&left, sizeof(left), &right, sizeof(right));
	if (ptr != &left) {
		*code = 1;
	}
}
__global__
void max_left_bigger_right_kernel(int* code) {
	std::size_t left = 13264;
	unsigned int right = 19894;
	auto ptr = Base256uMath::max(&left, sizeof(left), &right, sizeof(right));
	if (ptr != &right) {
		*code = 1;
	}
}
__global__
void max_left_smaller_right_kernel(int* code) {
	*code = 0;
	unsigned int left = 4548;
	std::size_t right = 30923;
	auto ptr = Base256uMath::max(&left, sizeof(left), &right, sizeof(right));
	if (ptr != &right) {
		*code = 1;
	}
}
__global__
void max_big_left_bigger_left_kernel(int* code) {
	*code = 0;
	unsigned char left[] = { 65, 128, 36, 71, 126, 195, 52, 194, 176, 0, 0 };
	unsigned char right[] = { 108, 128, 45, 116, 237, 77, 15, 158, 89 };
	auto ptr = Base256uMath::max(left, sizeof(left), right, sizeof(right));
	if (ptr != left) {
		*code = 1;
	}
}
__global__
void max_big_left_smaller_left_kernel(int* code) {
	*code = 0;
	unsigned char left[] = { 180, 67, 35, 216, 106, 3, 28, 187, 155 };
	unsigned char right[] = { 149, 169, 152, 146, 14, 240, 4, 241, 95, 0, 0 };
	auto ptr = Base256uMath::max(left, sizeof(left), right, sizeof(right));
	if (ptr != left) {
		*code = 1;
	}
}
__global__
void max_big_left_bigger_right_kernel(int* code) {
	*code = 0;
	unsigned char left[] = { 216, 116, 109, 138, 103, 52, 127, 58, 65, 0, 0 };
	unsigned char right[] = { 119, 78, 117, 53, 63, 130, 146, 168, 219 };
	auto ptr = Base256uMath::max(left, sizeof(left), right, sizeof(right));
	if (ptr != right) {
		*code = 1;
	}
}
__global__
void max_big_left_smaller_right_kernel(int* code) {
	*code = 0;
	unsigned char left[] = { 164, 254, 202, 93, 102, 155, 170, 243, 234 };
	unsigned char right[] = { 163, 24, 36, 50, 205, 211, 146, 12, 238, 0, 0 };
	auto ptr = Base256uMath::max(left, sizeof(left), right, sizeof(right));
	if (ptr != right) {
		*code = 1;
	}
}

// Yes another macro, again, lazy.
#define max_test_macro(kernel_name) \
int code = -1; \
int* d_code; \
auto err = cudaMalloc(&d_code, sizeof(int)); \
cudaMalloc_check_macro(err); \
KERNEL_CALL(kernel_name, d_code); \
err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost); \
cudaMemcpy_check_macro(err); \
assert(code == 0); \
cudaFree(d_code);


void Base256uMathTests::CUDA::max::ideal_case_left() {
	max_test_macro(max_ideal_case_left_kernel);
}
void Base256uMathTests::CUDA::max::ideal_case_right() {
	max_test_macro(max_ideal_case_right_kernel);
}
void Base256uMathTests::CUDA::max::big_ideal_case_left() {
	max_test_macro(max_big_ideal_case_left_kernel);
}
void Base256uMathTests::CUDA::max::big_ideal_case_right() {
	max_test_macro(max_big_ideal_case_right_kernel);
}
void Base256uMathTests::CUDA::max::left_bigger_left() {
	max_test_macro(max_left_bigger_left_kernel);
}
void Base256uMathTests::CUDA::max::left_smaller_left() {
	max_test_macro(max_left_smaller_left_kernel);
}
void Base256uMathTests::CUDA::max::left_bigger_right() {
	max_test_macro(max_left_bigger_right_kernel);
}
void Base256uMathTests::CUDA::max::left_smaller_right() {
	max_test_macro(max_left_smaller_right_kernel);
}
void Base256uMathTests::CUDA::max::big_left_bigger_left() {
	max_test_macro(max_big_left_bigger_left_kernel);
}
void Base256uMathTests::CUDA::max::big_left_smaller_left() {
	max_test_macro(max_big_left_smaller_left_kernel);
}
void Base256uMathTests::CUDA::max::big_left_bigger_right() {
	max_test_macro(max_big_left_bigger_right_kernel);
}
void Base256uMathTests::CUDA::max::big_left_smaller_right() {
	max_test_macro(max_big_left_smaller_right_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::min::test() {
	ideal_case_left();
	ideal_case_right();
	big_ideal_case_left();
	big_ideal_case_right();

	left_bigger_left();
	left_smaller_left();
	left_bigger_right();
	left_smaller_right();

	big_left_bigger_left();
	big_left_smaller_left();
	big_left_bigger_right();
	big_left_smaller_right();
}

__global__
void min_ideal_case_left_kernel(int* code) {
	*code = 0;
	unsigned int left = 8969;
	unsigned int right = 11219;
	auto ptr = Base256uMath::min(&left, sizeof(left), &right, sizeof(right));
	if (ptr != &left) {
		*code = 1;
	}
}
__global__
void min_ideal_case_right_kernel(int* code) {
	*code = 0;
	unsigned int left = 34063;
	unsigned int right = 16197;
	auto ptr = Base256uMath::min(&left, sizeof(left), &right, sizeof(right));
	if (ptr != &right) {
		*code = 1;
	}
}
__global__
void min_big_ideal_case_left_kernel(int* code) {
	*code = 0;
	unsigned char left[] = { 29, 236, 239, 48, 243, 6, 109, 228, 82 };
	unsigned char right[] = { 153, 65, 158, 142, 123, 85, 44, 225, 162 };
	auto ptr = Base256uMath::min(left, sizeof(left), right, sizeof(right));
	if (ptr != left) {
		*code = 1;
	}
}
__global__
void min_big_ideal_case_right_kernel(int* code) {
	*code = 0;
	unsigned char left[] = { 83, 167, 5, 136, 162, 1, 249, 140, 156 };
	unsigned char right[] = { 102, 251, 89, 166, 213, 231, 56, 54, 20 };
	auto ptr = Base256uMath::min(left, sizeof(left), right, sizeof(right));
	if (ptr != right) {
		*code = 1;
	}
}
__global__
void min_left_bigger_left_kernel(int* code) {
	*code = 0;
	std::size_t left = 28606;
	unsigned int right = 34288;
	auto ptr = Base256uMath::min(&left, sizeof(left), &right, sizeof(right));
	if (ptr != &left) {
		*code = 1;
	}
}
__global__
void min_left_smaller_left_kernel(int* code) {
	*code = 0;
	unsigned int left = 43810;
	std::size_t right = 47275;
	auto ptr = Base256uMath::min(&left, sizeof(left), &right, sizeof(right));
	if (ptr != &left) {
		*code = 1;
	}
}
__global__
void min_left_bigger_right_kernel(int* code) {
	*code = 0;
	std::size_t left = 49660;
	unsigned int right = 7010;
	auto ptr = Base256uMath::min(&left, sizeof(left), &right, sizeof(right));
	if (ptr != &right) {
		*code = 1;
	}
}
__global__
void min_left_smaller_right_kernel(int* code) {
	*code = 0;
	unsigned int left = 63729;
	std::size_t right = 46223;
	auto ptr = Base256uMath::min(&left, sizeof(left), &right, sizeof(right));
	if (ptr != &right) {
		*code = 1;
	}
}
__global__
void min_big_left_bigger_left_kernel(int* code) {
	*code = 0;
	unsigned char left[] = { 123, 68, 215, 46, 186, 97, 149, 27, 149, 0, 0 };
	unsigned char right[] = { 120, 114, 238, 213, 227, 7, 228, 47, 159 };
	auto ptr = Base256uMath::min(left, sizeof(left), right, sizeof(right));
	if (ptr != left) {
		*code = 1;
	}
}
__global__
void min_big_left_smaller_left_kernel(int* code) {
	*code = 0;
	unsigned char left[] = { 253, 37, 145, 49, 69, 19, 171, 189, 27 };
	unsigned char right[] = { 67, 228, 217, 39, 59, 24, 249, 194, 55, 0, 0 };
	auto ptr = Base256uMath::min(left, sizeof(left), right, sizeof(right));
	if (ptr != left) {
		*code = 1;
	}
}
__global__
void min_big_left_bigger_right_kernel(int* code) {
	*code = 0;
	unsigned char left[] = { 49, 27, 111, 206, 109, 89, 42, 220, 227, 0, 0 };
	unsigned char right[] = { 93, 22, 212, 80, 84, 184, 37, 130, 194 };
	auto ptr = Base256uMath::min(left, sizeof(left), right, sizeof(right));
	if (ptr != right) {
		*code = 1;
	}
}
__global__
void min_big_left_smaller_right_kernel(int* code) {
	*code = 0;
	unsigned char left[] = { 87, 220, 65, 201, 73, 117, 94, 29, 173 };
	unsigned char right[] = { 91, 247, 82, 39, 62, 19, 90, 174, 118, 0, 0 };
	auto ptr = Base256uMath::min(left, sizeof(left), right, sizeof(right));
	if (ptr != right) {
		*code = 1;
	}
}

void Base256uMathTests::CUDA::min::ideal_case_left() {
	max_test_macro(min_ideal_case_left_kernel);
}
void Base256uMathTests::CUDA::min::ideal_case_right() {
	max_test_macro(min_ideal_case_right_kernel);
}
void Base256uMathTests::CUDA::min::big_ideal_case_left() {
	max_test_macro(min_big_ideal_case_left_kernel);
}
void Base256uMathTests::CUDA::min::big_ideal_case_right() {
	max_test_macro(min_big_ideal_case_right_kernel);
}
void Base256uMathTests::CUDA::min::left_bigger_left() {
	max_test_macro(min_left_bigger_left_kernel);
}
void Base256uMathTests::CUDA::min::left_smaller_left() {
	max_test_macro(min_left_smaller_left_kernel);
}
void Base256uMathTests::CUDA::min::left_bigger_right() {
	max_test_macro(min_left_bigger_right_kernel);
}
void Base256uMathTests::CUDA::min::left_smaller_right() {
	max_test_macro(min_left_smaller_right_kernel);
}
void Base256uMathTests::CUDA::min::big_left_bigger_left() {
	max_test_macro(min_big_left_bigger_left_kernel);
}
void Base256uMathTests::CUDA::min::big_left_smaller_left() {
	max_test_macro(min_big_left_smaller_left_kernel);
}
void Base256uMathTests::CUDA::min::big_left_bigger_right() {
	max_test_macro(min_big_left_bigger_right_kernel);
}
void Base256uMathTests::CUDA::min::big_left_smaller_right() {
	max_test_macro(min_big_left_smaller_right_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::bitwise_and::test() {
	ideal_case();
	big_ideal_case();
	left_bigger();
	left_smaller();
	big_left_bigger();
	big_left_smaller();
	dst_too_small();
	big_dst_too_small();
	left_n_zero();
	right_n_zero();
	dst_n_zero();

	in_place_ideal_case();
	in_place_big_ideal_case();
	in_place_left_bigger();
	in_place_left_smaller();
	in_place_big_left_bigger();
	in_place_big_left_smaller();
	in_place_left_n_zero();
	in_place_right_n_zero();
}

__global__
void bitwise_and_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short left = 49816;
	unsigned short right = 13925;
	unsigned short dst;
	auto return_code = Base256uMath::bitwise_and(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != (left & right)) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_and_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 177, 191, 253, 203, 209, 95, 131, 49, 194 };
	unsigned char right[] = { 97, 78, 90, 246, 21, 127, 59, 186, 23 };
	unsigned char dst[9];
	auto return_code = Base256uMath::bitwise_and(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != (left[i] & right[i])) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void bitwise_and_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 3477483;
	unsigned short right = 16058;
	unsigned int dst;
	auto return_code = Base256uMath::bitwise_and(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != (left & right)) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_and_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short left = 20968;
	unsigned int right = 226081;
	unsigned int dst;
	auto return_code = Base256uMath::bitwise_and(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != (left & right)) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_and_big_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 53, 138, 217, 66, 48, 69, 213, 139, 91, 163, 230, 8 };
	unsigned char right[] = { 16, 126, 35, 76, 137, 248, 62, 10, 11 };
	unsigned char dst[12];
	auto return_code = Base256uMath::bitwise_and(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 16, 10, 1, 64, 0, 64, 20, 10, 11, 0, 0, 0 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void bitwise_and_big_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 191, 126, 201, 200, 181, 4, 177, 127, 163 };
	unsigned char right[] = { 145, 53, 22, 29, 169, 149, 201, 111, 97, 66, 21, 27 };
	unsigned char dst[12];
	auto return_code = Base256uMath::bitwise_and(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 145, 52, 0, 8, 161, 4, 129, 111, 33, 0, 0, 0 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void bitwise_and_dst_too_small_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 61107471;
	unsigned int right = 186824;
	unsigned short dst;
	unsigned short answer = left & right;
	auto return_code = Base256uMath::bitwise_and(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void bitwise_and_big_dst_too_small_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 31, 204, 101, 59, 181, 129, 29, 143, 123, 111 };
	unsigned char right[] = { 159, 29, 249, 164, 61, 95, 169, 60, 199, 5, 254 };
	unsigned char dst[9];
	auto return_code = Base256uMath::bitwise_and(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != (left[i] & right[i])) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void bitwise_and_left_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 2912879481;
	unsigned int right = -1;
	unsigned int dst = 5978137491;
	auto return_code = Base256uMath::bitwise_and(&left, 0, &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 0) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_and_right_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = -1;
	std::size_t right = 2912879481;
	unsigned int dst = 5978137491;
	auto return_code = Base256uMath::bitwise_and(&left, sizeof(left), &right, 0, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 0) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_and_dst_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 2739824923;
	std::size_t right = 248020302;
	unsigned char dst = 223;
	auto return_code = Base256uMath::bitwise_and(&left, sizeof(left), &right, sizeof(right), &dst, 0);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 223) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void bitwise_and_in_place_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short left = 49816;
	unsigned short right = 13925;
	decltype(left) answer = left & right;
	auto return_code = Base256uMath::bitwise_and(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_and_in_place_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 177, 191, 253, 203, 209, 95, 131, 49, 194 };
	unsigned char right[] = { 97, 78, 90, 246, 21, 127, 59, 186, 23 };
	unsigned char answer[] = { 33, 14, 88, 194, 17, 95, 3, 48, 2 };
	auto return_code = Base256uMath::bitwise_and(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void bitwise_and_in_place_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 3477482;
	unsigned short right = 16058;
	decltype(left) answer = left & right;
	auto return_code = Base256uMath::bitwise_and(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_and_in_place_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short left = 20968;
	unsigned int right = 22673;
	decltype(left) answer = left & right;
	auto return_code = Base256uMath::bitwise_and(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_and_in_place_big_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 53, 138, 217, 66, 48, 69, 213, 139, 91, 163, 230, 8 };
	unsigned char right[] = { 16, 126, 35, 76, 137, 248, 62, 10, 11 };
	auto return_code = Base256uMath::bitwise_and(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 16, 10, 1, 64, 0, 64, 20, 10, 11, 0, 0, 0 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void bitwise_and_in_place_big_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 191, 126, 201, 200, 181, 4, 177, 127, 163 };
	unsigned char right[] = { 145, 53, 22, 29, 169, 149, 201, 111, 97, 66, 21, 27 };
	auto return_code = Base256uMath::bitwise_and(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 145, 52, 0, 8, 161, 4, 129, 111, 33, 0, 0, 0 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void bitwise_and_in_place_left_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 2912879481;
	unsigned int right = -1;
	decltype(left) answer = 2912879481;
	auto return_code = Base256uMath::bitwise_and(&left, 0, &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_and_in_place_right_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = -1;
	std::size_t right = 2912879481;
	decltype(left) answer = 0;
	auto return_code = Base256uMath::bitwise_and(&left, sizeof(left), &right, 0);
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}

#define bitwise_test_macro(kernel_name) \
int code = -1; \
std::size_t size = 15; \
int* d_code; \
auto err = cudaMalloc(&d_code, sizeof(int)); \
cudaMalloc_check_macro(err); \
void* result = malloc(size); \
if (!result) { \
	std::cout << "couldn't allocate enough memory on the host" << std::endl; \
	assert(result != nullptr); \
} \
void* d_result; \
err = cudaMalloc(&d_result, size); \
cudaMalloc_check_macro(err); \
std::size_t* d_size; \
err = cudaMalloc(&d_size, sizeof(std::size_t)); \
cudaMalloc_check_macro(err); \
err = cudaMemcpy(d_size, &size, sizeof(std::size_t), cudaMemcpyHostToDevice); \
cudaMemcpy_check_macro(err); \
KERNEL_CALL3(kernel_name, d_code, d_result, d_size); \
err = cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost); \
cudaMemcpy_check_macro(err); \
err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost); \
cudaMemcpy_check_macro(err); \
if (code != 0) { \
	std::cout << "code: " << code << std::endl; \
	std::cout << "result: "; \
	for (unsigned char i = 0; i < size; i++) { \
		std::cout << std::to_string(reinterpret_cast<unsigned char*>(result)[i]) << " "; \
	} \
	std::cout << std::endl; \
	assert(code == 0); \
} \
free(result); \
cudaFree(d_result);

void Base256uMathTests::CUDA::bitwise_and::ideal_case() {
	bitwise_test_macro(bitwise_and_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::big_ideal_case() {
	bitwise_test_macro(bitwise_and_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::left_bigger() {
	bitwise_test_macro(bitwise_and_left_bigger_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::left_smaller() {
	bitwise_test_macro(bitwise_and_left_smaller_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::big_left_bigger() {
	bitwise_test_macro(bitwise_and_big_left_bigger_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::big_left_smaller() {
	bitwise_test_macro(bitwise_and_big_left_smaller_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::dst_too_small() {
	bitwise_test_macro(bitwise_and_dst_too_small_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::big_dst_too_small() {
	bitwise_test_macro(bitwise_and_big_dst_too_small_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::left_n_zero() {
	bitwise_test_macro(bitwise_and_left_n_zero_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::right_n_zero() {
	bitwise_test_macro(bitwise_and_right_n_zero_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::dst_n_zero() {
	bitwise_test_macro(bitwise_and_dst_n_zero_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::in_place_ideal_case() {
	bitwise_test_macro(bitwise_and_in_place_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::in_place_big_ideal_case() {
	bitwise_test_macro(bitwise_and_in_place_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::in_place_left_bigger() {
	bitwise_test_macro(bitwise_and_in_place_left_bigger_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::in_place_left_smaller() {
	bitwise_test_macro(bitwise_and_in_place_left_smaller_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::in_place_big_left_bigger() {
	bitwise_test_macro(bitwise_and_in_place_big_left_bigger_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::in_place_big_left_smaller() {
	bitwise_test_macro(bitwise_and_in_place_big_left_smaller_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::in_place_left_n_zero() {
	bitwise_test_macro(bitwise_and_in_place_left_n_zero_kernel);
}
void Base256uMathTests::CUDA::bitwise_and::in_place_right_n_zero() {
	bitwise_test_macro(bitwise_and_in_place_right_n_zero_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::bitwise_or::test() {
	ideal_case();
	big_ideal_case();
	left_bigger();
	left_smaller();
	big_left_bigger();
	big_left_smaller();
	dst_too_small();
	big_dst_too_small();
	left_n_zero();
	right_n_zero();
	dst_n_zero();

	in_place_ideal_case();
	in_place_big_ideal_case();
	in_place_left_bigger();
	in_place_left_smaller();
	in_place_big_left_bigger();
	in_place_big_left_smaller();
	in_place_left_n_zero();
	in_place_right_n_zero();
}

__global__
void bitwise_or_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short left = 49816;
	unsigned short right = 13925;
	unsigned short dst;
	auto return_code = Base256uMath::bitwise_or(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != (left | right)) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_or_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 177, 191, 253, 203, 209, 95, 131, 49, 194 };
	unsigned char right[] = { 97, 78, 90, 246, 21, 127, 59, 186, 23 };
	unsigned char dst[9];
	auto return_code = Base256uMath::bitwise_or(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != (left[i] | right[i])) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void bitwise_or_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 3477483;
	unsigned short right = 16058;
	unsigned int dst;
	auto return_code = Base256uMath::bitwise_or(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != (left | right)) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_or_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short left = 20968;
	unsigned int right = 226081;
	unsigned int dst;
	auto return_code = Base256uMath::bitwise_or(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != (left | right)) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_or_big_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 53, 138, 217, 66, 48, 69, 213, 139, 91, 163, 230, 8 };
	unsigned char right[] = { 16, 126, 35, 76, 137, 248, 62, 10, 11 };
	unsigned char dst[12];
	auto return_code = Base256uMath::bitwise_or(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 53, 254, 251, 78, 185, 253, 255, 139, 91, 163, 230, 8 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void bitwise_or_big_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 191, 126, 201, 200, 181, 4, 177, 127, 163 };
	unsigned char right[] = { 145, 53, 22, 29, 169, 149, 201, 111, 97, 66, 21, 27 };
	unsigned char dst[12];
	auto return_code = Base256uMath::bitwise_or(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 191, 127, 223, 221, 189, 149, 249, 127, 227, 66, 21, 27 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void bitwise_or_dst_too_small_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 61107471;
	unsigned int right = 186824;
	unsigned short dst;
	unsigned short answer = left | right;
	auto return_code = Base256uMath::bitwise_or(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void bitwise_or_big_dst_too_small_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 31, 204, 101, 59, 181, 129, 29, 143, 123, 111 };
	unsigned char right[] = { 159, 29, 249, 164, 61, 95, 169, 60, 199, 5, 254 };
	unsigned char dst[9];
	auto return_code = Base256uMath::bitwise_or(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != (left[i] | right[i])) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void bitwise_or_left_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 2912879481;
	unsigned int right = -1;
	unsigned int dst = 5978137491;
	auto return_code = Base256uMath::bitwise_or(&left, 0, &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != right) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_or_right_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = -1;
	std::size_t right = 2912879481;
	unsigned int dst = -1;
	auto return_code = Base256uMath::bitwise_or(&left, sizeof(left), &right, 0, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != left) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_or_dst_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 273983;
	std::size_t right = 24885;
	unsigned char dst = 223;
	auto return_code = Base256uMath::bitwise_or(&left, sizeof(left), &right, sizeof(right), &dst, 0);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 223) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void bitwise_or_in_place_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short left = 49816;
	unsigned short right = 13925;
	decltype(left) answer = left | right;
	auto return_code = Base256uMath::bitwise_or(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_or_in_place_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 177, 191, 253, 203, 209, 95, 131, 49, 194 };
	unsigned char right[] = { 97, 78, 90, 246, 21, 127, 59, 186, 23 };
	unsigned char answer[] = { 241, 255, 255, 255, 213, 127, 187, 187, 215 };
	auto return_code = Base256uMath::bitwise_or(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void bitwise_or_in_place_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 3477482;
	unsigned short right = 16058;
	decltype(left) answer = left | right;
	auto return_code = Base256uMath::bitwise_or(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_or_in_place_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short left = 20968;
	unsigned int right = 22673;
	decltype(left) answer = left | right;
	auto return_code = Base256uMath::bitwise_or(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_or_in_place_big_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 53, 138, 217, 66, 48, 69, 213, 139, 91, 163, 230, 8 };
	unsigned char right[] = { 16, 126, 35, 76, 137, 248, 62, 10, 11 };
	auto return_code = Base256uMath::bitwise_or(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 53, 254, 251, 78, 185, 253, 255, 139, 91, 163, 230, 8 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void bitwise_or_in_place_big_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 191, 126, 201, 200, 181, 4, 177, 127, 163 };
	unsigned char right[] = { 145, 53, 22, 29, 169, 149, 201, 111, 97, 66, 21, 27 };
	auto return_code = Base256uMath::bitwise_or(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 191, 127, 223, 221, 189, 149, 249, 127, 227, 66, 21, 27 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void bitwise_or_in_place_left_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 29128481;
	unsigned int right = -1;
	decltype(left) answer = 29128481;
	auto return_code = Base256uMath::bitwise_or(&left, 0, &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_or_in_place_right_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = -1;
	std::size_t right = 2912879481;
	decltype(left) answer = -1;
	auto return_code = Base256uMath::bitwise_or(&left, sizeof(left), &right, 0);
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}

void Base256uMathTests::CUDA::bitwise_or::ideal_case() {
	bitwise_test_macro(bitwise_or_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::big_ideal_case() {
	bitwise_test_macro(bitwise_or_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::left_bigger() {
	bitwise_test_macro(bitwise_or_left_bigger_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::left_smaller() {
	bitwise_test_macro(bitwise_or_left_smaller_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::big_left_bigger() {
	bitwise_test_macro(bitwise_or_big_left_bigger_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::big_left_smaller() {
	bitwise_test_macro(bitwise_or_big_left_smaller_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::dst_too_small() {
	bitwise_test_macro(bitwise_or_dst_too_small_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::big_dst_too_small() {
	bitwise_test_macro(bitwise_or_big_dst_too_small_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::left_n_zero() {
	bitwise_test_macro(bitwise_or_left_n_zero_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::right_n_zero() {
	bitwise_test_macro(bitwise_or_right_n_zero_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::dst_n_zero() {
	bitwise_test_macro(bitwise_or_dst_n_zero_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::in_place_ideal_case() {
	bitwise_test_macro(bitwise_or_in_place_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::in_place_big_ideal_case() {
	bitwise_test_macro(bitwise_or_in_place_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::in_place_left_bigger() {
	bitwise_test_macro(bitwise_or_in_place_left_bigger_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::in_place_left_smaller() {
	bitwise_test_macro(bitwise_or_in_place_left_smaller_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::in_place_big_left_bigger() {
	bitwise_test_macro(bitwise_or_in_place_big_left_bigger_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::in_place_big_left_smaller() {
	bitwise_test_macro(bitwise_or_in_place_big_left_smaller_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::in_place_left_n_zero() {
	bitwise_test_macro(bitwise_or_in_place_left_n_zero_kernel);
}
void Base256uMathTests::CUDA::bitwise_or::in_place_right_n_zero() {
	bitwise_test_macro(bitwise_or_in_place_right_n_zero_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::bitwise_xor::test() {
	ideal_case();
	big_ideal_case();
	left_bigger();
	left_smaller();
	big_left_bigger();
	big_left_smaller();
	dst_too_small();
	big_dst_too_small();
	left_n_zero();
	right_n_zero();
	dst_n_zero();

	in_place_ideal_case();
	in_place_big_ideal_case();
	in_place_left_bigger();
	in_place_left_smaller();
	in_place_big_left_bigger();
	in_place_big_left_smaller();
	in_place_left_n_zero();
	in_place_right_n_zero();
}

__global__
void bitwise_xor_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short left = 49816;
	unsigned short right = 13925;
	unsigned short dst;
	auto return_code = Base256uMath::bitwise_xor(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != (left ^ right)) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_xor_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 177, 191, 253, 203, 209, 95, 131, 49, 194 };
	unsigned char right[] = { 97, 78, 90, 246, 21, 127, 59, 186, 23 };
	unsigned char dst[9];
	auto return_code = Base256uMath::bitwise_xor(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != (left[i] ^ right[i])) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void bitwise_xor_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 3477483;
	unsigned short right = 16058;
	unsigned int dst;
	auto return_code = Base256uMath::bitwise_xor(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != (left ^ right)) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_xor_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short left = 20968;
	unsigned int right = 226081;
	unsigned int dst;
	auto return_code = Base256uMath::bitwise_xor(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != (left ^ right)) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_xor_big_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 53, 138, 217, 66, 48, 69, 213, 139, 91, 163, 230, 8 };
	unsigned char right[] = { 16, 126, 35, 76, 137, 248, 62, 10, 11 };
	unsigned char dst[12];
	auto return_code = Base256uMath::bitwise_xor(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 37, 244, 250, 14, 185, 189, 235, 129, 80, 163, 230, 8 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void bitwise_xor_big_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 191, 126, 201, 200, 181, 4, 177, 127, 163 };
	unsigned char right[] = { 145, 53, 22, 29, 169, 149, 201, 111, 97, 66, 21, 27 };
	unsigned char dst[12];
	auto return_code = Base256uMath::bitwise_xor(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 46, 75, 223, 213, 28, 145, 120, 16, 194, 66, 21, 27 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void bitwise_xor_dst_too_small_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 61107471;
	unsigned int right = 186824;
	unsigned short dst;
	unsigned short answer = left ^ right;
	auto return_code = Base256uMath::bitwise_xor(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void bitwise_xor_big_dst_too_small_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 31, 204, 101, 59, 181, 129, 29, 143, 123, 111 };
	unsigned char right[] = { 159, 29, 249, 164, 61, 95, 169, 60, 199, 5, 254 };
	unsigned char dst[9];
	auto return_code = Base256uMath::bitwise_xor(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != (left[i] ^ right[i])) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void bitwise_xor_left_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 2912879481;
	unsigned int right = -1;
	unsigned int dst = 5978137491;
	auto return_code = Base256uMath::bitwise_xor(&left, 0, &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != right) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_xor_right_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = -1;
	std::size_t right = 2912879481;
	unsigned int dst = -1;
	auto return_code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, 0, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != left) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_xor_dst_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 273983;
	std::size_t right = 24885;
	unsigned char dst = 223;
	auto return_code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, sizeof(right), &dst, 0);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 223) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void bitwise_xor_in_place_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short left = 49816;
	unsigned short right = 13925;
	decltype(left) answer = left ^ right;
	auto return_code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_xor_in_place_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 177, 191, 253, 203, 209, 95, 131, 49, 194 };
	unsigned char right[] = { 97, 78, 90, 246, 21, 127, 59, 186, 23 };
	unsigned char answer[] = { 208, 241, 167, 61, 196, 32, 184, 139, 213 };
	auto return_code = Base256uMath::bitwise_xor(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void bitwise_xor_in_place_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 3477482;
	unsigned short right = 16058;
	decltype(left) answer = left ^ right;
	auto return_code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_xor_in_place_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short left = 20968;
	unsigned int right = 22673;
	decltype(left) answer = left ^ right;
	auto return_code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_xor_in_place_big_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 53, 138, 217, 66, 48, 69, 213, 139, 91, 163, 230, 8 };
	unsigned char right[] = { 16, 126, 35, 76, 137, 248, 62, 10, 11 };
	auto return_code = Base256uMath::bitwise_xor(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 37, 244, 250, 14, 185, 189, 235, 129, 80, 163, 230, 8 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void bitwise_xor_in_place_big_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 191, 126, 201, 200, 181, 4, 177, 127, 163 };
	unsigned char right[] = { 145, 53, 22, 29, 169, 149, 201, 111, 97, 66, 21, 27 };
	auto return_code = Base256uMath::bitwise_xor(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 46, 75, 223, 213, 28, 145, 120, 16, 194, 66, 21, 27 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void bitwise_xor_in_place_left_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 29128481;
	unsigned int right = -1;
	decltype(left) answer = 29128481;
	auto return_code = Base256uMath::bitwise_xor(&left, 0, &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_xor_in_place_right_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = -1;
	std::size_t right = 2919481;
	decltype(left) answer = -1;
	auto return_code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, 0);
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}

void Base256uMathTests::CUDA::bitwise_xor::ideal_case() {
	bitwise_test_macro(bitwise_xor_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::big_ideal_case() {
	bitwise_test_macro(bitwise_xor_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::left_bigger() {
	bitwise_test_macro(bitwise_xor_left_bigger_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::left_smaller() {
	bitwise_test_macro(bitwise_xor_left_smaller_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::big_left_bigger() {
	bitwise_test_macro(bitwise_xor_big_left_bigger_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::big_left_smaller() {
	bitwise_test_macro(bitwise_xor_big_left_smaller_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::dst_too_small() {
	bitwise_test_macro(bitwise_xor_dst_too_small_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::big_dst_too_small() {
	bitwise_test_macro(bitwise_xor_big_dst_too_small_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::left_n_zero() {
	bitwise_test_macro(bitwise_xor_left_n_zero_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::right_n_zero() {
	bitwise_test_macro(bitwise_xor_right_n_zero_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::dst_n_zero() {
	bitwise_test_macro(bitwise_xor_dst_n_zero_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::in_place_ideal_case() {
	bitwise_test_macro(bitwise_xor_in_place_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::in_place_big_ideal_case() {
	bitwise_test_macro(bitwise_xor_in_place_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::in_place_left_bigger() {
	bitwise_test_macro(bitwise_xor_in_place_left_bigger_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::in_place_left_smaller() {
	bitwise_test_macro(bitwise_xor_in_place_left_smaller_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::in_place_big_left_bigger() {
	bitwise_test_macro(bitwise_xor_in_place_big_left_bigger_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::in_place_big_left_smaller() {
	bitwise_test_macro(bitwise_xor_in_place_big_left_smaller_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::in_place_left_n_zero() {
	bitwise_test_macro(bitwise_xor_in_place_left_n_zero_kernel);
}
void Base256uMathTests::CUDA::bitwise_xor::in_place_right_n_zero() {
	bitwise_test_macro(bitwise_xor_in_place_right_n_zero_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::bitwise_not::test() {
	ideal_case();
	big_ideal_case();
	src_n_zero();
}

__global__
void bitwise_not_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 2493050980;
	unsigned int answer = ~src;
	auto return_code = Base256uMath::bitwise_not(&src, sizeof(src));
	memset(output, 0, *size);
	memcpy(output, &src, sizeof(src));
	if (src != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bitwise_not_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 180, 127, 35, 146, 158, 174, 69, 249, 147 };
	auto return_code = Base256uMath::bitwise_not(src, sizeof(src));
	memset(output, 0, *size);
	memcpy(output, src, sizeof(src));
	unsigned char answer[] = { 75, 128, 220, 109, 97, 81, 186, 6, 108 };
	for (unsigned char i = 0; i < sizeof(src); i++) {
		if (src[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(src) + 1;
	}
}
__global__
void bitwise_not_src_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	auto return_code = Base256uMath::bitwise_not(src, 0);
	memset(output, 0, *size);
	memcpy(output, src, sizeof(src));
	for (unsigned char i = 0; i < sizeof(src); i++) {
		if (src[i] != i) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(src) + 1;
	}
}

#define bitwise_not_test_macro(kernel_name) \
int code = -1; \
int* d_code; \
auto err = cudaMalloc(&d_code, sizeof(int)); \
cudaMalloc_check_macro(err); \
std::size_t size = 15; \
void* result = malloc(size); \
if (!result) { \
	std::cout << "couldn't allocate enough memory on the host" << std::endl; \
	assert(result != nullptr); \
} \
void* d_result; \
err = cudaMalloc(&d_result, size); \
cudaMalloc_check_macro(err); \
std::size_t* d_size; \
err = cudaMalloc(&d_size, sizeof(std::size_t)); \
cudaMalloc_check_macro(err); \
err = cudaMemcpy(d_size, &size, sizeof(std::size_t), cudaMemcpyHostToDevice); \
cudaMemcpy_check_macro(err); \
KERNEL_CALL3(kernel_name, d_code, d_result, d_size); \
err = cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost); \
cudaMemcpy_check_macro(err); \
err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost); \
cudaMemcpy_check_macro(err); \
if (code != 0) { \
	std::cout << "code: " << std::to_string(code) << std::endl; \
	std::cout << "result: "; \
	for (std::size_t i = 0; i < size; i++) { \
		std::cout << std::to_string(reinterpret_cast<unsigned char*>(result)[i]) << " "; \
	} \
	std::cout << std::endl; \
	assert(code == 0); \
} \
free(result); \
cudaFree(d_code); \
cudaFree(d_size); \
cudaFree(d_result);

void Base256uMathTests::CUDA::bitwise_not::ideal_case() {
	bitwise_not_test_macro(bitwise_not_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bitwise_not::big_ideal_case() {
	bitwise_not_test_macro(bitwise_not_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bitwise_not::src_n_zero() {
	bitwise_not_test_macro(bitwise_not_src_n_zero_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::byte_shift_left::test() {
	ideal_case();
	big_ideal_case();
	src_n_less_than_by();
	src_n_greater_than_dst_n();
	src_n_less_than_dst_n();
	src_n_zero();
	dst_n_zero();
	by_is_zero();

	in_place_ideal_case();
	in_place_big_ideal_case();
	in_place_src_n_less_than_by();
	in_place_src_n_zero();
	in_place_by_is_zero();
}

__global__
void byte_shift_left_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 1305258424;
	std::size_t by = 5;
	std::size_t dst;
	std::size_t answer = src << (by * 8);
	auto return_code = Base256uMath::byte_shift_left(&src, sizeof(src), by, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void byte_shift_left_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 223, 192, 7, 188, 111, 229, 33, 55, 8 };
	std::size_t by = 3;
	unsigned char dst[9];
	auto return_code = Base256uMath::byte_shift_left(src, sizeof(src), by, dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 0, 0, 0, 223, 192, 7, 188, 111, 229 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void byte_shift_left_src_n_less_than_by_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short src = 1337;
	std::size_t by = sizeof(src) + 1;
	unsigned short dst = 394021884;
	auto return_code = Base256uMath::byte_shift_left(&src, sizeof(src), by, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 0) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void byte_shift_left_src_n_greater_than_dst_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 39158;
	std::size_t by = 1;
	unsigned short dst;
	auto return_code = Base256uMath::byte_shift_left(&src, sizeof(src), by, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	decltype(dst) answer = src << (by * 8);
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void byte_shift_left_src_n_less_than_dst_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 244, 184, 73, 236, 228, 182, 41, 107, 81 };
	unsigned char dst[] = { 159, 188, 20, 222, 209, 85, 173, 112, 72, 73, 40, 123 };
	std::size_t by = 3;
	auto return_code = Base256uMath::byte_shift_left(src, sizeof(src), by, dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 0, 0, 0, 244, 184, 73, 236, 228, 182, 0, 0, 0 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void byte_shift_left_src_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	unsigned char dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	auto return_code = Base256uMath::byte_shift_left(src, 0, 3, dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != 0) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void byte_shift_left_dst_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	unsigned char dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	auto return_code = Base256uMath::byte_shift_left(src, sizeof(src), 3, dst, 0);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != i + 10) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void byte_shift_left_by_is_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 1334;
	unsigned int dst = 39301;
	auto return_code = Base256uMath::byte_shift_left(&src, sizeof(src), 0, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 1334) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void byte_shift_left_in_place_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 258424;
	std::size_t by = 5;
	std::size_t answer = src << (by * 8);
	auto return_code = Base256uMath::byte_shift_left(&src, sizeof(src), by);
	memset(output, 0, *size);
	memcpy(output, &src, sizeof(src));
	if (src != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void byte_shift_left_in_place_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 223, 192, 7, 188, 111, 229, 33, 55, 8 };
	std::size_t by = 3;
	auto return_code = Base256uMath::byte_shift_left(src, sizeof(src), by);
	memset(output, 0, *size);
	memcpy(output, src, sizeof(src));
	unsigned char answer[] = { 0, 0, 0, 223, 192, 7, 188, 111, 229 };
	for (unsigned char i = 0; i < sizeof(src); i++) {
		if (src[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(src) + 1;
	}
}
__global__
void byte_shift_left_in_place_src_n_less_than_by_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short src = 1337;
	std::size_t by = sizeof(src) + 1;
	auto return_code = Base256uMath::byte_shift_left(&src, sizeof(src), by);
	memset(output, 0, *size);
	memcpy(output, &src, sizeof(src));
	if (src != 0) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void byte_shift_left_in_place_src_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	auto return_code = Base256uMath::byte_shift_left(src, 0, 3);
	memset(output, 0, *size);
	memcpy(output, src, sizeof(src));
	for (unsigned char i = 0; i < sizeof(src); i++) {
		if (src[i] != i) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(src) + 1;
	}
}
__global__
void byte_shift_left_in_place_by_is_zero_kernel(int* code, void* output, std::size_t* size) {
	unsigned int src = 133804;
	auto return_code = Base256uMath::byte_shift_left(&src, sizeof(src), 0);
	memset(output, 0, *size);
	memcpy(output, &src, sizeof(src));
	if (src != 133804) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}

#define byte_shift_test_macro(kernel_name) \
int code = -1; \
int* d_code; \
auto err = cudaMalloc(&d_code, sizeof(int)); \
cudaMalloc_check_macro(err); \
std::size_t size = 15; \
unsigned char result[size]; \
void* d_result; \
err = cudaMalloc(&d_result, size); \
cudaMalloc_check_macro(err); \
std::size_t* d_size; \
err = cudaMalloc(&d_size, sizeof(std::size_t)); \
cudaMalloc_check_macro(err); \
err = cudaMemcpy(d_size, &size, sizeof(std::size_t), cudaMemcpyHostToDevice); \
cudaMemcpy_check_macro(err); \
KERNEL_CALL3(kernel_name, d_code, d_result, d_size); \
err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost); \
cudaMemcpy_check_macro(err); \
err = cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost); \
cudaMemcpy_check_macro(err); \
if (code != 0) { \
	std::cout << "code: " << std::to_string(code) << std::endl; \
	std::cout << "result: "; \
	for (std::size_t i = 0; i < size; i++) { \
		std::cout << std::to_string(reinterpret_cast<unsigned char*>(result)[i]) << " "; \
	} \
	std::cout << std::endl; \
	assert(code == 0); \
} \
cudaFree(d_code); \
cudaFree(d_result); \
cudaFree(d_size);

void Base256uMathTests::CUDA::byte_shift_left::ideal_case() {
	byte_shift_test_macro(byte_shift_left_ideal_case_kernel);
}
void Base256uMathTests::CUDA::byte_shift_left::big_ideal_case() {
	byte_shift_test_macro(byte_shift_left_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::byte_shift_left::src_n_less_than_by() {
	byte_shift_test_macro(byte_shift_left_src_n_less_than_by_kernel);
}
void Base256uMathTests::CUDA::byte_shift_left::src_n_greater_than_dst_n() {
	byte_shift_test_macro(byte_shift_left_src_n_greater_than_dst_n_kernel);
}
void Base256uMathTests::CUDA::byte_shift_left::src_n_less_than_dst_n() {
	byte_shift_test_macro(byte_shift_left_src_n_less_than_dst_n_kernel);
}
void Base256uMathTests::CUDA::byte_shift_left::src_n_zero() {
	byte_shift_test_macro(byte_shift_left_src_n_zero_kernel);
}
void Base256uMathTests::CUDA::byte_shift_left::dst_n_zero() {
	byte_shift_test_macro(byte_shift_left_dst_n_zero_kernel);
}
void Base256uMathTests::CUDA::byte_shift_left::by_is_zero() {
	byte_shift_test_macro(byte_shift_left_by_is_zero_kernel);
}
void Base256uMathTests::CUDA::byte_shift_left::in_place_ideal_case() {
	byte_shift_test_macro(byte_shift_left_in_place_ideal_case_kernel);
}
void Base256uMathTests::CUDA::byte_shift_left::in_place_big_ideal_case() {
	byte_shift_test_macro(byte_shift_left_in_place_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::byte_shift_left::in_place_src_n_less_than_by() {
	byte_shift_test_macro(byte_shift_left_in_place_src_n_less_than_by_kernel);
}
void Base256uMathTests::CUDA::byte_shift_left::in_place_src_n_zero() {
	byte_shift_test_macro(byte_shift_left_in_place_src_n_zero_kernel);
}
void Base256uMathTests::CUDA::byte_shift_left::in_place_by_is_zero() {
	byte_shift_test_macro(byte_shift_left_in_place_by_is_zero_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::byte_shift_right::test() {
	ideal_case();
	big_ideal_case();
	src_n_less_than_by();
	src_n_greater_than_dst_n();
	src_n_less_than_dst_n();
	src_n_zero();
	dst_n_zero();
	by_is_zero();

	in_place_ideal_case();
	in_place_big_ideal_case();
	in_place_src_n_less_than_by();
	in_place_src_n_zero();
	in_place_by_is_zero();
}

__global__
void byte_shift_right_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 1305258424;
	std::size_t by = 5;
	std::size_t dst;
	std::size_t answer = src >> (by * 8);
	auto return_code = Base256uMath::byte_shift_right(&src, sizeof(src), by, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void byte_shift_right_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 223, 192, 7, 188, 111, 229, 33, 55, 8 };
	std::size_t by = 3;
	unsigned char dst[9];
	auto return_code = Base256uMath::byte_shift_right(src, sizeof(src), by, dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 188, 111, 229, 33, 55, 8, 0, 0, 0 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void byte_shift_right_src_n_less_than_by_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short src = 1337;
	std::size_t by = sizeof(src) + 1;
	unsigned short dst = 394021884;
	auto return_code = Base256uMath::byte_shift_right(&src, sizeof(src), by, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 0) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void byte_shift_right_src_n_greater_than_dst_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 39158;
	std::size_t by = 1;
	unsigned short dst;
	auto return_code = Base256uMath::byte_shift_right(&src, sizeof(src), by, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	decltype(dst) answer = src >> (by * 8);
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void byte_shift_right_src_n_less_than_dst_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 244, 184, 73, 236, 228, 182, 41, 107, 81 };
	unsigned char dst[] = { 159, 188, 20, 222, 209, 85, 173, 112, 72, 73, 40, 123 };
	std::size_t by = 3;
	auto return_code = Base256uMath::byte_shift_right(src, sizeof(src), by, dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 236, 228, 182, 41, 107, 81, 0, 0, 0 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void byte_shift_right_src_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	unsigned char dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	auto return_code = Base256uMath::byte_shift_right(src, 0, 3, dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != 0) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void byte_shift_right_dst_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	unsigned char dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	auto return_code = Base256uMath::byte_shift_right(src, sizeof(src), 3, dst, 0);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != i + 10) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void byte_shift_right_by_is_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 1334;
	unsigned int dst = 39301;
	auto return_code = Base256uMath::byte_shift_right(&src, sizeof(src), 0, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 1334) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void byte_shift_right_in_place_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 258424;
	std::size_t by = 5;
	std::size_t answer = src >> (by * 8);
	auto return_code = Base256uMath::byte_shift_right(&src, sizeof(src), by);
	memset(output, 0, *size);
	memcpy(output, &src, sizeof(src));
	if (src != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void byte_shift_right_in_place_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 223, 192, 7, 188, 111, 229, 33, 55, 8 };
	std::size_t by = 3;
	auto return_code = Base256uMath::byte_shift_right(src, sizeof(src), by);
	memset(output, 0, *size);
	memcpy(output, src, sizeof(src));
	unsigned char answer[] = { 188, 111, 229, 33, 55, 8, 0, 0, 0 };
	for (unsigned char i = 0; i < sizeof(src); i++) {
		if (src[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(src) + 1;
	}
}
__global__
void byte_shift_right_in_place_src_n_less_than_by_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short src = 1337;
	std::size_t by = sizeof(src) + 1;
	auto return_code = Base256uMath::byte_shift_right(&src, sizeof(src), by);
	memset(output, 0, *size);
	memcpy(output, &src, sizeof(src));
	if (src != 0) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void byte_shift_right_in_place_src_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	auto return_code = Base256uMath::byte_shift_right(src, 0, 3);
	memset(output, 0, *size);
	memcpy(output, src, sizeof(src));
	for (unsigned char i = 0; i < sizeof(src); i++) {
		if (src[i] != i) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(src) + 1;
	}
}
__global__
void byte_shift_right_in_place_by_is_zero_kernel(int* code, void* output, std::size_t* size) {
	unsigned int src = 133804;
	auto return_code = Base256uMath::byte_shift_right(&src, sizeof(src), 0);
	memset(output, 0, *size);
	memcpy(output, &src, sizeof(src));
	if (src != 133804) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}

void Base256uMathTests::CUDA::byte_shift_right::ideal_case() {
	byte_shift_test_macro(byte_shift_right_ideal_case_kernel);
}
void Base256uMathTests::CUDA::byte_shift_right::big_ideal_case() {
	byte_shift_test_macro(byte_shift_right_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::byte_shift_right::src_n_less_than_by() {
	byte_shift_test_macro(byte_shift_right_src_n_less_than_by_kernel);
}
void Base256uMathTests::CUDA::byte_shift_right::src_n_greater_than_dst_n() {
	byte_shift_test_macro(byte_shift_right_src_n_greater_than_dst_n_kernel);
}
void Base256uMathTests::CUDA::byte_shift_right::src_n_less_than_dst_n() {
	byte_shift_test_macro(byte_shift_right_src_n_less_than_dst_n_kernel);
}
void Base256uMathTests::CUDA::byte_shift_right::src_n_zero() {
	byte_shift_test_macro(byte_shift_right_src_n_zero_kernel);
}
void Base256uMathTests::CUDA::byte_shift_right::dst_n_zero() {
	byte_shift_test_macro(byte_shift_right_dst_n_zero_kernel);
}
void Base256uMathTests::CUDA::byte_shift_right::by_is_zero() {
	byte_shift_test_macro(byte_shift_right_by_is_zero_kernel);
}
void Base256uMathTests::CUDA::byte_shift_right::in_place_ideal_case() {
	byte_shift_test_macro(byte_shift_right_in_place_ideal_case_kernel);
}
void Base256uMathTests::CUDA::byte_shift_right::in_place_big_ideal_case() {
	byte_shift_test_macro(byte_shift_right_in_place_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::byte_shift_right::in_place_src_n_less_than_by() {
	byte_shift_test_macro(byte_shift_right_in_place_src_n_less_than_by_kernel);
}
void Base256uMathTests::CUDA::byte_shift_right::in_place_src_n_zero() {
	byte_shift_test_macro(byte_shift_right_in_place_src_n_zero_kernel);
}
void Base256uMathTests::CUDA::byte_shift_right::in_place_by_is_zero() {
	byte_shift_test_macro(byte_shift_right_in_place_by_is_zero_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::bit_shift_left::test() {
	ideal_case();
	big_ideal_case();
	src_n_less_than_by();
	src_n_greater_than_dst_n();
	src_n_less_than_dst_n();
	src_n_zero();
	dst_n_zero();
	by_n_zero();

	in_place_ideal_case();
	in_place_big_ideal_case();
	in_place_src_n_less_than_by();
	in_place_src_n_zero();
	in_place_by_n_zero();
}

__global__
void bit_shift_left_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 14687480,
		dst = 0,
		by = 45;
	std::size_t answer = src << by;
	auto return_code = Base256uMath::bit_shift_left(&src, sizeof(src), by, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_left_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 215, 204, 79, 242, 35, 94, 138, 32, 147 };
	unsigned char dst[9];
	std::size_t by = 45;
	unsigned char answer[] = { 0, 0, 0, 0, 0, 224, 154, 249, 73, 126, 196, 75, 17, 100, 18 };
	auto return_code = Base256uMath::bit_shift_left(src, sizeof(src), &by, sizeof(by), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void bit_shift_left_src_n_less_than_by_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 182389;
	std::size_t dst = 42070131;
	std::size_t by = sizeof(src) * 8 + 1;
	auto return_code = Base256uMath::bit_shift_left(&src, sizeof(src), &by, sizeof(by), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 0) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_left_src_n_greater_than_dst_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 146418;
	unsigned int dst;
	std::size_t by = 35;
	auto return_code = Base256uMath::bit_shift_left(&src, sizeof(src), &by, sizeof(by), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	unsigned int answer = src << by;
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void bit_shift_left_src_n_less_than_dst_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 39816;
	std::size_t dst = 7168245;
	std::size_t by = 24;
	auto return_code = Base256uMath::bit_shift_left(&src, sizeof(src), &by, sizeof(by), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	std::size_t answer = src << by;
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_left_src_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 2423423;
	unsigned int dst = 42831231;
	std::size_t by = 15;
	auto return_code = Base256uMath::bit_shift_left(&src, 0, &by, sizeof(by), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 0) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_left_dst_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 1234567890;
	unsigned int dst = 987654321;
	std::size_t by = 10;
	auto return_code = Base256uMath::bit_shift_left(&src, sizeof(src), &by, sizeof(by), &dst, 0);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 987654321) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void bit_shift_left_by_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 246810;
	unsigned int dst = 1357911;
	std::size_t by = 25;
	auto return_code = Base256uMath::bit_shift_left(&src, sizeof(src), &by, 0, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != src) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_left_in_place_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 146146596,
		by = 45;
	decltype(src) answer = src << by;
	
	auto return_code = Base256uMath::bit_shift_left(&src, sizeof(src), &by, sizeof(by));
	memset(output, 0, *size);
	memcpy(output, &src, sizeof(src));
	if (src != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_left_in_place_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 215, 204, 79, 242, 35, 94, 138, 32, 147 };
	std::size_t by = 45;
	auto return_code = Base256uMath::bit_shift_left(src, sizeof(src), &by, sizeof(by));
	memset(output, 0, *size);
	memcpy(output, src, sizeof(src));
	unsigned char answer[] = { 0, 0, 0, 0, 0, 224, 154, 249, 73, 126, 196, 75, 17, 100, 18 };
	for (unsigned char i = 0; i < sizeof(src); i++) {
		if (src[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(src) + 1;
	}
}
__global__
void bit_shift_left_in_place_src_n_less_than_by_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 1823827429,
		answer = 0;
	std::size_t by = sizeof(src) * 8 + 1;
	auto return_code = Base256uMath::bit_shift_left(&src, sizeof(src), &by, sizeof(by));
	memset(output, 0, *size);
	memcpy(output, &src, sizeof(src));
	if (src != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_left_in_place_src_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 24234233,
		answer = src;
	std::size_t by = 15;
	auto return_code = Base256uMath::bit_shift_left(&src, 0, &by, sizeof(by));
	memset(output, 0, *size);
	memcpy(output, &src, sizeof(src));
	if (src != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_left_in_place_by_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 246810,
		answer = src;
	std::size_t by = 25;
	auto return_code = Base256uMath::bit_shift_left(&src, sizeof(src), &by, 0);
	memset(output, 0, *size);
	memcpy(output, &src, sizeof(src));
	if (src != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}

void Base256uMathTests::CUDA::bit_shift_left::ideal_case() {
	byte_shift_test_macro(bit_shift_left_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bit_shift_left::big_ideal_case() {
	byte_shift_test_macro(bit_shift_left_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bit_shift_left::src_n_less_than_by() {
	byte_shift_test_macro(bit_shift_left_src_n_less_than_by_kernel);
}
void Base256uMathTests::CUDA::bit_shift_left::src_n_greater_than_dst_n() {
	byte_shift_test_macro(bit_shift_left_src_n_greater_than_dst_n_kernel);
}
void Base256uMathTests::CUDA::bit_shift_left::src_n_less_than_dst_n() {
	byte_shift_test_macro(bit_shift_left_src_n_less_than_dst_n_kernel);
}
void Base256uMathTests::CUDA::bit_shift_left::src_n_zero() {
	byte_shift_test_macro(bit_shift_left_src_n_zero_kernel);
}
void Base256uMathTests::CUDA::bit_shift_left::dst_n_zero() {
	byte_shift_test_macro(bit_shift_left_dst_n_zero_kernel);
}
void Base256uMathTests::CUDA::bit_shift_left::by_n_zero() {
	byte_shift_test_macro(bit_shift_left_by_n_zero_kernel);
}
void Base256uMathTests::CUDA::bit_shift_left::in_place_ideal_case() {
	byte_shift_test_macro(bit_shift_left_in_place_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bit_shift_left::in_place_big_ideal_case() {
	byte_shift_test_macro(bit_shift_left_in_place_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bit_shift_left::in_place_src_n_less_than_by() {
	byte_shift_test_macro(bit_shift_left_in_place_src_n_less_than_by_kernel);
}
void Base256uMathTests::CUDA::bit_shift_left::in_place_src_n_zero() {
	byte_shift_test_macro(bit_shift_left_in_place_src_n_zero_kernel);
}
void Base256uMathTests::CUDA::bit_shift_left::in_place_by_n_zero() {
	byte_shift_test_macro(bit_shift_left_in_place_by_n_zero_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::bit_shift_right::test() {
	ideal_case();
	big_ideal_case();
	src_n_less_than_by();
	src_n_greater_than_dst_n();
	src_n_less_than_dst_n();
	src_n_zero();
	dst_n_zero();
	by_n_zero();

	in_place_ideal_case();
	in_place_big_ideal_case();
	in_place_src_n_less_than_by();
	in_place_src_n_zero();
	in_place_by_n_zero();
}

__global__
void bit_shift_right_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 14687480,
		dst = 0,
		by = 45;
	std::size_t answer = src >> by;
	auto return_code = Base256uMath::bit_shift_right(&src, sizeof(src), by, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_right_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 215, 204, 79, 242, 35, 94, 138, 32, 147 };
	unsigned char dst[9];
	std::size_t by = 45;
	unsigned char answer[] = { 82, 4, 153, 4 };
	auto return_code = Base256uMath::bit_shift_right(src, sizeof(src), &by, sizeof(by), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void bit_shift_right_src_n_less_than_by_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 182389;
	std::size_t dst = 42070131;
	std::size_t by = sizeof(src) * 8 + 1;
	auto return_code = Base256uMath::bit_shift_right(&src, sizeof(src), &by, sizeof(by), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 0) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_right_src_n_greater_than_dst_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 146418;
	unsigned int dst;
	std::size_t by = 35;
	auto return_code = Base256uMath::bit_shift_right(&src, sizeof(src), &by, sizeof(by), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	unsigned int answer = src >> by;
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void bit_shift_right_src_n_less_than_dst_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 39816;
	std::size_t dst = 7168245;
	std::size_t by = 24;
	auto return_code = Base256uMath::bit_shift_right(&src, sizeof(src), &by, sizeof(by), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	std::size_t answer = src >> by;
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_right_src_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 2423423;
	unsigned int dst = 42831231;
	std::size_t by = 15;
	auto return_code = Base256uMath::bit_shift_right(&src, 0, &by, sizeof(by), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 0) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_right_dst_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 1234567890;
	unsigned int dst = 987654321;
	std::size_t by = 10;
	auto return_code = Base256uMath::bit_shift_right(&src, sizeof(src), &by, sizeof(by), &dst, 0);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 987654321) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void bit_shift_right_by_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 246810;
	unsigned int dst = 1357911;
	std::size_t by = 25;
	auto return_code = Base256uMath::bit_shift_right(&src, sizeof(src), &by, 0, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != src) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_right_in_place_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 146146596,
		by = 45;
	decltype(src) answer = src >> by;
	
	auto return_code = Base256uMath::bit_shift_right(&src, sizeof(src), &by, sizeof(by));
	memset(output, 0, *size);
	memcpy(output, &src, sizeof(src));
	if (src != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_right_in_place_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 215, 204, 79, 242, 35, 94, 138, 32, 147 };
	std::size_t by = 45;
	auto return_code = Base256uMath::bit_shift_right(src, sizeof(src), &by, sizeof(by));
	memset(output, 0, *size);
	memcpy(output, src, sizeof(src));
	unsigned char answer[] = { 82, 4, 153, 4 };
	for (unsigned char i = 0; i < sizeof(src); i++) {
		if (src[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(src) + 1;
	}
}
__global__
void bit_shift_right_in_place_src_n_less_than_by_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 1823827429,
		answer = 0;
	std::size_t by = sizeof(src) * 8 + 1;
	auto return_code = Base256uMath::bit_shift_right(&src, sizeof(src), &by, sizeof(by));
	memset(output, 0, *size);
	memcpy(output, &src, sizeof(src));
	if (src != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_right_in_place_src_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 24234233,
		answer = src;
	std::size_t by = 15;
	auto return_code = Base256uMath::bit_shift_right(&src, 0, &by, sizeof(by));
	memset(output, 0, *size);
	memcpy(output, &src, sizeof(src));
	if (src != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void bit_shift_right_in_place_by_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 246810,
		answer = src;
	std::size_t by = 25;
	auto return_code = Base256uMath::bit_shift_right(&src, sizeof(src), &by, 0);
	memset(output, 0, *size);
	memcpy(output, &src, sizeof(src));
	if (src != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}

void Base256uMathTests::CUDA::bit_shift_right::ideal_case() {
	byte_shift_test_macro(bit_shift_right_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bit_shift_right::big_ideal_case() {
	byte_shift_test_macro(bit_shift_right_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bit_shift_right::src_n_less_than_by() {
	byte_shift_test_macro(bit_shift_right_src_n_less_than_by_kernel);
}
void Base256uMathTests::CUDA::bit_shift_right::src_n_greater_than_dst_n() {
	byte_shift_test_macro(bit_shift_right_src_n_greater_than_dst_n_kernel);
}
void Base256uMathTests::CUDA::bit_shift_right::src_n_less_than_dst_n() {
	byte_shift_test_macro(bit_shift_right_src_n_less_than_dst_n_kernel);
}
void Base256uMathTests::CUDA::bit_shift_right::src_n_zero() {
	byte_shift_test_macro(bit_shift_right_src_n_zero_kernel);
}
void Base256uMathTests::CUDA::bit_shift_right::dst_n_zero() {
	byte_shift_test_macro(bit_shift_right_dst_n_zero_kernel);
}
void Base256uMathTests::CUDA::bit_shift_right::by_n_zero() {
	byte_shift_test_macro(bit_shift_right_by_n_zero_kernel);
}
void Base256uMathTests::CUDA::bit_shift_right::in_place_ideal_case() {
	byte_shift_test_macro(bit_shift_right_in_place_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bit_shift_right::in_place_big_ideal_case() {
	byte_shift_test_macro(bit_shift_right_in_place_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::bit_shift_right::in_place_src_n_less_than_by() {
	byte_shift_test_macro(bit_shift_right_in_place_src_n_less_than_by_kernel);
}
void Base256uMathTests::CUDA::bit_shift_right::in_place_src_n_zero() {
	byte_shift_test_macro(bit_shift_right_in_place_src_n_zero_kernel);
}
void Base256uMathTests::CUDA::bit_shift_right::in_place_by_n_zero() {
	byte_shift_test_macro(bit_shift_right_in_place_by_n_zero_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::increment::test() {
	ideal_case();
	big_ideal_case();
	overflow();
	big_overflow();
}

__global__
void increment_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	unsigned int num = 14703;
	auto return_code = Base256uMath::increment(&num, sizeof(num));
	memset(output, 0, *size);
	memcpy(output, &num, sizeof(num));
	if (num != 14704) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void increment_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	unsigned char num[] = { 72, 202, 187, 220, 23, 141, 160, 38, 41 };
	auto return_code = Base256uMath::increment(num, sizeof(num));
	memset(output, 0, *size);
	memcpy(output, num, sizeof(num));
	unsigned char answer[] = { 73, 202, 187, 220, 23, 141, 160, 38, 41 };
	for (unsigned char i = 0; i < sizeof(num); i++) {
		if (num[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(num) + 1;
	}
}
__global__
void increment_overflow_kernel(int* code, void* output, std::size_t* size) {
	unsigned int num = -1;
	auto return_code = Base256uMath::increment(&num, sizeof(num));
	memset(output, 0, *size);
	memcpy(output, &num, sizeof(num));
	if (num != 0) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::FLOW) {
		*code = 2;
	}
}
__global__
void increment_big_overflow_kernel(int* code, void* output, std::size_t* size) {
	unsigned char num[] = { 255, 255, 255, 255, 255, 255, 255, 255, 255 };
	auto return_code = Base256uMath::increment(num, sizeof(num));
	memset(output, 0, *size);
	memcpy(output, num, sizeof(num));
	for (unsigned char i = 0; i < sizeof(num); i++) {
		if (num[i] != 0) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::FLOW) {
		*code = sizeof(num) + 1;
	}
}

void Base256uMathTests::CUDA::increment::ideal_case() {
	byte_shift_test_macro(increment_ideal_case_kernel);
}
void Base256uMathTests::CUDA::increment::big_ideal_case() {
	byte_shift_test_macro(increment_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::increment::overflow() {
	byte_shift_test_macro(increment_overflow_kernel);
}
void Base256uMathTests::CUDA::increment::big_overflow() {
	byte_shift_test_macro(increment_big_overflow_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::decrement::test() {
	ideal_case();
	big_ideal_case();
	underflow();
	big_underflow();
}

__global__
void decrement_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	unsigned int num = 47157;
	auto return_code = Base256uMath::decrement(&num, sizeof(num));
	memset(output, 0, *size);
	memcpy(output, &num, sizeof(num));
	if (num != 47156) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void decrement_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	unsigned char num[] = { 82, 130, 64, 83, 78, 107, 211, 34, 158 };
	auto return_code = Base256uMath::decrement(num, sizeof(num));
	memset(output, 0, *size);
	memcpy(output, num, sizeof(num));
	unsigned char answer[] = { 81, 130, 64, 83, 78, 107, 211, 34, 158 };
	for (unsigned char i = 0; i < sizeof(num); i++) {
		if (num[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(num) + 1;
	}
}
__global__
void decrement_underflow_kernel(int* code, void* output, std::size_t* size) {
	unsigned int num = 0;
	auto return_code = Base256uMath::decrement(&num, sizeof(num));
	memset(output, 0, *size);
	memcpy(output, &num, sizeof(num));
	if (num != (unsigned int)-1) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::FLOW) {
		*code = 2;
	}
}
__global__
void decrement_big_underflow_kernel(int* code, void* output, std::size_t* size) {
	unsigned char num[] = { 0,0,0,0,0,0,0,0,0 };
	auto return_code = Base256uMath::decrement(num, sizeof(num));
	memset(output, 0, *size);
	memcpy(output, num, sizeof(num));
	for (unsigned char i = 0; i < sizeof(num); i++) {
		if (num[i] != 255) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::FLOW) {
		*code = sizeof(num) + 1;
	}
}

void Base256uMathTests::CUDA::decrement::ideal_case() {
	byte_shift_test_macro(decrement_ideal_case_kernel);
}
void Base256uMathTests::CUDA::decrement::big_ideal_case() {
	byte_shift_test_macro(decrement_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::decrement::underflow() {
	byte_shift_test_macro(decrement_underflow_kernel);
}
void Base256uMathTests::CUDA::decrement::big_underflow() {
	byte_shift_test_macro(decrement_big_underflow_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::add::test() {
	ideal_case();
	big_ideal_case();
	left_bigger();
	left_smaller();
	big_left_bigger();
	big_left_smaller();
	overflow();
	big_overflow();
	dst_too_small();
	big_dst_too_small();
	zero_for_left_n();
	zero_for_right_n();
	zero_for_dst_n();

	in_place_ideal_case();
	in_place_big_ideal_case();
	in_place_left_bigger();
	in_place_left_smaller();
	in_place_big_left_bigger();
	in_place_big_left_smaller();
	in_place_overflow();
	in_place_big_overflow();
	in_place_zero_for_left_n();
	in_place_zero_for_right_n();
}

__global__
void add_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 21048;
	unsigned int right = 13196;
	unsigned int dst = 0;
	auto return_code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 34244) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void add_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 133, 141, 239, 45, 85, 113, 36, 5, 18 };
	unsigned char right[] = { 71, 61, 127, 205, 77, 38, 168, 183, 100 };
	unsigned char dst[] = { 0,0,0,0,0,0,0,0,0 };
	auto return_code = Base256uMath::add(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 204, 202, 110, 251, 162, 151, 204, 188, 118 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void add_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 838898231;
	unsigned int right = 62557;
	std::size_t dst = 0,
		answer = left + right;
	auto return_code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void add_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 14728;
	std::size_t right = 6254439797035876545;
	std::size_t dst = 0,
		answer = left + right;
	auto return_code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void add_big_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 89, 41, 94, 204, 226, 89, 158, 240, 172, 184, 0, 248 };
	unsigned char right[] = { 175, 209, 133, 96, 128, 118, 74, 9, 212 };
	unsigned char dst[] = { 0,0,0,0,0,0,0,0,0,0,0,0 };
	
	auto return_code = Base256uMath::add(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 8, 251, 227, 44, 99, 208, 232, 249, 128, 185, 0, 248 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void add_big_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 156, 140, 94, 99, 248, 185, 215, 241, 43 };
	unsigned char right[] = { 226, 149, 147, 57, 68, 129, 92, 115, 20, 129, 106, 73 };
	unsigned char dst[] = { 0,0,0,0,0,0,0,0,0,0,0,0 };
	auto return_code = Base256uMath::add(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 126, 34, 242, 156, 60, 59, 52, 101, 64, 129, 106, 73 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void add_overflow_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = -1;
	unsigned int right = 1;
	unsigned int dst = 0;
	auto return_code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 0) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::FLOW) {
		*code = 2;
	}
}
__global__
void add_big_overflow_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 255, 255, 255, 255, 255, 255, 255, 255, 255 };
	unsigned char right[] = { 1, 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char dst[] = { 52, 81, 217, 207, 245, 155, 109, 25, 252 };
	auto return_code = Base256uMath::add(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != 0) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::FLOW) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void add_dst_too_small_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	// when dst is too small, it should truncate the answer and return the appropriate code.

	std::size_t left = 8024591321371708722;
	unsigned int right = 64081;
	unsigned int dst = 0,
		answer = left + right;
	auto return_code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void add_big_dst_too_small_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	// when dst is too small, it should truncate the answer and return the appropriate code.

	unsigned char left[] = { 81, 161, 205, 28, 5, 231, 145, 223, 39, 28, 13, 92 };
	unsigned char right[] = { 250, 17, 104, 13, 192, 89, 177, 235, 10, 100 };
	unsigned char dst[] = { 0,0,0,0,0,0,0,0,0 };
	auto return_code = Base256uMath::add(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 75, 179, 53, 42, 197, 64, 67, 203, 50, 128, 13, 92 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void add_zero_for_left_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	// If left_n is 0, then it's the equivalent of adding 0 to right.

	unsigned int left = 1337;
	std::size_t right = -1;
	std::size_t dst = 0;
	auto return_code = Base256uMath::add(&left, 0, &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (right != dst) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void add_zero_for_right_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	// If right_n is 0, then it's the equivalent of adding 0 to left.

	unsigned char left[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	unsigned short right = 1;
	unsigned char dst[9];
	auto return_code = Base256uMath::add(left, sizeof(left), &right, 0, dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != left[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void add_zero_for_dst_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	// If dst_n is 0, then adding won't do anything but return a truncated error code.

	unsigned char left[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	unsigned char right[] = { 19, 20, 21, 22, 23, 24, 25, 26, 27 };
	unsigned char dst[] = { 28, 29, 30 };
	auto return_code = Base256uMath::add(left, sizeof(left), right, sizeof(right), dst, 0);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	if (dst[0] != 28) {
		*code = 1;
	}
	else if (dst[1] != 29) {
		*code = 2;
	}
	else if (dst[2] != 30) {
		*code = 3;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void add_in_place_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 21048;
	unsigned int right = 13196;
	unsigned int answer = left + right;
	auto return_code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void add_in_place_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 133, 141, 239, 45, 85, 113, 36, 5, 18 };
	unsigned char right[] = { 71, 61, 127, 205, 77, 38, 168, 183, 100 };
	auto return_code = Base256uMath::add(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 204, 202, 110, 251, 162, 151, 204, 188, 118 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void add_in_place_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 8388997231;
	unsigned int right = 62557;
	std::size_t answer = left + right;
	auto return_code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void add_in_place_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 62557;
	std::size_t right = 8388997231;
	decltype(left) answer = left + right;
	auto return_code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void add_in_place_big_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 89, 41, 94, 204, 226, 89, 158, 240, 172, 184, 0, 248 };
	unsigned char right[] = { 175, 209, 133, 96, 128, 118, 74, 9, 212 };
	auto return_code = Base256uMath::add(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 8, 251, 227, 44, 99, 208, 232, 249, 128, 185, 0, 248 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void add_in_place_big_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 156, 140, 94, 99, 248, 185, 215, 241, 43 };
	unsigned char right[] = { 226, 149, 147, 57, 68, 129, 92, 115, 20, 129, 106, 73 };
	auto return_code = Base256uMath::add(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 126, 34, 242, 156, 60, 59, 52, 101, 64, 129, 106, 73 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = sizeof(left) + 1;
	}
}
__global__
void add_in_place_overflow_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = -1;
	unsigned int right = 1;
	auto return_code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != 0) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::FLOW) {
		*code = 2;
	}
}
__global__
void add_in_place_big_overflow_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 255, 255, 255, 255, 255, 255, 255, 255, 255 };
	unsigned char right[] = { 1, 0, 0, 0, 0, 0, 0, 0, 0 };
	auto return_code = Base256uMath::add(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != 0) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::FLOW) {
		*code = sizeof(left) + 1;
	}
}
__global__
void add_in_place_zero_for_left_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	// If left_n is 0, then adding won't do anything but return a truncated error code.

	unsigned char left[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	unsigned char right[] = { 19, 20, 21, 22, 23, 24, 25, 26, 27 };
	auto return_code = Base256uMath::add(left, 0, right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != 10 + i) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = sizeof(left) + 1;
	}
}
__global__
void add_in_place_zero_for_right_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	// If right_n is 0, then it's the equivalent of adding 0 to left.

	unsigned char left[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	unsigned short right = 1;
	auto return_code = Base256uMath::add(left, sizeof(left), &right, 0);
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != i + 1) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}

void Base256uMathTests::CUDA::add::ideal_case() {
	byte_shift_test_macro(add_ideal_case_kernel);
}
void Base256uMathTests::CUDA::add::big_ideal_case() {
	byte_shift_test_macro(add_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::add::left_bigger() {
	byte_shift_test_macro(add_left_bigger_kernel);
}
void Base256uMathTests::CUDA::add::left_smaller() {
	byte_shift_test_macro(add_left_smaller_kernel);
}
void Base256uMathTests::CUDA::add::big_left_bigger() {
	byte_shift_test_macro(add_big_left_bigger_kernel);
}
void Base256uMathTests::CUDA::add::big_left_smaller() {
	byte_shift_test_macro(add_big_left_smaller_kernel);
}
void Base256uMathTests::CUDA::add::overflow() {
	byte_shift_test_macro(add_overflow_kernel);
}
void Base256uMathTests::CUDA::add::big_overflow() {
	byte_shift_test_macro(add_big_overflow_kernel);
}
void Base256uMathTests::CUDA::add::dst_too_small() {
	byte_shift_test_macro(add_dst_too_small_kernel);
}
void Base256uMathTests::CUDA::add::big_dst_too_small() {
	byte_shift_test_macro(add_big_dst_too_small_kernel);
}
void Base256uMathTests::CUDA::add::zero_for_left_n() {
	byte_shift_test_macro(add_zero_for_left_n_kernel);
}
void Base256uMathTests::CUDA::add::zero_for_right_n() {
	byte_shift_test_macro(add_zero_for_right_n_kernel);
}
void Base256uMathTests::CUDA::add::zero_for_dst_n() {
	byte_shift_test_macro(add_zero_for_dst_n_kernel);
}
void Base256uMathTests::CUDA::add::in_place_ideal_case() {
	byte_shift_test_macro(add_in_place_ideal_case_kernel);
}
void Base256uMathTests::CUDA::add::in_place_big_ideal_case() {
	byte_shift_test_macro(add_in_place_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::add::in_place_left_bigger() {
	byte_shift_test_macro(add_in_place_left_bigger_kernel);
}
void Base256uMathTests::CUDA::add::in_place_left_smaller() {
	byte_shift_test_macro(add_in_place_left_smaller_kernel);
}
void Base256uMathTests::CUDA::add::in_place_big_left_bigger() {
	byte_shift_test_macro(add_in_place_big_left_bigger_kernel);
}
void Base256uMathTests::CUDA::add::in_place_big_left_smaller() {
	byte_shift_test_macro(add_in_place_big_left_smaller_kernel);
}
void Base256uMathTests::CUDA::add::in_place_overflow() {
	byte_shift_test_macro(add_in_place_overflow_kernel);
}
void Base256uMathTests::CUDA::add::in_place_big_overflow() {
	byte_shift_test_macro(add_in_place_big_overflow_kernel);
}
void Base256uMathTests::CUDA::add::in_place_zero_for_left_n() {
	byte_shift_test_macro(add_in_place_zero_for_left_n_kernel);
}
void Base256uMathTests::CUDA::add::in_place_zero_for_right_n() {
	byte_shift_test_macro(add_in_place_zero_for_right_n_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::subtract::test() {
	ideal_case();
	big_ideal_case();
	left_bigger();
	left_smaller();
	big_left_bigger();
	big_left_smaller();
	underflow();
	big_underflow();
	dst_too_small();
	big_dst_too_small();
	zero_for_left_n();
	zero_for_right_n();
	zero_for_dst_n();

	in_place_ideal_case();
	in_place_big_ideal_case();
	in_place_left_bigger();
	in_place_left_smaller();
	in_place_big_left_bigger();
	in_place_big_left_smaller();
	in_place_underflow();
	in_place_big_underflow();
	in_place_zero_for_left_n();
	in_place_zero_for_right_n();
}

__global__
void subtract_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 47462;
	unsigned int right = 36840;
	unsigned int dst = 0,
		answer = left - right;
	auto return_code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void subtract_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 73, 120, 227, 232, 214, 48, 11, 250, 184 };
	unsigned char right[] = { 66, 115, 195, 196, 65, 141, 141, 8, 46 };
	unsigned char dst[9];
	auto return_code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 7, 5, 32, 36, 149, 163, 125, 241, 138 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void subtract_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 9886306996502392208;
	unsigned int right = 2536;
	decltype(left) dst,
		answer = left - right;
	auto return_code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void subtract_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 27639;
	std::size_t right = 15223;
	decltype(right) dst,
		answer = left - right;
	auto return_code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void subtract_big_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 144, 165, 47, 40, 109, 135, 246, 58, 243, 129, 123, 49 };
	unsigned char right[] = { 99, 52, 93, 254, 211, 44, 168, 77, 192 };
	unsigned char dst[12];
	auto return_code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 45, 113, 210, 41, 153, 90, 78, 237, 50, 129, 123, 49 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void subtract_big_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 55, 110, 240, 162, 231, 119, 65, 145, 251 };
	unsigned char right[] = { 99, 148, 120, 205, 172, 215, 125, 26, 14, 0, 0, 0 };
	unsigned char dst[12];
	auto return_code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 212, 217, 119, 213, 58, 160, 195, 118, 237, 0, 0, 0 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void subtract_underflow_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 0;
	unsigned int right = 60331;
	unsigned int dst = 0,
		answer = left - right;
	auto return_code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::FLOW) {
		*code = 2;
	}
}
__global__
void subtract_big_underflow_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char right[] = { 111, 198, 135, 255, 25, 61, 175, 193, 75 };
	unsigned char dst[9];
	auto return_code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 145, 57, 120, 0, 230, 194, 80, 62, 180 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::FLOW) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void subtract_dst_too_small_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 3971310598;
	unsigned int right = 3473639866;
	unsigned short dst = 0,
		answer = left - right;
	auto return_code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}
__global__
void subtract_big_dst_too_small_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 220, 139, 205, 222, 88, 100, 192, 105, 59, 106, 0, 179 };
	unsigned char right[] = { 206, 72, 107, 123, 155, 149, 24, 175, 134 };
	unsigned char dst[9];
	auto return_code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 14, 67, 98, 99, 189, 206, 167, 186, 180, 105, 0, 179 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (answer[i] != dst[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void subtract_zero_for_left_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 1212121212121;
	unsigned short right = 29332;
	unsigned short dst = 0;
	auto return_code = Base256uMath::subtract(&left, 0, &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	unsigned short answer = ~right + 1;
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::FLOW) {
		*code = 2;
	}
}
__global__
void subtract_zero_for_right_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 85985313099138956;
	unsigned int right = 1929328482;
	std::size_t dst,
		answer = left;
	auto return_code = Base256uMath::subtract(&left, sizeof(left), &right, 0, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void subtract_zero_for_dst_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 1937650443;
	unsigned short right = 5232;
	unsigned int dst = 69,
		answer = dst;
	auto return_code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right), &dst, 0);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}

__global__
void subtract_in_place_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 47462;
	unsigned int right = 36840;
	unsigned int answer = left - right;
	auto return_code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void subtract_in_place_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 73, 120, 227, 232, 214, 48, 11, 250, 184 };
	unsigned char right[] = { 66, 115, 195, 196, 65, 141, 141, 8, 46 };
	unsigned char dst[9];
	auto return_code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 7, 5, 32, 36, 149, 163, 125, 241, 138 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void subtract_in_place_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 9886306996502392208;
	unsigned int right = 2536;
	std::size_t answer = left - right;
	auto return_code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void subtract_in_place_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 27639;
	std::size_t right = 15223;
	decltype(left) answer = left - right;
	auto return_code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void subtract_in_place_big_left_bigger_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 144, 165, 47, 40, 109, 135, 246, 58, 243, 129, 123, 49 };
	unsigned char right[] = { 99, 52, 93, 254, 211, 44, 168, 77, 192 };
	auto return_code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 45, 113, 210, 41, 153, 90, 78, 237, 50, 129, 123, 49 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void subtract_in_place_big_left_smaller_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 55, 110, 240, 162, 231, 119, 65, 145, 251 };
	unsigned char right[] = { 99, 148, 120, 205, 172, 215, 125, 26, 14, 0, 0, 0 };
	auto return_code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 212, 217, 119, 213, 58, 160, 195, 118, 237, 0, 0, 0 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void subtract_in_place_underflow_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 0;
	unsigned int right = 60331;
	decltype(left) answer = left - right;
	auto return_code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::FLOW) {
		*code = 2;
	}
}
__global__
void subtract_in_place_big_underflow_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char right[] = { 111, 198, 135, 255, 25, 61, 175, 193, 75 };
	auto return_code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 145, 57, 120, 0, 230, 194, 80, 62, 180 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::FLOW) {
		*code = sizeof(left) + 1;
	}
}
__global__
void subtract_in_place_zero_for_left_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 1212121212121;
	unsigned short right = 29332;
	decltype(left) answer = 1212121212121;
	auto return_code = Base256uMath::subtract(&left, 0, &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void subtract_in_place_zero_for_right_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t left = 85985313099138956;
	unsigned int right = 1929328482;
	decltype(left) answer = 85985313099138956;
	auto return_code = Base256uMath::subtract(&left, sizeof(left), &right, 0);
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}

void Base256uMathTests::CUDA::subtract::ideal_case() {
	byte_shift_test_macro(subtract_ideal_case_kernel);
}
void Base256uMathTests::CUDA::subtract::big_ideal_case() {
	byte_shift_test_macro(subtract_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::subtract::left_bigger() {
	byte_shift_test_macro(subtract_left_bigger_kernel);
}
void Base256uMathTests::CUDA::subtract::left_smaller() {
	byte_shift_test_macro(subtract_left_smaller_kernel);
}
void Base256uMathTests::CUDA::subtract::big_left_bigger() {
	byte_shift_test_macro(subtract_big_left_bigger_kernel);
}
void Base256uMathTests::CUDA::subtract::big_left_smaller() {
	byte_shift_test_macro(subtract_big_left_smaller_kernel);
}
void Base256uMathTests::CUDA::subtract::underflow() {
	byte_shift_test_macro(subtract_underflow_kernel);
}
void Base256uMathTests::CUDA::subtract::big_underflow() {
	byte_shift_test_macro(subtract_big_underflow_kernel);
}
void Base256uMathTests::CUDA::subtract::dst_too_small() {
	byte_shift_test_macro(subtract_dst_too_small_kernel);
}
void Base256uMathTests::CUDA::subtract::big_dst_too_small() {
	byte_shift_test_macro(subtract_big_dst_too_small_kernel);
}
void Base256uMathTests::CUDA::subtract::zero_for_left_n() {
	byte_shift_test_macro(subtract_zero_for_left_n_kernel);
}
void Base256uMathTests::CUDA::subtract::zero_for_right_n() {
	byte_shift_test_macro(subtract_zero_for_right_n_kernel);
}
void Base256uMathTests::CUDA::subtract::zero_for_dst_n() {
	byte_shift_test_macro(subtract_zero_for_dst_n_kernel);
}

void Base256uMathTests::CUDA::subtract::in_place_ideal_case() {
	byte_shift_test_macro(subtract_in_place_ideal_case_kernel);
}
void Base256uMathTests::CUDA::subtract::in_place_big_ideal_case() {
	byte_shift_test_macro(subtract_in_place_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::subtract::in_place_left_bigger() {
	byte_shift_test_macro(subtract_in_place_left_bigger_kernel);
}
void Base256uMathTests::CUDA::subtract::in_place_left_smaller() {
	byte_shift_test_macro(subtract_in_place_left_smaller_kernel);
}
void Base256uMathTests::CUDA::subtract::in_place_big_left_bigger() {
	byte_shift_test_macro(subtract_in_place_big_left_bigger_kernel);
}
void Base256uMathTests::CUDA::subtract::in_place_big_left_smaller() {
	byte_shift_test_macro(subtract_in_place_big_left_smaller_kernel);
}
void Base256uMathTests::CUDA::subtract::in_place_underflow() {
	byte_shift_test_macro(subtract_in_place_underflow_kernel);
}
void Base256uMathTests::CUDA::subtract::in_place_big_underflow() {
	byte_shift_test_macro(subtract_in_place_big_underflow_kernel);
}
void Base256uMathTests::CUDA::subtract::in_place_zero_for_left_n() {
	byte_shift_test_macro(subtract_in_place_zero_for_left_n_kernel);
}
void Base256uMathTests::CUDA::subtract::in_place_zero_for_right_n() {
	byte_shift_test_macro(subtract_in_place_zero_for_right_n_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::log2::test() {
	ideal_case();
	big_ideal_case();
	src_is_zero();
	src_n_zero();
	dst_n_zero();
}

__global__
void log2_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 0b00010000000100000;
	std::size_t dst = 0;
	auto return_code = Base256uMath::log2(&src, sizeof(src), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 13) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void log2_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
	unsigned char dst[sizeof(std::size_t) + 1];
	auto return_code = Base256uMath::log2(src, sizeof(src), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	if (dst[0] != 64) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void log2_src_is_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 0;
	std::size_t dst = 1234567890;
	auto return_code = Base256uMath::log2(&src, sizeof(src), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 1234567890) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::DIVIDE_BY_ZERO) {
		*code = 2;
	}
}
__global__
void log2_src_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 1337;
	std::size_t dst = 1234567890;
	auto return_code = Base256uMath::log2(&src, 0, &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 1234567890) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::DIVIDE_BY_ZERO) {
		*code = 2;
	}
}
__global__
void log2_dst_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int src = 1337;
	std::size_t dst = 1234567890;
	auto return_code = Base256uMath::log2(&src, sizeof(src), &dst, 0);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 1234567890) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 2;
	}
}

void Base256uMathTests::CUDA::log2::ideal_case() {
	byte_shift_test_macro(log2_ideal_case_kernel);
}
void Base256uMathTests::CUDA::log2::big_ideal_case() {
	byte_shift_test_macro(log2_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::log2::src_is_zero() {
	byte_shift_test_macro(log2_src_is_zero_kernel);
}
void Base256uMathTests::CUDA::log2::src_n_zero() {
	byte_shift_test_macro(log2_src_n_zero_kernel);
}
void Base256uMathTests::CUDA::log2::dst_n_zero() {
	byte_shift_test_macro(log2_dst_n_zero_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::log256::test() {
	ideal_case();
	big_ideal_case();
	src_is_zero();
	src_n_zero();
}

__global__
void log256_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 16289501482060108362;
	std::size_t dst;
	auto return_code = Base256uMath::log256(&src, sizeof(src), &dst);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 7) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void log256_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char src[] = { 139, 46, 187, 204, 123, 55, 217, 147, 102, 0 };
	std::size_t dst;
	auto return_code = Base256uMath::log256(src, sizeof(src), &dst);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 8) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void log256_src_is_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t zero = 0;
	std::size_t dst = 1467;
	auto return_code = Base256uMath::log256(&zero, sizeof(zero), &dst);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 1467) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::DIVIDE_BY_ZERO) {
		*code = 2;
	}
}
__global__
void log256_src_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	std::size_t src = 230841808201;
	std::size_t dst = 1337;
	auto return_code = Base256uMath::log256(&src, 0, &dst);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	if (dst != 1337) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::DIVIDE_BY_ZERO) {
		*code = 2;
	}
}

void Base256uMathTests::CUDA::log256::ideal_case() {
	byte_shift_test_macro(log256_ideal_case_kernel);
}
void Base256uMathTests::CUDA::log256::big_ideal_case() {
	byte_shift_test_macro(log256_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::log256::src_is_zero() {
	byte_shift_test_macro(log256_src_is_zero_kernel);
}
void Base256uMathTests::CUDA::log256::src_n_zero() {
	byte_shift_test_macro(log256_src_n_zero_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::multiply::test() {
	ideal_case();
	big_ideal_case();
	multiply_zero();
	multiply_one();
	left_n_greater_than_right_n();
	left_n_less_than_right_n();
	dst_n_less_than_both();
	dst_n_zero();

	in_place_ideal_case();
	in_place_big_ideal_case();
	in_place_multiply_zero();
	in_place_multiply_one();
	in_place_left_n_greater_than_right_n();
	in_place_left_n_less_than_right_n();
}

__global__
void multiply_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned short left = 25603;
	unsigned short right = 6416;
	unsigned int dst;
	auto return_code = Base256uMath::multiply(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	unsigned int answer = left * right;
	if (dst != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void multiply_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 138, 204, 16, 163, 74, 68, 68, 39, 2 };
	unsigned char right[] = { 72, 80, 180, 160, 32, 125, 248, 160, 102 };
	unsigned char dst[18];
	auto return_code = Base256uMath::multiply(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 208, 166, 172, 45, 217, 162, 152, 6, 155, 229, 217, 94, 0, 248, 212, 255, 220, 0 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void multiply_multiply_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 253, 112, 1, 250, 242, 174, 77, 35, 242 };
	unsigned char right[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char dst[9];
	auto return_code = Base256uMath::multiply(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != 0) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void multiply_multiply_one_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 1, 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char right[] = { 14, 92, 130, 38, 174, 216, 149, 5, 169 };
	unsigned char dst[9];
	auto return_code = Base256uMath::multiply(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != right[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void multiply_left_n_greater_than_right_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 217, 86, 117, 188, 220, 28, 236, 105, 119 };
	unsigned int right = 707070;
	unsigned char dst[sizeof(left) + sizeof(right)];
	auto return_code = Base256uMath::multiply(left, sizeof(left), &right, sizeof(right), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 78, 140, 22, 130, 144, 79, 141, 155, 222, 91, 8, 5, 0 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void multiply_left_n_less_than_right_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 707070;
	unsigned char right[] = { 217, 86, 117, 188, 220, 28, 236, 105, 119 };
	unsigned char dst[sizeof(left) + sizeof(right)];
	auto return_code = Base256uMath::multiply(&left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 78, 140, 22, 130, 144, 79, 141, 155, 222, 91, 8, 5, 0 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void multiply_dst_n_less_than_both_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	// if dst_n < left_n <=> right_n, then the answer will be truncated, and 
	// return the truncated error code

	unsigned char left[] = { 217, 86, 117, 188, 220, 28, 236, 105, 119 };
	unsigned int right = 707070;
	unsigned char dst[sizeof(right) - 1];
	auto return_code = Base256uMath::multiply(left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	unsigned char answer[] = { 78, 140, 22, 130, 144, 79, 141, 155, 222, 91, 8, 5 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void multiply_dst_n_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	// if dst_n is zero, then nothing will happen and the truncated error code will be returned

	unsigned char left[] = { 101, 165, 53, 155, 99, 101, 83, 23, 42 };
	unsigned short right = 35;
	unsigned char dst[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	auto return_code = Base256uMath::multiply(left, sizeof(left), &right, sizeof(right), dst, 0);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != i) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = sizeof(dst) + 1;
	}
}
__global__
void multiply_in_place_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned int left = 25603;
	unsigned short right = 6416;
	unsigned int answer = left * right;
	auto return_code = Base256uMath::multiply(&left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	if (left != answer) {
		*code = 1;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 2;
	}
}
__global__
void multiply_in_place_big_ideal_case_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 138, 204, 16, 163, 74, 68, 68, 39, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char right[] = { 72, 80, 180, 160, 32, 125, 248, 160, 102 };
	auto return_code = Base256uMath::multiply(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 208, 166, 172, 45, 217, 162, 152, 6, 155, 229, 217, 94, 0, 248, 212, 255, 220, 0 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void multiply_in_place_multiply_zero_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 253, 112, 1, 250, 242, 174, 77, 35, 242 };
	unsigned char right[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	auto return_code = Base256uMath::multiply(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != 0) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void multiply_in_place_multiply_one_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 1, 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char right[] = { 14, 92, 130, 38, 174, 216, 149, 5, 169 };
	auto return_code = Base256uMath::multiply(left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != right[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void multiply_in_place_left_n_greater_than_right_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 217, 86, 117, 188, 220, 28, 236, 105, 119, 0, 0, 0, 0 };
	unsigned int right = 707070;
	
	auto return_code = Base256uMath::multiply(left, sizeof(left), &right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 78, 140, 22, 130, 144, 79, 141, 155, 222, 91, 8, 5, 0 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}
__global__
void multiply_in_place_left_n_less_than_right_n_kernel(int* code, void* output, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 254, 201, 10 };
	unsigned char right[] = { 217, 86, 117, 188, 220, 28, 236, 105, 119 };
	auto return_code = Base256uMath::multiply(&left, sizeof(left), right, sizeof(right));
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	unsigned char answer[] = { 78, 140, 22, 130, 144, 79, 141, 155, 222, 91, 8, 5, 0 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = sizeof(left) + 1;
	}
}

void Base256uMathTests::CUDA::multiply::ideal_case() {
	byte_shift_test_macro(multiply_ideal_case_kernel);
}
void Base256uMathTests::CUDA::multiply::big_ideal_case() {
	byte_shift_test_macro(multiply_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::multiply::multiply_zero() {
	byte_shift_test_macro(multiply_multiply_zero_kernel);
}
void Base256uMathTests::CUDA::multiply::multiply_one() {
	byte_shift_test_macro(multiply_multiply_one_kernel);
}
void Base256uMathTests::CUDA::multiply::left_n_greater_than_right_n() {
	byte_shift_test_macro(multiply_left_n_greater_than_right_n_kernel);
}
void Base256uMathTests::CUDA::multiply::left_n_less_than_right_n() {
	byte_shift_test_macro(multiply_left_n_less_than_right_n_kernel);
}
void Base256uMathTests::CUDA::multiply::dst_n_less_than_both() {
	byte_shift_test_macro(multiply_dst_n_less_than_both_kernel);
}
void Base256uMathTests::CUDA::multiply::dst_n_zero() {
	byte_shift_test_macro(multiply_dst_n_zero_kernel);
}
void Base256uMathTests::CUDA::multiply::in_place_ideal_case() {
	byte_shift_test_macro(multiply_in_place_ideal_case_kernel);
}
void Base256uMathTests::CUDA::multiply::in_place_big_ideal_case() {
	byte_shift_test_macro(multiply_in_place_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::multiply::in_place_multiply_zero() {
	byte_shift_test_macro(multiply_in_place_multiply_zero_kernel);
}
void Base256uMathTests::CUDA::multiply::in_place_multiply_one() {
	byte_shift_test_macro(multiply_in_place_multiply_one_kernel);
}
void Base256uMathTests::CUDA::multiply::in_place_left_n_greater_than_right_n() {
	byte_shift_test_macro(multiply_in_place_left_n_greater_than_right_n_kernel);
}
void Base256uMathTests::CUDA::multiply::in_place_left_n_less_than_right_n() {
	byte_shift_test_macro(multiply_in_place_left_n_less_than_right_n_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::divide::test() {
	ideal_case();
	big_ideal_case();
	left_is_zero();
	left_n_zero();
	right_is_zero();
	right_n_zero();
	left_n_less();
	dst_n_less();
	dst_n_zero();
	remainder_n_less();
	remainder_n_zero();
	in_place_ideal_case();
	in_place_big_ideal_case();
	in_place_left_is_zero();
	in_place_left_n_zero();
	in_place_right_is_zero();
	in_place_right_n_zero();
	in_place_left_n_less();
	in_place_remainder_n_less();
	in_place_remainder_n_zero();
}

__global__
void divide_ideal_case_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	std::size_t left = 0b11001000000000111111010,
		right = 0b1100100010000,
		dst, mod,
		answer = left / right,
		answer_mod = left % right;
	auto return_code = Base256uMath::divide(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst),
		&mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, &dst, sizeof(dst));
	memset(output_mod, 0, *size);
	memcpy(output_mod, &mod, sizeof(mod));
	if (dst != answer) {
		*code = 1;
	}
	else if (mod != answer_mod) {
		*code = 2;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 999;
	}
}
__global__
void divide_big_ideal_case_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	unsigned char right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	unsigned char dst[10];
	unsigned char mod[10];
	unsigned char answer[] = { 91, 1 };
	unsigned char answer_mod[] = { 101, 110, 228, 117, 44, 39, 11, 193, 83 };
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	for (unsigned char i = 0; i < sizeof(answer); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	for (unsigned char i = 0; i < sizeof(answer_mod); i++) {
		if (mod[i] != answer_mod[i]) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 999;
	}
}
__global__
void divide_left_is_zero_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// if left is zero, then dst and mod become all zeros

	unsigned char left[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	unsigned char dst[sizeof(left)];
	unsigned char mod[sizeof(left)];
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != 0) {
			*code = i + 1;
			return;
		}
		if (mod[i] != 0) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 999;
	}
}
__global__
void divide_right_is_zero_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// if right is zero, then nothing happens and a division by zero error code is returned

	unsigned char left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	unsigned char right = 0;
	unsigned char dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	unsigned char mod[] = { 20, 21, 22, 23, 24, 25, 26, 27, 28 };
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		&right, sizeof(right),
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	for (unsigned char i = 0; i < sizeof(left); i++) {
		assert(left[i] == i);
		if (dst[i] != (i + 10)) {
			*code = i + 1;
			return;
		}
		if (mod[i] != (i + 20)) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::DIVIDE_BY_ZERO) {
		*code = 999;
	}
}
__global__
void divide_right_n_zero_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// if right_n is zero, then right is assumed to be all zeros and the function
	// behaves as if right were zero.

	unsigned char left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	unsigned char right = 5;
	unsigned char dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	unsigned char mod[] = { 20, 21, 22, 23, 24, 25, 26, 27, 28 };
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		&right, 0,
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	for (unsigned char i = 0; i < sizeof(left); i++) {
		assert(left[i] == i);
		if (dst[i] != (i + 10)) {
			*code = i + 1;
			return;
		}
		if (mod[i] != (i + 20)) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::DIVIDE_BY_ZERO) {
		*code = 999;
	}
}
__global__
void divide_left_n_less_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// left > right, but 0 < left_n < right_n. 
	unsigned char left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 116 };
	unsigned char right[] = { 182, 193, 139, 54, 147, 128, 223, 45, 0, 0 };
	unsigned char dst[sizeof(left)];
	unsigned char mod[sizeof(left)];
	assert(sizeof(left) < sizeof(right));
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	unsigned char answer[] = { 139, 2, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char answer_mod[] = { 69, 102, 238, 175, 91, 120, 162, 41, 0 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	for (unsigned char i = 0; i < sizeof(mod); i++) {
		if (mod[i] != answer_mod[i]) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 999;
	}
}
__global__
void divide_left_n_zero_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// if left_n is zero, then it is assumed to be all zeros.
	// that means dst and mod will be all zeros

	unsigned char left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 116 };
	unsigned char right[] = { 182, 193, 139, 54, 147, 128, 223, 45, 0, 0 };
	unsigned char dst[sizeof(left)];
	unsigned char mod[sizeof(left)];
	auto return_code = Base256uMath::divide(
		left, 0,
		right, sizeof(right),
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != 0) {
			*code = i + 1;
			return;
		}
		if (mod[i] != 0) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 999;
	}
}
__global__
void divide_dst_n_less_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// If dst_n is less than left_n, truncation might occur.
	// To guarantee no truncation, dst should be the same size as left.
	// If mod is of adequate size, then it should yield the correct answer.
	// In any case, truncation or no, the function should return the truncated warning code.

	unsigned char left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	unsigned char right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139, 0 };
	unsigned char dst[1]; // the answer has 2 significant characters, so this will demonstrate truncation
	unsigned char mod[sizeof(left)];
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	unsigned char answer[] = { 91, 1 };
	unsigned char answer_mod[] = { 101, 110, 228, 117, 44, 39, 11, 193, 83 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	for (unsigned char i = 0; i < sizeof(answer_mod); i++) {
		if (mod[i] != answer_mod[i]) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 999;
	}
}
__global__
void divide_dst_n_zero_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// if dst_n is zero, truncation is guaranteed and nothing will happen.
	// Even if mod is of correct size, it will not be changed.
	// The function should return the truncated warning code.

	unsigned char left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	unsigned char right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	unsigned char dst[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	unsigned char mod[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		dst, 0,
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != i) {
			*code = i + 1;
			return;
		}
		if (mod[i] != (10 + i)) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 999;
	}
}
__global__
void divide_remainder_n_less_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// if remainder_n is less than left_n, then left_n is treated as if it were
	// of size remainder_n. Guarantees truncation.

	unsigned char left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	unsigned char right[] = { 182, 193, 139, 54, 147, 128, 0, 0, 0, 0 }; //, 88, 139, 0 };
	unsigned char dst[sizeof(left)];
	unsigned char mod[7]; // the remainder has 9 significant chars
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	unsigned char answer[] = { 250, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char answer_mod[] = { 91, 30, 23, 149, 189, 75, 0 };
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	for (unsigned char i = 0; i < sizeof(mod); i++) {
		if (mod[i] != answer_mod[i]) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 999;
	}
}
__global__
void divide_remainder_n_zero_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// if remainder_n is zero, then the function behaves as if left_n
	// was zero. Returns a truncated error code.

	unsigned char left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	unsigned char right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	unsigned char dst[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	unsigned char mod[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst),
		mod, 0
	);
	memset(output, 0, *size);
	memcpy(output, dst, sizeof(dst));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	for (unsigned char i = 0; i < sizeof(dst); i++) {
		if (dst[i] != 0) {
			*code = i + 1;
			return;
		}
		if (mod[i] != (10 + i)) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::TRUNCATED) {
		*code = 999;
	}
}
__global__
void divide_in_place_ideal_case_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	std::size_t left = 0b11001000000000111111010,
		right = 0b1100100010000,
		mod,
		answer = left / right,
		answer_mod = left % right;
	auto return_code = Base256uMath::divide(
		&left, sizeof(left),
		&right, sizeof(right),
		&mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, &left, sizeof(left));
	memset(output_mod, 0, *size);
	memcpy(output_mod, &mod, sizeof(mod));
	if (left != answer) {
		*code = 1;
	}
	else if (mod != answer_mod) {
		*code = 2;
	}
	else if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 999;
	}
}
__global__
void divide_in_place_big_ideal_case_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	unsigned char left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	unsigned char right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	unsigned char mod[10];
	unsigned char answer[] = { 91, 1 };
	unsigned char answer_mod[] = { 101, 110, 228, 117, 44, 39, 11, 193, 83 };
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	for (unsigned char i = 0; i < sizeof(answer); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	for (unsigned char i = 0; i < sizeof(answer_mod); i++) {
		if (mod[i] != answer_mod[i]) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 999;
	}
}
__global__
void divide_in_place_left_is_zero_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// if left is zero, then left will be all zeros and mod will be a copy of left.
	// 

	unsigned char left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	unsigned char right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	unsigned char mod[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != 0) {
			*code = i + 1;
			return;
		}
		if (mod[i] != i) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 999;
	}
}
__global__
void divide_in_place_left_n_zero_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// if left_n is zero, then left and mod are untouched.

	unsigned char left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	unsigned char right[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	unsigned char mod[] = { 100, 101, 102, 103, 104, 105, 106, 107, 108 };
	auto return_code = Base256uMath::divide(
		left, 0,
		right, sizeof(right),
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != i) {
			*code = i + 1;
			return;
		}
		if (mod[i] != (i + 100)) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 999;
	}
}
__global__
void divide_in_place_right_is_zero_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// if right is zero, then nothing happens and a division by zero error code is returned

	unsigned char left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	unsigned char right = 0;
	unsigned char mod[] = { 20, 21, 22, 23, 24, 25, 26, 27, 28 };
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		&right, sizeof(right),
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != i) {
			*code = i + 1;
			return;
		}
		if (mod[i] != (i + 20)) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::DIVIDE_BY_ZERO) {
		*code = 999;
	}
}
__global__
void divide_in_place_right_n_zero_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// if right_n is zero, then right is assumed to be all zeros and the function
	// behaves as if right were zero.

	unsigned char left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	unsigned char right = 5;
	unsigned char mod[] = { 20, 21, 22, 23, 24, 25, 26, 27, 28 };
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		&right, 0,
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != i) {
			*code = i + 1;
			return;
		}
		if (mod[i] != (i + 20)) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::DIVIDE_BY_ZERO) {
		*code = 999;
	}
}
__global__
void divide_in_place_left_n_less_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// left > right, but 0 < left_n < right_n. 
	unsigned char left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 116 };
	unsigned char right[] = { 182, 193, 139, 54, 147, 128, 223, 45, 0, 0 };
	unsigned char mod[sizeof(left)];
	assert(sizeof(left) < sizeof(right));
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	unsigned char answer[] = { 139, 2, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char answer_mod[] = { 69, 102, 238, 175, 91, 120, 162, 41, 0 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	for (unsigned char i = 0; i < sizeof(mod); i++) {
		if (mod[i] != answer_mod[i]) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 999;
	}
}
__global__
void divide_in_place_remainder_n_less_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// if remainder_n is less than left_n, then left_n is treated as if it were
	// of size remainder_n.

	unsigned char left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	unsigned char right[] = { 182, 193, 139, 54, 147, 128, 0, 0, 0, 0 }; //, 88, 139, 0 };
	unsigned char mod[7]; // the remainder has 9 significant chars
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		mod, sizeof(mod)
	);
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	unsigned char answer[] = { 250, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char answer_mod[] = { 91, 30, 23, 149, 189, 75, 0 };
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != answer[i]) {
			*code = i + 1;
			return;
		}
	}
	for (unsigned char i = 0; i < sizeof(mod); i++) {
		if (mod[i] != answer_mod[i]) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 999;
	}
}
__global__
void divide_in_place_remainder_n_zero_kernel(int* code, void* output, void* output_mod, std::size_t* size) {
	*code = 0;
	// if remainder_n is zero, then the function behaves as if left_n
	// was zero. 

	unsigned char left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	unsigned char right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	unsigned char mod[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
	auto return_code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		mod, 0
	);
	memset(output, 0, *size);
	memcpy(output, left, sizeof(left));
	memset(output_mod, 0, *size);
	memcpy(output_mod, mod, sizeof(mod));
	for (unsigned char i = 0; i < sizeof(left); i++) {
		if (left[i] != 0) {
			*code = i + 1;
			return;
		}
		if (mod[i] != (10 + i)) {
			*code = i + 1 + 50;
			return;
		}
	}
	if (return_code != Base256uMath::ErrorCodes::OK) {
		*code = 999;
	}
}

#define divide_test_macro(kernel_name) \
int code = -1; \
int* d_code; \
auto err = cudaMalloc(&d_code, sizeof(int)); \
cudaMalloc_check_macro(err); \
std::size_t size = 15; \
unsigned char result[size]; \
void* d_result; \
err = cudaMalloc(&d_result, size); \
cudaMalloc_check_macro(err); \
unsigned char result_mod[size]; \
void* d_result_mod; \
err = cudaMalloc(&d_result_mod, size); \
cudaMalloc_check_macro(err); \
std::size_t* d_size; \
err = cudaMalloc(&d_size, sizeof(std::size_t)); \
cudaMalloc_check_macro(err); \
err = cudaMemcpy(d_size, &size, sizeof(std::size_t), cudaMemcpyHostToDevice); \
cudaMemcpy_check_macro(err); \
KERNEL_CALL4(kernel_name, d_code, d_result, d_result_mod, d_size); \
err = cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost); \
cudaMemcpy_check_macro(err); \
err = cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost); \
cudaMemcpy_check_macro(err); \
err = cudaMemcpy(result_mod, d_result_mod, size, cudaMemcpyDeviceToHost); \
cudaMemcpy_check_macro(err); \
if (code != 0) { \
	std::cout << "code: " << std::to_string(code) << std::endl; \
	std::cout << "result: "; \
	for (std::size_t i = 0; i < size; i++) { \
		std::cout << std::to_string(reinterpret_cast<unsigned char*>(result)[i]) << " "; \
	} \
	std::cout << "remainder: "; \
	for (std::size_t i = 0; i < size; i++) { \
		std::cout << std::to_string(reinterpret_cast<unsigned char*>(result_mod)[i]) << " "; \
	} \
	std::cout << std::endl; \
	assert(code == 0); \
} \
cudaFree(d_code); \
cudaFree(d_result); \
cudaFree(d_result_mod); \
cudaFree(d_size);

void Base256uMathTests::CUDA::divide::ideal_case() {
	divide_test_macro(divide_ideal_case_kernel);
}
void Base256uMathTests::CUDA::divide::big_ideal_case() {
	divide_test_macro(divide_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::divide::left_is_zero() {
	divide_test_macro(divide_left_is_zero_kernel);
}
void Base256uMathTests::CUDA::divide::left_n_zero() {
	divide_test_macro(divide_left_n_zero_kernel);
}
void Base256uMathTests::CUDA::divide::right_is_zero() {
	divide_test_macro(divide_right_is_zero_kernel);
}
void Base256uMathTests::CUDA::divide::right_n_zero() {
	divide_test_macro(divide_right_n_zero_kernel);
}
void Base256uMathTests::CUDA::divide::left_n_less() {
	divide_test_macro(divide_left_n_less_kernel);
}
void Base256uMathTests::CUDA::divide::dst_n_less() {
	divide_test_macro(divide_dst_n_less_kernel);
}
void Base256uMathTests::CUDA::divide::dst_n_zero() {
	divide_test_macro(divide_dst_n_zero_kernel);
}
void Base256uMathTests::CUDA::divide::remainder_n_less() {
	divide_test_macro(divide_remainder_n_less_kernel);
}
void Base256uMathTests::CUDA::divide::remainder_n_zero() {
	divide_test_macro(divide_remainder_n_zero_kernel);
}
void Base256uMathTests::CUDA::divide::in_place_ideal_case() {
	divide_test_macro(divide_in_place_ideal_case_kernel);
}
void Base256uMathTests::CUDA::divide::in_place_big_ideal_case() {
	divide_test_macro(divide_in_place_big_ideal_case_kernel);
}
void Base256uMathTests::CUDA::divide::in_place_left_is_zero() {
	divide_test_macro(divide_in_place_left_is_zero_kernel);
}
void Base256uMathTests::CUDA::divide::in_place_left_n_zero() {
	divide_test_macro(divide_in_place_left_n_zero_kernel);
}
void Base256uMathTests::CUDA::divide::in_place_right_is_zero() {
	divide_test_macro(divide_in_place_right_is_zero_kernel);
}
void Base256uMathTests::CUDA::divide::in_place_right_n_zero() {
	divide_test_macro(divide_in_place_right_n_zero_kernel);
}
void Base256uMathTests::CUDA::divide::in_place_left_n_less() {
	divide_test_macro(divide_in_place_left_n_less_kernel);
}
void Base256uMathTests::CUDA::divide::in_place_remainder_n_less() {
	divide_test_macro(divide_in_place_remainder_n_less_kernel);
}
void Base256uMathTests::CUDA::divide::in_place_remainder_n_zero() {
	divide_test_macro(divide_in_place_remainder_n_zero_kernel);
}

// ===================================================================================

void Base256uMathTests::CUDA::divide_no_mod::test() {}

// ===================================================================================

void Base256uMathTests::CUDA::mod::test() {}

#endif