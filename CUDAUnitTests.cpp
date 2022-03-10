#ifdef __NVCC__
#include "CUDAUnitTests.h"
#include "Base256uMath.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define KERNEL_CALL(func_name, code_ptr) func_name<<<1,1>>>(code_ptr); cudaDeviceSynchronize()
#define KERNEL_CALL2(func_name, code_ptr, ptr1) func_name<<<1,1>>>(code_ptr, ptr1); cudaDeviceSynchronize()
#define KERNEL_CALL3(func_name, code_ptr, ptr1, ptr2) func_name<<<1,1>>>(code_ptr, ptr1, ptr2); cudaDeviceSynchronize()
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
	assert(sizeof(src) * 8 > by);
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
	assert(sizeof(src) * 8 > by);
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