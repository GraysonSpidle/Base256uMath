#define BASE256UMATH_SUPPRESS_TRUNCATED_CODE 0
#define BASE256UMATH_FAST_OPERATORS 1
#include "Base256uMath.h"
#include "UnitTests.h"
#ifdef __NVCC__
#include "CUDAUnitTests.h"
#endif
#include <iostream>

int main() {
#ifndef _DEBUG
	std::cout << "Make sure you're in debug mode to perform tests!" << std::endl;
#endif

	Base256uMathTests::test_unit_tests();
#ifdef __NVCC__
	Base256uMathTests::CUDA::test_unit_tests();
#endif
	return 0;
}