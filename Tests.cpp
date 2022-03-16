#include "Base256uMath.h"
#include "UnitTests.h"
#include "PerformanceTests.h"
#ifdef __CUDACC__
#include "CUDAUnitTests.h"
#endif
#include <iostream>
#include <string>

#ifndef MIN(a,b)
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

constexpr std::size_t KB = 1024;
constexpr std::size_t MB = 1024 * KB;
constexpr std::size_t GB = 1024 * MB;

int main() {
#ifndef _DEBUG
	std::cout << "Make sure you're in debug mode to perform tests!" << std::endl;
#endif

#ifdef BASE256UMATH_FAST_OPERATORS
	std::cout << "FAST_OPERATORS enabled" << std::endl;
#endif
	std::cout << "Running unit tests..." << std::endl;
	Base256uMathTests::test_unit_tests();
#ifdef __CUDACC__
	std::cout << "CUDA detected, running CUDA unit tests..." << std::endl;
	Base256uMathTests::CUDA::test_unit_tests();
#endif

	std::cout << "All unit tests passed." << std::endl << std::endl;
	std::cout << "Running performance tests..." << std::endl;
	Base256uMathTests::Performance::test(100, 1 * MB);
	std::cout << "Performance tests concluded." << std::endl;
	return 0;
}