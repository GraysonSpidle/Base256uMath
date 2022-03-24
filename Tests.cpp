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

const int _tab64[64] = {
	63,  0, 58,  1, 59, 47, 53,  2,
	60, 39, 48, 27, 54, 33, 42,  3,
	61, 51, 37, 40, 49, 18, 28, 20,
	55, 30, 34, 11, 43, 14, 22,  4,
	62, 57, 46, 52, 38, 26, 32, 41,
	50, 36, 17, 19, 29, 10, 13, 21,
	56, 45, 25, 31, 35, 16,  9, 12,
	44, 24, 15,  8, 23,  7,  6,  5 };

int _log2(std::size_t value)
{
	value |= value >> 1;
	value |= value >> 2;
	value |= value >> 4;
	value |= value >> 8;
	value |= value >> 16;
	value |= value >> 32;
	return _tab64[((std::size_t)((value - (value >> 1)) * 0x07EDD5E59A4E28C2)) >> 58];
}

/* Divide and conquer
- Half the numbers in size until one of them is <= sizeof(std::size_t)
- If both are <= sizeof(std::size_t) in size, then do regular multiplication
- If only 1 number is <= sizeof(std::size_t) in size then do some special thing



*/


inline void split_at256(const std::size_t& num, const std::size_t& n, std::size_t* high, std::size_t* low) {
	*low = num & (((std::size_t)1 << (n << 3)) - 1);
	*high = (num ^ *low) >> (n << 3);
}

void karatsuba_recurse256(const size_t& left, const size_t& right, size_t* dst) {
	if (left <= 255 || right <= 255) {
		*dst = left * right;
		return;
	}

	std::size_t m;
	switch (Base256uMath::compare(&left, sizeof(left), &right, sizeof(right))) {
	case 0:
	case -1:
		Base256uMath::log256(&left, sizeof(left), &m);
		break;
	case 1:
		Base256uMath::log256(&right, sizeof(right), &m);
		break;
	default:
		abort();
	}

	std::size_t temp0, temp1, z;

	// z1
	// low1 + high1
	Base256uMath::add(
		&left,
		sizeof(left) >> 1,
		reinterpret_cast<const unsigned char*>(&left) + (sizeof(left) - (m >> 8)),
		sizeof(left) - (m >> 8),
		&temp0,
		sizeof(temp0)
	);
	// low2 + high2
	Base256uMath::add(
		&right,
		sizeof(right) >> 1,
		reinterpret_cast<const unsigned char*>(&right) + (sizeof(right) - (m >> 8)),
		sizeof(right) - (m >> 8),
		&temp1, sizeof(temp1)
	);
	karatsuba_recurse256(temp0, temp1, &z);
	*dst += z << (m >> 8);

	// z2
	split_at256(left, (m >> 8), &temp0, &z);
	split_at256(right, (m >> 8), &temp1, &z);
	karatsuba_recurse256(temp0, temp1, &z);
	*dst += z << m;
	*dst -= z << (m >> 8);

	// z0
	split_at256(left, (m >> 8), &z, &temp0);
	split_at256(right, (m >> 8), &z, &temp1);
	karatsuba_recurse256(temp0, temp1, &z);
	*dst += z;
	*dst -= z << (m >> 8);
}

void karatsuba_iterative(const size_t& left, const size_t& right, size_t* dst) {
	std::size_t m;
	switch (Base256uMath::compare(&left, sizeof(left), &right, sizeof(right))) {
	case 0:
	case -1:
		Base256uMath::log256(&left, sizeof(left), &m);
		break;
	case 1:
		Base256uMath::log256(&right, sizeof(right), &m);
		break;
	default:
		abort();
	}


}

inline void split_at(const std::size_t& num, const std::size_t& n, std::size_t* high, std::size_t* low) {
	*low = num & (((std::size_t)1 << n) - 1);
	*high = (num ^ *low) >> n;
}

void karatsuba_recurse(const size_t& left, const size_t& right, size_t* dst) {
	if (left <= 255 || right <= 255) {
		*dst = left * right;
		return;
	}
	/*
		std::size_t m = MIN(_log2(left), _log2(right));
		std::size_t m2 = m / 2;

		std::size_t high1, low1, high2, low2;
		split_at(left, m2, &high1, &low1);
		split_at(right, m2, &high2, &low2);

		std::size_t z0, z1, z2;
		karatsuba_recurse(low1, low2, &z0);
		karatsuba_recurse(high1, high2, &z2);
		karatsuba_recurse(low1 + high1, low2 + high2, &z1);
		*dst = (z2 << m) + ((z1 - z2 - z0) << m2) + z0;
	*/

	* dst = 0;

	std::size_t m = MIN(_log2(left), _log2(right));
	bool is_odd = m & 0b1;
	m >>= 1;

	std::size_t temp0, temp1, z;

	// z1
	split_at(left, m, &z, &temp0);
	temp0 += z;
	split_at(right, m, &z, &temp1);
	temp1 += z;
	karatsuba_recurse(temp0, temp1, &z);
	*dst += z << m;

	// z2
	split_at(left, m, &temp0, &z);
	split_at(right, m, &temp1, &z);
	karatsuba_recurse(temp0, temp1, &z);
	m <<= 1;
	if (is_odd)
		m ^= 0b1;
	*dst += z << m;
	m >>= 1;
	*dst -= z << m;

	// z0
	split_at(left, m, &z, &temp0);
	split_at(right, m, &z, &temp1);
	karatsuba_recurse(temp0, temp1, &z);
	*dst += z;
	*dst -= z << m;
}

void karatsuba(std::size_t left, std::size_t right, std::size_t* dst) {
	*dst = 0;
	karatsuba_recurse256(left, right, dst);
}

__host__ __device__
std::size_t _convert_to_size_t(const void* const src, std::size_t n) {
	std::size_t output = *reinterpret_cast<const std::size_t*>(src);
	if (n == 8)
		return output;
	std::size_t mask = -1;
	mask <<= 8 * n;
	return output & ~mask;
}

int main() {
#ifndef _DEBUG
	std::cout << "Make sure you're in debug mode to perform tests!" << std::endl;
#endif

#ifdef BASE256UMATH_FAST_OPERATORS
	std::cout << "FAST_OPERATORS enabled";
#ifdef __CUDACC__
	std::cout << ", but some are disabled for CUDA";
#endif
	std::cout << std::endl;
#endif

	std::cout << "Running unit tests..." << std::endl;
	Base256uMathTests::test_unit_tests();
#ifdef __CUDACC__
	std::cout << "CUDA detected, running CUDA unit tests..." << std::endl;
	Base256uMathTests::CUDA::test_unit_tests();
#endif
	std::cout << "All unit tests passed." << std::endl;

#ifndef __CUDACC__
	std::cout << std::endl;
	std::cout << "Running performance tests..." << std::endl;
	Base256uMathTests::Performance::test(100, 10 * KB);
	std::cout << "Performance tests concluded." << std::endl;
#endif

	return 0;
}