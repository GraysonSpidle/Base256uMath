/* Base256uMath.cpp

Function definitions of all the declared functions in Base256uMath.h

Author: Grayson Spidle
*/

#include "Base256uMath.h"
#ifndef __CUDACC__
#include <cstring> // memset
#include <stdint.h>
#endif // nvcc has memset natively

#ifndef MIN(a,b)
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX(a,b)
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

#include <iostream>
#include <string>

// ==============================================================================================

#if BASE256UMATH_ARCHITECTURE == 64
typedef uint32_t half_size_t;
#elif BASE256UMATH_ARCHITECTURE == 32
typedef uint16_t half_size_t;
#endif

__host__ __device__
void bit_shift_left_fast(
	void* const dst,
	const std::size_t& dst_n,
	unsigned char by_bits
);

__host__ __device__
std::size_t convert_to_size_t(const void* const src, const std::size_t& n) {
	/*
	std::size_t output = *reinterpret_cast<const std::size_t*>(src);
	if (n >= BASE256UMATH_ARCHITECTURE / 8)
		return output;
	std::size_t mask = -1;
	mask <<= (BASE256UMATH_ARCHITECTURE / 8) * n;
	return output & ~mask;
	*/
	std::size_t output = 0;
	memcpy(&output, src, MIN(sizeof(output), n));
	return output;
}

__host__ __device__
bool Base256uMath::is_zero(
	const void* const src,
	std::size_t src_n
) {
	auto ptr = reinterpret_cast<const unsigned char*>(src) + src_n,
		end = reinterpret_cast<const unsigned char*>(src) - 1;
	while( --ptr != end) {
		if (*ptr)
			return false;
	}
	return true;
}

int Base256uMath::compare(
	const void* const left,
	std::size_t left_n,
	const void* const right,
	std::size_t right_n
) {
	if (left_n > right_n) {	
		if (!is_zero(reinterpret_cast<const unsigned char*>(left) + right_n, left_n - right_n))
			return 1;
	}
	else if (left_n < right_n) {
		if (!is_zero(reinterpret_cast<const unsigned char*>(right) + left_n, right_n - left_n))
			return -1;
	}

	auto l_ptr = reinterpret_cast<const unsigned char*>(left) + MIN(left_n, right_n);
	auto r_ptr = reinterpret_cast<const unsigned char*>(right) + MIN(left_n, right_n);
	while (--l_ptr != reinterpret_cast<const unsigned char*>(left) - 1) {
		if (*l_ptr > *--r_ptr)
			return 1;
		else if (*l_ptr < *r_ptr)
			return -1;
	}
	return 0;
}

int Base256uMath::compare(
	const void* const left,
	std::size_t left_n,
	std::size_t right
) {
	return compare(left, left_n, &right, sizeof(right));
}

const void* const Base256uMath::max(
	const void* const left,
	std::size_t left_n,
	const void* const right,
	std::size_t right_n
) {
	int cmp = compare(left, left_n, right, right_n);
	if (cmp > 0)
		return left;
	else if (cmp < 0)
		return right;
	return left;
}

void* const Base256uMath::max(
	void* const left,
	std::size_t left_n,
	void* const right,
	std::size_t right_n
) {
	int cmp = compare(left, left_n, right, right_n);
	if (cmp > 0)
		return left;
	else if (cmp < 0)
		return right;
	return left;
}

const void* const Base256uMath::min(
	const void* const left,
	std::size_t left_n,
	const void* const right,
	std::size_t right_n
) {
	int cmp = compare(left, left_n, right, right_n);
	if (cmp < 0)
		return left;
	else if (cmp > 0)
		return right;
	return left;
}

void* const Base256uMath::min(
	void* const left,
	std::size_t left_n,
	void* const right,
	std::size_t right_n
) {
	int cmp = compare(left, left_n, right, right_n);
	if (cmp < 0)
		return left;
	else if (cmp > 0)
		return right;
	return left;
}

__host__ __device__
inline int increment_fast(void* const block, const std::size_t& n) {
	if (!n)
		return Base256uMath::ErrorCodes::OK;

	auto dst_ptr = reinterpret_cast<unsigned char*>(block);

	if (n / (BASE256UMATH_ARCHITECTURE / 8)) {
		for (std::size_t i = 0; i < n / (BASE256UMATH_ARCHITECTURE / 8) - 1; i++) {
			*reinterpret_cast<std::size_t*>(dst_ptr) += 1;

			if (*reinterpret_cast<std::size_t*>(dst_ptr)) // non-zero check
				return Base256uMath::ErrorCodes::OK;

			dst_ptr += sizeof(std::size_t); // offsetting ptr
		}
	}

	// If we get here, then it is assumed we still need to carry

	bool b = false;

#if BASE256UMATH_ARCHITECTURE >= 64
	// 64 bit
	b = n & (uint8_t)0b1000;
	*reinterpret_cast<uint64_t*>(dst_ptr) += b;
	if (b && *reinterpret_cast<uint64_t*>(dst_ptr))
		return Base256uMath::ErrorCodes::OK;
	dst_ptr += sizeof(uint64_t) * b;
#endif

	// 32 bit
	b = n & (uint8_t)0b100;
	*reinterpret_cast<uint32_t*>(dst_ptr) += b;
	if (b && *reinterpret_cast<uint32_t*>(dst_ptr))
		return Base256uMath::ErrorCodes::OK;
	dst_ptr += sizeof(uint32_t) * b;

	// 16 bit
	b = n & (uint8_t)0b10;
	*reinterpret_cast<uint16_t*>(dst_ptr) += b;
	if (b && *reinterpret_cast<uint16_t*>(dst_ptr))
		return Base256uMath::ErrorCodes::OK;
	dst_ptr += sizeof(uint16_t) * b;

	// 8 bit
	b = n & (uint8_t)0b1;
	*reinterpret_cast<uint8_t*>(dst_ptr) += b;
	if (b && *reinterpret_cast<uint8_t*>(dst_ptr))
		return Base256uMath::ErrorCodes::OK;

	return Base256uMath::ErrorCodes::FLOW * Base256uMath::is_zero(block, n);
}

int Base256uMath::increment(
	void* const block,
	std::size_t n
) {
#ifdef BASE256UMATH_FAST_OPERATORS
	return increment_fast(block, n);
#else
	// Starting at the smallest byte.
	// Add 1 to each byte until carrying is no longer necessary.

	unsigned char* page_ptr = reinterpret_cast<unsigned char*>(block);
	unsigned char* end = page_ptr + n;
	for (; page_ptr != end; ++page_ptr) {
		if (*page_ptr != 255) { // checking if carrying is necessary
			*page_ptr += 1;
			return ErrorCodes::OK;
		}
		else { // carrying is necessary
			*page_ptr += 1;
		}
	}
	return ErrorCodes::FLOW;
#endif
}

__host__ __device__
inline int decrement_fast(void* const block, const std::size_t& n) {
	auto dst_ptr = reinterpret_cast<unsigned char*>(block);

	if (n / (BASE256UMATH_ARCHITECTURE / 8)) {
		for (std::size_t i = 0; i < n / (BASE256UMATH_ARCHITECTURE / 8) - 1; i++) {
			if (*reinterpret_cast<std::size_t*>(dst_ptr)) { // non-zero check
				*reinterpret_cast<std::size_t*>(dst_ptr) -= 1;
				return Base256uMath::ErrorCodes::OK;
			}
			else // borrowing is necessary
				*reinterpret_cast<std::size_t*>(dst_ptr) -= 1;

			dst_ptr += sizeof(std::size_t);
		}
	}

	// if we get here, then it is assumed we still need to borrow

	bool b = false;
	bool b2 = false;

#if BASE256UMATH_ARCHITECTURE >= 64
	// 64 bit
	b = n & 0b1000;
	b2 = *reinterpret_cast<uint64_t*>(dst_ptr);
	*reinterpret_cast<uint64_t*>(dst_ptr) -= b;
	if (b && b2)
		return Base256uMath::ErrorCodes::OK;
	dst_ptr += sizeof(uint64_t) * b;
#endif

	// 32 bit
	b = n & 0b100;
	b2 = *reinterpret_cast<uint32_t*>(dst_ptr);
	*reinterpret_cast<uint32_t*>(dst_ptr) -= b;
	if (b && b2)
		return Base256uMath::ErrorCodes::OK;
	dst_ptr += sizeof(uint32_t) * b;

	// 16 bit
	b = n & 0b10;
	b2 = *reinterpret_cast<uint16_t*>(dst_ptr);
	*reinterpret_cast<uint16_t*>(dst_ptr) -= b;
	if (b && b2)
		return Base256uMath::ErrorCodes::OK;
	dst_ptr += sizeof(uint16_t) * b;

	// 8 bit
	b = n & 0b1;
	b2 = *reinterpret_cast<uint8_t*>(dst_ptr);
	*reinterpret_cast<uint8_t*>(dst_ptr) -= b;
	if (b && b2)
		return Base256uMath::ErrorCodes::OK;

	return Base256uMath::ErrorCodes::FLOW;
}

int Base256uMath::decrement(
	void* const block,
	std::size_t n
) {
#ifdef BASE256UMATH_FAST_OPERATORS
	return decrement_fast(block, n);
#else
	// Starting at the smallest byte.
	// Subtract 1 from each byte until borrowing is no longer necessary.
	unsigned char* page_ptr = reinterpret_cast<unsigned char*>(block);
	unsigned char* end = page_ptr + n;
	for (; page_ptr != end; ++page_ptr) {
		if (*page_ptr) { // checking if borrowing is necessary
			*page_ptr -= 1;
			return ErrorCodes::OK;
		}
		else { // borrowing is necessary
			*page_ptr -= 1;
		}
	}
	return ErrorCodes::FLOW;
#endif
}

__host__ __device__
inline bool binary_add_fast(
	void* const left,
	const std::size_t& left_n,
	const void* const right,
	const std::size_t& right_n
) {
	auto l = reinterpret_cast<uint8_t*>(left);
	auto l_end = reinterpret_cast<uint8_t*>(left) + (left_n & ~(std::size_t)0b111);
	auto r = reinterpret_cast<const uint8_t*>(right);
	auto r_end = reinterpret_cast<const uint8_t*>(right) + (right_n & ~(std::size_t)0b111);

	bool carry = false;
	std::size_t buffer;
	while (l != l_end && r != r_end) {
		buffer = *reinterpret_cast<std::size_t*>(l);
		*reinterpret_cast<std::size_t*>(l) += *reinterpret_cast<const std::size_t*>(r) + carry;
		carry = *reinterpret_cast<std::size_t*>(l) < MAX(buffer, *reinterpret_cast<const std::size_t*>(r));

		l += sizeof(std::size_t);
		r += sizeof(std::size_t);
	}

	bool b;
	std::size_t n = MIN(left_n, right_n);
#if BASE256UMATH_ARCHITECTURE == 64
	b = n & 0b100;
	buffer = *reinterpret_cast<uint32_t*>(l) * b;
	*reinterpret_cast<uint32_t*>(l) += (*reinterpret_cast<const uint32_t*>(r) + carry) * b;
	carry = (b && *reinterpret_cast<uint32_t*>(l) < MAX(buffer, *reinterpret_cast<const uint32_t*>(r))) || (!b && carry);

	l += sizeof(uint32_t*) * b;
	r += sizeof(uint32_t*) * b;
#endif

	b = n & 0b10;
	buffer = *reinterpret_cast<uint16_t*>(l) * b;
	*reinterpret_cast<uint16_t*>(l) += (*reinterpret_cast<const uint16_t*>(r) + carry) * b;
	carry = (b && *reinterpret_cast<uint16_t*>(l) < MAX(buffer, *reinterpret_cast<const uint16_t*>(r))) || (!b && carry);

	l += sizeof(uint16_t*) * b;
	r += sizeof(uint16_t*) * b;

	b = n & 0b1;
	buffer = *l * b;
	*l += (*r + carry) * b;
	carry = (b && *l < MAX(buffer, *r)) || (!b && carry);
	l += b;

	l_end = reinterpret_cast<decltype(l)>(left) + left_n;

	if (carry && l < l_end)
		return increment_fast(l, l_end - l);
	return carry;
}

__host__ __device__
inline bool binary_add(
	void* const left,
	const std::size_t& left_n,
	const void* const right,
	const std::size_t& right_n
) {
	auto l = reinterpret_cast<unsigned char*>(left);
	auto l_end = reinterpret_cast<decltype(l)>(left) + left_n;
	auto r = reinterpret_cast<const unsigned char*>(right);
	auto r_end = reinterpret_cast<decltype(r)>(right) + right_n;

	bool carry = false;
	unsigned char buffer; // we need this to keep track of l[i] before it gets added
	for (; l != l_end && r != r_end; l++, r++) {
		buffer = *l;
		*l += *r + carry;
		carry = *l < (MAX(buffer, *r));
	}
	for (; carry && l != l_end; l++) {
		buffer = *l;
		*l += 1;
		carry = !bool(*l);
	}
	return carry;
}

int Base256uMath::add(
	const void* const left,
	std::size_t left_n,
	const void* const right, 
	std::size_t right_n,
	void* const dst,
	std::size_t dst_n
) {
	memset(dst, 0, dst_n);
	memcpy(dst, left, MIN(left_n, dst_n));
#if defined(BASE256UMATH_FAST_OPERATORS) && !defined(__CUDACC__)
	if (binary_add_fast(dst, dst_n, right, right_n))
#else
	if (binary_add(dst, dst_n, right, right_n))
#endif
		return ErrorCodes::FLOW;
#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (dst_n < MAX(left_n, right_n))
		return ErrorCodes::TRUNCATED;
#endif
	return ErrorCodes::OK;
}

int Base256uMath::add(
	const void* const left,
	std::size_t left_n,
	std::size_t right,
	void* const dst,
	std::size_t dst_n
) {
	memset(dst, 0, dst_n);
	memcpy(dst, left, MIN(left_n, dst_n));
	if (binary_add(dst, dst_n, &right, sizeof(right))) {
		return ErrorCodes::FLOW;
	}
#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (dst_n < MAX(left_n, sizeof(right)))
		return ErrorCodes::TRUNCATED;
#endif
	return ErrorCodes::OK;
}

int Base256uMath::add(
	void* const left,
	std::size_t left_n,
	const void* const right,
	std::size_t right_n
) {
	return ErrorCodes::FLOW * binary_add(left, left_n, right, right_n);
}

int Base256uMath::add(
	void* const left,
	std::size_t left_n, 
	std::size_t right
) {
	if (binary_add(left, left_n, &right, sizeof(right)))
		return ErrorCodes::FLOW;
	return ErrorCodes::OK;
}

__host__ __device__
inline bool binary_subtract_fast(
	void* const left,
	const std::size_t& left_n,
	const void* const right,
	const std::size_t& right_n
) {
	auto l = reinterpret_cast<unsigned char*>(left);
	auto l_end = reinterpret_cast<decltype(l)>(left) + (left_n & ~(std::size_t)0b111);
	auto r = reinterpret_cast<const unsigned char*>(right);
	auto r_end = reinterpret_cast<decltype(r)>(right) + (right_n & ~(std::size_t)0b111);

	bool borrow = false;
	std::size_t buffer;
	while (l != l_end && r != r_end) {
		buffer = *reinterpret_cast<std::size_t*>(l) - borrow;
		*reinterpret_cast<std::size_t*>(l) -= *reinterpret_cast<const std::size_t*>(r) + borrow;
		borrow = (buffer == (std::size_t)-1) || buffer < *reinterpret_cast<const std::size_t*>(r);

		l += sizeof(std::size_t);
		r += sizeof(std::size_t);
	}

	std::size_t n = MIN(left_n, right_n);
	bool b;
#if BASE256UMATH_ARCHITECTURE >= 64
	b = n & 0b100;
	buffer = (*reinterpret_cast<uint32_t*>(l) - borrow) * b;
	*reinterpret_cast<uint32_t*>(l) -= (*reinterpret_cast<const uint32_t*>(r) + borrow) * b;
	borrow = (b && ((buffer == (uint32_t)-1) || buffer < *reinterpret_cast<const uint32_t*>(r))) ||
		(borrow && !b);

	l += sizeof(uint32_t) * b;
	r+= sizeof(uint32_t) * b;
#endif

	b = n & 0b10;
	buffer = (*reinterpret_cast<uint16_t*>(l) - borrow) * b;
	*reinterpret_cast<uint16_t*>(l) -= (*reinterpret_cast<const uint16_t*>(r) + borrow) * b;
	borrow = (b && ((buffer == (uint16_t)-1) || buffer < *reinterpret_cast<const uint16_t*>(r))) ||
		(borrow && !b);

	l += sizeof(uint16_t) * b;
	r += sizeof(uint16_t) * b;

	b = n & 0b1;
	buffer = (uint8_t)(*l - borrow) * b;
	*l -= (*r + borrow) * b;
	borrow = (b && ((buffer == 255) || buffer < *r)) ||
		(borrow && !b);

	l += b;
	
	l_end = reinterpret_cast<decltype(l)>(left) + left_n;

	if (borrow && l < l_end)
		return decrement_fast(l, l_end - l);
	return borrow;
}

__host__ __device__
inline bool binary_subtract(
	void* const left,
	const std::size_t& left_n,
	const void* const right,
	const std::size_t& right_n
) {
	auto l = reinterpret_cast<unsigned char*>(left);
	auto l_end = reinterpret_cast<decltype(l)>(left) + left_n;
	auto r = reinterpret_cast<const unsigned char*>(right);
	auto r_end = reinterpret_cast<decltype(r)>(right) + right_n;

	bool borrow = false;
	unsigned char buffer;

	for (; l != l_end && r != r_end; l++, r++) {
		buffer = *l - borrow;
		*l -= *r + borrow;
		borrow = (buffer == 255) || buffer < *r;
	}
	for (; borrow && l != l_end; l++) {
		buffer = *l;
		*l -= 1;
		borrow = (*l == 255);
	}
	return borrow;
}

int Base256uMath::subtract(
	const void* const left, 
	std::size_t left_n, 
	const void* const right, 
	std::size_t right_n, 
	void* const dst, 
	std::size_t dst_n
) {
	memset(dst, 0, dst_n);
	memcpy(dst, left, MIN(left_n, dst_n));
#if defined(BASE256UMATH_FAST_OPERATORS) && !defined(__CUDACC__)
	if (binary_subtract_fast(dst, dst_n, right, right_n))
#else
	if (binary_subtract(dst, dst_n, right, right_n))
#endif
		return ErrorCodes::FLOW;
#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (dst_n < MAX(left_n, right_n))
		return ErrorCodes::TRUNCATED;
#endif
	return ErrorCodes::OK;
}

int Base256uMath::subtract(
	const void* const left,
	std::size_t left_n,
	std::size_t right,
	void* const dst,
	std::size_t dst_n
) {
	memset(dst, 0, dst_n);
	memcpy(dst, left, MIN(left_n, dst_n));
	if (binary_subtract(dst, dst_n, &right, sizeof(std::size_t)))
		return ErrorCodes::FLOW;
#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (dst_n < MAX(left_n, sizeof(std::size_t)))
		return ErrorCodes::TRUNCATED;
#endif
	return ErrorCodes::OK;
}

int Base256uMath::subtract(
	void* const left,
	std::size_t left_n,
	const void* const right,
	std::size_t right_n
) {
	if (binary_subtract(left, left_n, right, right_n))
		return ErrorCodes::FLOW;
	return ErrorCodes::OK;
}

int Base256uMath::subtract(
	void* const left,
	std::size_t left_n,
	std::size_t right
) {
	if (binary_subtract(left, left_n, &right, sizeof(right)))
		return ErrorCodes::FLOW;
	return ErrorCodes::OK;
}

__host__ __device__
inline void multiply_char_and_char(
	const unsigned char& left,
	const unsigned char& right,
	unsigned char* dst,
	unsigned char* overflow
) {
	uint16_t big = left * right;
	*dst = big;
	*overflow = big >> 8;
}

__host__ __device__
inline void multiply_big_and_char(
	const void* const big,
	const std::size_t& big_n,
	const unsigned char& right,
	unsigned char* dst
) {
	// dst better be of size of at least big_n + 1

	*dst = 0;
	unsigned char product;
	for (std::size_t i = 0; i < big_n; i++) {
		multiply_char_and_char(
			*(reinterpret_cast<const unsigned char*>(big) + i),
			right,
			&product,
			dst + i + 1
		);

		product += *(dst + i);

		*(dst + i + 1) += product < *(dst + i);
		*(dst + i) = product;
	}
}

__host__ __device__
inline int binary_multiply(
	const void* const left,
	const std::size_t& left_n,
	const void* const right,
	const std::size_t& right_n,
	void* const dst,
	const std::size_t& dst_n
) {
	// We're approaching this like binary, but we're using bytes instead of bits
	memset(dst, 0, dst_n);

	// Typically you would want your product to be left_n + right_n, but it's not necessary
	// for this sub-product. We will be using this product as the dst for left * a char.
	// So we can save memory by only allocating for that much.
	unsigned char* product = new unsigned char[left_n + 1];
	if (!product)
		return Base256uMath::ErrorCodes::OOM;

	for (std::size_t i = 0; i < MIN(MAX(left_n, right_n), dst_n); i++) {
		memset(product, 0, left_n + 1);
		multiply_big_and_char(
			left,
			left_n,
			*(reinterpret_cast<const unsigned char*>(right) + i) * (i < right_n),
			product
		);

		// Here is where we would have to shift the sub-product to the left before we add, but
		// since we would only have to shift by whole bytes we can save some time by just offsetting
		// pointers and calling it a day.

		binary_add(
			reinterpret_cast<unsigned char*>(dst) + i,
			dst_n - i,
			product,
			left_n + 1
		);
	}
	delete[] product;
	return Base256uMath::ErrorCodes::OK;
}

__host__ __device__
inline void binary_multiply_fast_big_and_half_size_t(
	const void* const left,
	const std::size_t& left_n,
	const half_size_t& right,
	void* const dst,
	const std::size_t& dst_n
) {
	/* This function is supposed to be used in conjunction with another.
	DO NOT CALL THIS FUNCTION ON ITS OWN, UNLESS YOU KNOW WHAT YOU'RE DOING.

	This function, given a half_size_t number, will multiply it by every 
	number in the "left" number. Just like in regular binary multiplication.
	*/

	auto left_ptr = reinterpret_cast<const uint8_t*>(left);
	auto dst_ptr = reinterpret_cast<uint8_t*>(dst);
	auto dst_end = reinterpret_cast<uint8_t*>(dst) + dst_n;

	// will temporarily store the sub-product
	std::size_t product;

	// iterating through all the half_size_t sections in the "left" number while
	// ensuring dst can acommodate
	for (std::size_t k = 0; k < left_n / sizeof(half_size_t) && dst_ptr < dst_end; k++) {
		product = (std::size_t)reinterpret_cast<const half_size_t*>(left_ptr)[0] *
			(std::size_t)right;

		binary_add(
			dst_ptr,
			dst_end - dst_ptr,
			&product,
			sizeof(product)
		);

		left_ptr += sizeof(half_size_t);
		dst_ptr += sizeof(half_size_t);
	}

	/* 3 Scenarios can occur (all are remedied by the code below):
	
	Scenario 1 - primitive multiplication
	Condition: left_n < sizeof(half_size_t)

	If the condition is met, then the code below should do something, and the
	code above did not do anything.

	Scenario 2 - big multiplication with some residual data
	Condition: left_n > sizeof(half_size_t) && left_n % sizeof(half_size_t) != 0

	If the condition is met, then the code below should do something, and the
	code above did do something just not enough.

	Scenario 3 - ideal case, the code above did everything necessary
	Condition: left_n >= sizeof(half_size_t) && left_n % sizeof(half_size_t) == 0

	If the condition is met, then all the code down below should do nothing since
	we've already iterated through all the bytes in the left number.
	*/

	bool b; // to aid in branchless programming
#if BASE256UMATH_ARCHITECTURE >= 64
	// 16 bit
	b = left_n & 0b10;
	// checking if dst can accommodate
	b = b && dst_ptr < dst_end;
	// if there are at least 2 bytes left over, then we multiply them by the right number
	product = b * (std::size_t)reinterpret_cast<const uint16_t*>(left_ptr)[0] *
		(std::size_t)right;

	binary_add(
		dst_ptr,
		dst_end - dst_ptr,
		&product,
		b * sizeof(product)
	);

	left_ptr += sizeof(uint16_t) * b;
	dst_ptr += sizeof(uint16_t) * b;
#endif

	// 8 bit
	b = left_n & 0b1;
	// checking if dst can accommodate
	b = b && dst_ptr < dst_end;
	// if there is at least 1 byte left over, then we multiply it by the right number
	product = b * (std::size_t)reinterpret_cast<const uint8_t*>(left_ptr)[0] *
		(std::size_t)right;

	binary_add(
		dst_ptr,
		dst_end - dst_ptr,
		&product,
		b * sizeof(product)
	);
}

__host__ __device__
inline int binary_multiply_fast(
	const void* const left,
	const std::size_t& left_n,
	const void* const right,
	const std::size_t& right_n,
	void* const dst,
	const std::size_t& dst_n
) {
	/* A bit of an explanation of what's happening and how this contrasts with
	the non-fast variant.

	The difference with this implementation is that it attempts to utilize as much
	of the register as it can. The non-fast variant only focuses on 1 byte at a time.

	Do essentially the same multiplication as before, but with different sized numbers.
	It helps if you understand regular binary multiplication before understanding this.

	We are using half_size_t instead of size_t because if we were to use size_t, then we
	cannot store the result in a number without allocating memory from the heap. We are
	leveraging the principle that a half_size_t number times a half_size_t number will
	be (at most) a size_t number.
	*/

	memset(dst, 0, dst_n);

	auto right_ptr = reinterpret_cast<const uint8_t*>(right);
	auto dst_ptr = reinterpret_cast<uint8_t*>(dst);

	// We are going to cut the "right" number into half_size_t sections
	// and then feed each section into another function that will
	// multiply each section by each section of the "left" number.
	for (std::size_t i = 0; i < right_n / sizeof(half_size_t); i++) {
		binary_multiply_fast_big_and_half_size_t(
			left, left_n,
			*reinterpret_cast<const half_size_t*>(right_ptr),
			dst_ptr,
			(dst_n + reinterpret_cast<decltype(dst_ptr)>(dst)) - dst_ptr
		);

		// Going on to the next section

		right_ptr += sizeof(half_size_t);
		dst_ptr += sizeof(half_size_t);
	}

	// Sometimes the "right" number is not able to evenly cut into half_size_t sections.
	// This means we have to scale down our register size to fit some of the sections.
	
	bool b; // to aid in branchless programming

#if BASE256UMATH_ARCHITECTURE >= 64
	// interpreting the "right" number as a uint16
	b = (reinterpret_cast<const uint8_t*>(right) + right_n - right_ptr) & 0b10;
	binary_multiply_fast_big_and_half_size_t(
		left, left_n,
		reinterpret_cast<const uint16_t*>(right_ptr)[0],
		dst_ptr,
		(dst_n + reinterpret_cast<decltype(dst_ptr)>(dst) - dst_ptr) * b
	);
	right_ptr += sizeof(uint16_t) * b;
	dst_ptr += sizeof(uint16_t) * b;
#endif

	// interpreting the "right" number as a uint8
	b = (reinterpret_cast<const uint8_t*>(right) + right_n - right_ptr) & 0b1;
	binary_multiply_fast_big_and_half_size_t(
		left, left_n,
		reinterpret_cast<const uint8_t*>(right_ptr)[0],
		dst_ptr,
		(dst_n + reinterpret_cast<decltype(dst_ptr)>(dst) - dst_ptr) * b
	);
	right_ptr += sizeof(uint8_t) * b;
	dst_ptr += sizeof(uint8_t) * b;

	return 0;
}

int Base256uMath::multiply(
	const void* const left,
	std::size_t left_n,
	const void* const right,
	std::size_t right_n,
	void* const dst,
	std::size_t dst_n
) {
	if (Base256uMath::is_zero(left, left_n) || Base256uMath::is_zero(right, right_n)) {
		memset(dst, 0, dst_n);
		return Base256uMath::ErrorCodes::OK;
	}
#if defined(BASE256UMATH_FAST_OPERATORS) && !defined(__CUDACC__)
	auto code = binary_multiply_fast(left, left_n, right, right_n, dst, dst_n);
#else
	auto code = binary_multiply(left, left_n, right, right_n, dst, dst_n);
	if (code < 0)
		return code;
#endif

#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (!bool(dst_n) || dst_n < MIN(left_n, right_n))
		return Base256uMath::ErrorCodes::TRUNCATED;
#endif
	return code;
}

int Base256uMath::multiply(
	const void* const left,
	std::size_t left_n,
	std::size_t right,
	void* const dst,
	std::size_t dst_n
) {
	return multiply(
		left, left_n,
		&right, sizeof(right),
		dst, dst_n
	);
}

int Base256uMath::multiply(
	void* const left,
	std::size_t left_n,
	const void* const right,
	std::size_t right_n
) {
	// We're approaching this like binary, but we're using bytes instead of bits
	// So the in place multiplication is kinda iffy with this approach. 
	// Probably should choose a different approach for this, but for right now
	// I'm just gonna cheat and use the other function.

	unsigned char* dst = new unsigned char[left_n];
	if (!dst)
		return ErrorCodes::OOM;
	auto code = multiply(
		left, left_n,
		right, right_n,
		dst, left_n
	);
	memcpy(left, dst, left_n);
	delete[] dst;
	return code;
}

int Base256uMath::multiply(
	void* const left,
	std::size_t left_n,
	std::size_t right
) {
	return multiply(
		left, left_n,
		&right, sizeof(right)
	);
}

// https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers
// Thank God for stack overflow
#if BASE256UMATH_ARCHITECTURE == 64
#ifndef __CUDACC__
const int tab64[64] = {
	63,  0, 58,  1, 59, 47, 53,  2,
	60, 39, 48, 27, 54, 33, 42,  3,
	61, 51, 37, 40, 49, 18, 28, 20,
	55, 30, 34, 11, 43, 14, 22,  4,
	62, 57, 46, 52, 38, 26, 32, 41,
	50, 36, 17, 19, 29, 10, 13, 21,
	56, 45, 25, 31, 35, 16,  9, 12,
	44, 24, 15,  8, 23,  7,  6,  5 };
#endif
__host__ __device__
int log2_64(uint64_t value)
{
#ifdef __CUDACC__
	const int tab64[64] = {
	63,  0, 58,  1, 59, 47, 53,  2,
	60, 39, 48, 27, 54, 33, 42,  3,
	61, 51, 37, 40, 49, 18, 28, 20,
	55, 30, 34, 11, 43, 14, 22,  4,
	62, 57, 46, 52, 38, 26, 32, 41,
	50, 36, 17, 19, 29, 10, 13, 21,
	56, 45, 25, 31, 35, 16,  9, 12,
	44, 24, 15,  8, 23,  7,  6,  5 };
#endif
	value |= value >> 1;
	value |= value >> 2;
	value |= value >> 4;
	value |= value >> 8;
	value |= value >> 16;
	value |= value >> 32;
	return tab64[((uint64_t)((value - (value >> 1)) * 0x07EDD5E59A4E28C2)) >> 58];
}
#elif BASE256UMATH_ARCHITECTURE == 32
#ifndef __CUDACC__
const int tab32[32] = {
	 0,  9,  1, 10, 13, 21,  2, 29,
	11, 14, 16, 18, 22, 25,  3, 30,
	 8, 12, 20, 28, 15, 17, 24,  7,
	19, 27, 23,  6, 26,  5,  4, 31 };
#endif
__host__ __device__
int log2_32(uint32_t value)
{
#ifdef __CUDACC__
	const int tab32[32] = {
	 0,  9,  1, 10, 13, 21,  2, 29,
	11, 14, 16, 18, 22, 25,  3, 30,
	 8, 12, 20, 28, 15, 17, 24,  7,
	19, 27, 23,  6, 26,  5,  4, 31 };
#endif
	value |= value >> 1;
	value |= value >> 2;
	value |= value >> 4;
	value |= value >> 8;
	value |= value >> 16;
	return tab32[(uint32_t)(value * 0x07C4ACDD) >> 27];
}
#else
#error Unsupported architecture
#endif

__host__ __device__
unsigned long sig_bit(std::size_t n) { // essentially just log2
#if BASE256UMATH_ARCHITECTURE == 64
	return log2_64(n);
#elif BASE256UMATH_ARCHITECTURE == 32
	return log2_32(n);
#else
#error Unsupported architecture
#endif
}

__host__ __device__
inline int binary_long_division(
	const void* const dividend,
	std::size_t& dividend_n,
	const void* const divisor,
	std::size_t& divisor_n,
	void* const dst,
	std::size_t& dst_n,
	void* const remainder,
	std::size_t& remainder_n
) {
	// we assume that dividend > divisor, now we must set up some numbers.

	// Copy dividend into remainder
	memcpy(remainder, dividend, dividend_n);

	// In binary long division, you do a kind of bit shifting so we need
	// to know the index of the most significant bit for both numbers.

	Base256uMath::bit_size_t dividend_log2;
	Base256uMath::bit_size_t divisor_log2;

	Base256uMath::log2(dividend, dividend_n, dividend_log2);
	Base256uMath::log2(divisor, divisor_n, divisor_log2);

	// To get the amount we must bit shift the divisor by, we must subtract
	// the two log2 numbers. We aren't concerned about an underflow error,
	// because we have already checked that dividend > divisor so at minimum
	// the difference would be 0.

	Base256uMath::bit_size_t log2_diff;
	memcpy(log2_diff, dividend_log2, sizeof(log2_diff));
	binary_subtract(
		log2_diff, sizeof(log2_diff),
		divisor_log2, sizeof(divisor_log2)
	);

	// This number is for an offset for the quotient
	std::size_t bytes;
	memcpy(&bytes, log2_diff, sizeof(bytes));
	bytes >>= 3;

	// We need to copy the divisor because we need to be able to bit shift it

	auto divisor_copy = new unsigned char[dividend_n];
	if (!divisor_copy)
		return Base256uMath::ErrorCodes::OOM;

	// Now here's where the dividing starts

	bool b = true;

	// while remainder >= divisor (not the divisor_copy)
	while (Base256uMath::compare(remainder, remainder_n, divisor, divisor_n) >= 0 && b) {
		// At the beginning of each pass, we reset the divisor_copy and bit shift it again
		memset(divisor_copy, 0, dividend_n);
		memcpy(divisor_copy, divisor, MIN(divisor_n, dividend_n));
		Base256uMath::bit_shift_left(divisor_copy, dividend_n, log2_diff, sizeof(log2_diff));

		// check if the newly shifted divisor_copy is <=> to remainder

		int cmp = Base256uMath::compare(
			remainder, remainder_n,
			divisor_copy, dividend_n
		);
		switch (cmp) {
		case 0: // remainder == divisor_copy
		case 1: // remainder > divisor_copy
			// Do the subtraction
			binary_subtract(
				remainder, remainder_n,
				divisor_copy, dividend_n
			);
			// if the subtract function returns a flow warning code, then that's bad
			// and is an indicator of flawed logic

			// here, we are essentially setting the bit in the quotient.
			// We use `bytes` to essentially byte shift for us through pointer offsetting
			// Then we take the remaining amount of bits to shift and shift by that.
			if (dst_n - bytes)
				reinterpret_cast<unsigned char*>(dst)[bytes] |= (unsigned char)1 << (log2_diff[0] & 0b111);
		case -1: // remainder < divisor_copy
			// All cases eventually get here. In binary long division, you keep
			// shifting to the right until you can no longer do so. This effectively
			// does that.
			b = !bool(decrement_fast(log2_diff, sizeof(log2_diff)));
			memcpy(&bytes, log2_diff, sizeof(bytes));
			bytes >>= 3;
			break;
		}

	}
	delete[] divisor_copy;

	// Now we check if there could be possible truncation
	// To guarantee no truncation whatsoever these conditions must be met:
	// 	- dst_n and remainder_n must be >= to dividend_n
	// 	- dividend_n >= divisor_n
	// This is just to be 100% careful, but it's a warning code for a reason,
	// it is only a suggestion and is in no way fatal.

#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (MIN(dst_n, remainder_n) < dividend_n || dividend_n < divisor_n) {
		return Base256uMath::ErrorCodes::TRUNCATED;
	}
#endif
	return Base256uMath::ErrorCodes::OK;
}

// in-place
__host__ __device__
inline int binary_long_division(
	void* const dividend,
	std::size_t& dividend_n,
	const void* const divisor,
	std::size_t& divisor_n,
	void* const remainder,
	std::size_t& remainder_n
) {
	// in-place binary long division *should* work the same way as regular just with a few tweaks

	// Copy dividend into remainder
	memcpy(remainder, dividend, dividend_n);

	// In binary long division, you do a kind of bit shifting so we need
	// to know the index of the most significant bit for both numbers.

	Base256uMath::bit_size_t dividend_log2;
	Base256uMath::bit_size_t divisor_log2;

	Base256uMath::log2(dividend, dividend_n, dividend_log2);
	// That was the last use of the dividend parameter with its original data.
	// That means we are now free to use dividend as the quotient

	memset(dividend, 0, dividend_n);

	Base256uMath::log2(divisor, divisor_n, divisor_log2);

	// To get the amount we must bit shift the divisor by, we must subtract
	// the two log2 numbers. We aren't concerned about an underflow error,
	// because we have already checked that dividend > divisor so at minimum
	// the difference would be 0.

	Base256uMath::bit_size_t log2_diff;
	memcpy(log2_diff, dividend_log2, sizeof(log2_diff));
	binary_subtract(
		log2_diff, sizeof(log2_diff),
		divisor_log2, sizeof(divisor_log2)
	);

	// This number is for an offset for the quotient
	std::size_t bytes;
	memcpy(&bytes, log2_diff, sizeof(bytes));
	bytes >>= 3;
	//Base256uMath::bit_shift_right(log2_diff, sizeof(log2_diff), 3, &bytes, sizeof(bytes));

	// We need to copy the divisor because we need to be able to bit shift it

	auto divisor_copy = new unsigned char[dividend_n];
	if (!divisor_copy)
		return Base256uMath::ErrorCodes::OOM;

	// Now here's where the dividing starts

	// while remainder >= divisor (not the divisor_copy)
	while (Base256uMath::compare(remainder, remainder_n, divisor, divisor_n) >= 0) {
		// At the beginning of each pass, we reset the divisor_copy and bit shift it again
		memset(divisor_copy, 0, dividend_n);
		memcpy(divisor_copy, divisor, MIN(divisor_n, dividend_n));
		Base256uMath::bit_shift_left(divisor_copy, dividend_n, log2_diff, sizeof(log2_diff));

		// check if the newly shifted divisor_copy is <=> to remainder

		int cmp = Base256uMath::compare(
			remainder, remainder_n,
			divisor_copy, dividend_n
		);
		switch (cmp) {
		case 0: // remainder == divisor_copy
		case 1: // remainder > divisor_copy
			// Do the subtraction
			binary_subtract(
				remainder, remainder_n,
				divisor_copy, dividend_n
			);
			// if the subtract function returns a flow warning code, then that's bad
			// and is an indicator of flawed logic

			// here, we are essentially setting the bit in the quotient.
			// We use `bytes` to essentially byte shift for us through pointer offsetting
			// Then we take the remaining amount of bits to shift and shift by that.
			if (dividend_n - bytes)
				reinterpret_cast<unsigned char*>(dividend)[bytes] |= (unsigned char)1 << (log2_diff[0] & 0b111);
		case -1: // remainder < divisor_copy
			// All cases eventually get here. In binary long division, you keep
			// shifting to the right until you can no longer do so. This effectively
			// does that.
			decrement_fast(log2_diff, sizeof(log2_diff));
			memcpy(&bytes, log2_diff, sizeof(bytes));
			bytes >> 3;
			break;
		}
	}
	delete[] divisor_copy;

	// Now we check if there could be possible truncation
	// To guarantee no truncation whatsoever these conditions must be met:
	// 	- dst_n and remainder_n must be >= to dividend_n
	// 	- dividend_n >= divisor_n
	// This is just to be 100% careful, but it's a warning code for a reason,
	// it is only a suggestion and is in no way fatal.

#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (dividend_n < divisor_n) {
		return Base256uMath::ErrorCodes::TRUNCATED;
	}
#endif
	return Base256uMath::ErrorCodes::OK;

}

int Base256uMath::divide(
	const void* const dividend,
	std::size_t dividend_n,
	const void* const divisor,
	std::size_t divisor_n,
	void* const dst,
	std::size_t dst_n,
	void* const remainder,
	std::size_t remainder_n
) {
	/* Division sucks. Our approach is gonna be modeled off binary long division.
	I know, I know, boooooo! But I'm so over trying to be clever with making a division
	algorithm, so I'm not gonna bother.
	*/

	dividend_n = MIN(dividend_n, remainder_n);

	if (Base256uMath::is_zero(divisor, divisor_n)) {
		// cannot divide by zero
		return ErrorCodes::DIVIDE_BY_ZERO;
	}

#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (!dst_n) {
		return ErrorCodes::TRUNCATED;
	}
#endif

	memset(dst, 0, dst_n);
	memset(remainder, 0, remainder_n);

	if (Base256uMath::is_zero(dividend, dividend_n)) {
#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
		if (remainder_n == 0)
			return ErrorCodes::TRUNCATED;
#endif
		// 0 divided by anything is 0 and the remainder is also 0.
		return ErrorCodes::OK;
	}

	// Check, before we do any division, if dividend > divisor.

	switch (Base256uMath::compare(dividend, dividend_n, divisor, divisor_n)) {
	case -1: // dividend < divisor
		// Quotient becomes 0 and remainder becomes dividend
		memcpy(remainder, dividend, dividend_n);
#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
		if (MIN(dst_n, remainder_n) < dividend_n)
			return ErrorCodes::TRUNCATED;
#endif
		return ErrorCodes::OK;
	case 0: // dividend == divisor
		// Quotient becomes 1 and remainder becomes 0
		reinterpret_cast<unsigned char*>(dst)[0] = 1;
#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
		if (MIN(dst_n, remainder_n) < dividend_n)
			return ErrorCodes::TRUNCATED;
#endif
		return ErrorCodes::OK;
	}
	return binary_long_division(dividend, dividend_n, divisor, divisor_n, dst, dst_n, remainder, remainder_n);	
}

int Base256uMath::divide(
	const void* const left,
	std::size_t left_n,
	std::size_t right,
	void* const dst,
	std::size_t dst_n,
	void* const remainder,
	std::size_t remainder_n
) {
	return divide(left, left_n, &right, sizeof(right), dst, dst_n, remainder, remainder_n);
}

int Base256uMath::divide(
	void* const dividend,
	std::size_t dividend_n,
	const void* const divisor,
	std::size_t divisor_n,
	void* const remainder,
	std::size_t remainder_n
) {
	if (!dividend_n)
		return ErrorCodes::OK;

	void* quotient = malloc(dividend_n);
	if (!quotient)
		return ErrorCodes::OOM;

	if (divide(dividend, dividend_n, divisor, divisor_n, quotient, dividend_n, remainder, remainder_n) == ErrorCodes::DIVIDE_BY_ZERO) {
		free(quotient);
		return ErrorCodes::DIVIDE_BY_ZERO;
	}

	if (Base256uMath::is_zero(dividend, dividend_n))
		memset(remainder, 0, remainder_n);
	memcpy(dividend, quotient, dividend_n);

	free(quotient);

	return ErrorCodes::OK;
}

int Base256uMath::divide(
	void* const left,
	std::size_t left_n,
	std::size_t right,
	void* const remainder,
	std::size_t remainder_n
) {
	return divide(left, left_n, &right, sizeof(right), remainder, remainder_n);
}

int Base256uMath::divide_no_mod(
	const void* const dividend,
	std::size_t dividend_n,
	const void* const divisor,
	std::size_t divisor_n,
	void* const dst,
	std::size_t dst_n
) {
	void* remainder = malloc(dividend_n);
	if (!remainder)
		return Base256uMath::ErrorCodes::OOM;

	if (Base256uMath::is_zero(divisor, divisor_n)) {
		// cannot divide by zero
		return ErrorCodes::DIVIDE_BY_ZERO;
	}

#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (!dst_n) {
		return ErrorCodes::TRUNCATED;
	}
#endif

	memset(dst, 0, dst_n);
	memset(remainder, 0, dividend_n);

	if (Base256uMath::is_zero(dividend, dividend_n)) {
		// 0 divided by anything is 0 and the remainder is also 0.
		return ErrorCodes::OK;
	}

	// Check, before we do any division, if dividend > divisor.

	switch (Base256uMath::compare(dividend, dividend_n, divisor, divisor_n)) {
	case -1: // dividend < divisor
		// Quotient becomes 0 and remainder becomes dividend
		memcpy(remainder, dividend, dividend_n);
#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
		if (dst_n < dividend_n)
			return ErrorCodes::TRUNCATED;
#endif
		return ErrorCodes::OK;
	case 0: // dividend == divisor
		// Quotient becomes 1 and remainder becomes 0
		reinterpret_cast<unsigned char*>(dst)[0] = 1;
#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
		if (dst_n < dividend_n)
			return ErrorCodes::TRUNCATED;
#endif
		return ErrorCodes::OK;
	}

	auto code = binary_long_division(dividend, dividend_n, divisor, divisor_n, dst, dst_n, remainder, dividend_n);
	free(remainder);
	return code;
}

int Base256uMath::divide_no_mod(
	const void* const left,
	std::size_t left_n,
	std::size_t right,
	void* const dst,
	std::size_t dst_n
) {
	return divide_no_mod(left, left_n, &right, sizeof(right), dst, dst_n);
}

int Base256uMath::divide_no_mod(
	void* const left,
	std::size_t left_n,
	const void* const right,
	std::size_t right_n
) {
	if (!left_n)
		return ErrorCodes::OK;

	void* remainder = malloc(left_n);
	if (!remainder)
		return ErrorCodes::OOM;

	if (divide(left, left_n, right, right_n, remainder, left_n) == ErrorCodes::DIVIDE_BY_ZERO) {
		free(remainder);
		return ErrorCodes::DIVIDE_BY_ZERO;
	}

	free(remainder);
	return 0;
}

int Base256uMath::divide_no_mod(void* const left, std::size_t left_n, std::size_t right) {
	return divide_no_mod(left, left_n, &right, sizeof(right));
}

int Base256uMath::mod(
	const void* const dividend,
	std::size_t dividend_n,
	const void* const divisor,
	std::size_t divisor_n,
	void* const dst,
	std::size_t dst_n
) {
	// so this will look identical to the division implementation, and that's because it is,
	// but with a few tweaks.

	dividend_n = MIN(dividend_n, dst_n);

	if (Base256uMath::is_zero(divisor, divisor_n)) {
		// cannot divide by zero
		return ErrorCodes::DIVIDE_BY_ZERO;
	}

	memset(dst, 0, dst_n);

#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (!dst_n)
		return ErrorCodes::TRUNCATED;
#endif

	if (Base256uMath::is_zero(dividend, dividend_n)) {
		// 0 divided by anything is 0 and the remainder is also 0.
		return ErrorCodes::OK;
	}

	// Check, before we do any division, if dividend > divisor.

	switch (Base256uMath::compare(dividend, dividend_n, divisor, divisor_n)) {
	case -1: // dividend < divisor
		// Quotient becomes 0 and remainder becomes dividend
		memcpy(dst, dividend, dividend_n);
#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
		if (dst_n < dividend_n)
			return ErrorCodes::TRUNCATED;
#endif
		return ErrorCodes::OK;
	case 0: // dividend == divisor
		// Quotient becomes 1 and remainder becomes 0
		reinterpret_cast<unsigned char*>(dst)[0] = 1;
#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
		if (dst_n < dividend_n)
			return ErrorCodes::TRUNCATED;
#endif
		return ErrorCodes::OK;
	}

	// Now we know that dividend > divisor, we must set up some numbers.

	// Copy dividend into remainder
	memcpy(dst, dividend, dividend_n);

	// In binary long division, you do a kind of bit shifting so we need
	// to know the index of the most significant bit for both numbers.

	Base256uMath::bit_size_t dividend_log2;
	Base256uMath::bit_size_t divisor_log2;

	Base256uMath::log2(dividend, dividend_n, dividend_log2);
	Base256uMath::log2(divisor, divisor_n, divisor_log2);

	// To get the amount we must bit shift the divisor by, we must subtract
	// the two log2 numbers. We aren't concerned about an underflow error,
	// because we have already checked that dividend > divisor so at minimum
	// the difference would be 0.

	Base256uMath::bit_size_t log2_diff;
	Base256uMath::subtract(
		dividend_log2, sizeof(dividend_log2),
		divisor_log2, sizeof(divisor_log2),
		log2_diff, sizeof(log2_diff)
	);

	// This number is for an offset for the quotient
	std::size_t bytes;
	Base256uMath::bit_shift_right(log2_diff, sizeof(log2_diff), 3, &bytes, sizeof(bytes));

	// We need to copy the divisor because we need to be able to bit shift it

	auto divisor_copy = new unsigned char[dividend_n];
	if (!divisor_copy)
		return ErrorCodes::OOM;

	// Now here's where the dividing starts

	bool b = true;;
	// while dst >= divisor (not the divisor_copy)
	while (Base256uMath::compare(dst, dst_n, divisor, divisor_n) >= 0 && b) {
		// At the beginning of each pass, we reset the divisor_copy and bit shift it again
		memset(divisor_copy, 0, dividend_n);
		memcpy(divisor_copy, divisor, MIN(divisor_n, dividend_n));
		Base256uMath::bit_shift_left(divisor_copy, dividend_n, log2_diff, sizeof(log2_diff));

		// check if the newly shifted divisor_copy is <=> to remainder

		int cmp = Base256uMath::compare(
			dst, dst_n,
			divisor_copy, dividend_n
		);
		switch (cmp) {
		case 0: // remainder == divisor_copy
		case 1: // remainder > divisor_copy
			// Do the subtraction
			Base256uMath::subtract(
				dst, dst_n,
				divisor_copy, dividend_n
			);
			// if the subtract function returns a flow warning code, then that's bad
			// and is an indicator of flawed logic

			// here is where the division and modulo implementations differ.
			// We don't give 2 shits about the quotient so we don't use any
			// code that pertains to it. Simple as that.
		case -1: // remainder < divisor_copy
			// All cases eventually get here. In binary long division, you keep
			// shifting to the right until you can no longer do so. This effectively
			// does that.
			b = !bool(Base256uMath::decrement(log2_diff, sizeof(log2_diff)));
			Base256uMath::bit_shift_right(log2_diff, sizeof(log2_diff), 3, &bytes, sizeof(bytes));
			break;
		}
	}
	delete[] divisor_copy;

	// Now we check if there could be possible truncation
	// To guarantee no truncation whatsoever these conditions must be met:
	// 	- dst_n >= dividend_n
	// 	- dividend_n >= divisor_n
	// This is just to be 100% careful, but it's a warning code for a reason,
	// it is only a suggestion and is in no way fatal.

#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (dst_n < dividend_n || dividend_n < divisor_n) {
		return ErrorCodes::TRUNCATED;
	}
#endif
	return ErrorCodes::OK;
}

int Base256uMath::mod(const void* const left, std::size_t left_n, std::size_t right, void* const dst, std::size_t dst_n) {
	return mod(left, left_n, &right, sizeof(right), dst, dst_n);
}

int Base256uMath::mod(void* const left, std::size_t left_n, const void* const right, std::size_t right_n) {
	if (!left_n)
		return ErrorCodes::OK;

	void* dst = malloc(left_n);
	if (!dst)
		return ErrorCodes::OOM;

	if (mod(left, left_n, right, right_n, dst, left_n) == ErrorCodes::DIVIDE_BY_ZERO) {
		free(dst);
		return ErrorCodes::DIVIDE_BY_ZERO;
	}

	memcpy(left, dst, left_n);
	free(dst);

	return ErrorCodes::OK;
}

int Base256uMath::mod(void* const left, std::size_t left_n, std::size_t right) {
	return mod(left, left_n, &right, sizeof(right));
}

int Base256uMath::log2(
	const void* const src,
	std::size_t src_n,
	void* const dst,
	std::size_t dst_n
) {
	// The biggest dst_n should ever be is sizeof(std::size_t) + 1 bytes.
	// So using the Base256uMath::bit_size_t typedef will guarantee getting all data.

	// Essentially, just find the most significant bit.

	std::size_t sig_byte;

	if (log256(src, src_n, &sig_byte) == ErrorCodes::DIVIDE_BY_ZERO)
		return ErrorCodes::DIVIDE_BY_ZERO;

	// From this point on, src is non-zero.

	memset(dst, 0, dst_n);

	memcpy(dst, &sig_byte, MIN(sizeof(std::size_t), dst_n));

#ifndef __CUDACC__
	bit_shift_left_fast(dst, dst_n, 3); // Multiplying by 8 since there are 8 bits in a byte
#else
	bit_shift_left(dst, dst_n, 3);
#endif

	auto num = sig_bit(reinterpret_cast<const unsigned char*>(src)[sig_byte]);

	reinterpret_cast<unsigned char*>(dst)[0] |= num;
	//add(dst, dst_n, &num, sizeof(num));

#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (Base256uMath::compare(dst, dst_n, &sig_byte, sizeof(sig_byte)) < 0)
		return ErrorCodes::TRUNCATED;
#endif
	return ErrorCodes::OK;
}

int Base256uMath::log2(
	const void* const src,
	std::size_t src_n,
	Base256uMath::bit_size_t dst
) {
	// probably could make this faster due to knowing dst is of correct size,
	// but I might get to that later
	return log2(
		src, src_n,
		dst, sizeof(std::size_t) + 1 // if you do sizeof(dst), then it gets it wrong.
	);
}

// Essentially gets the most significant byte
int Base256uMath::log256(
	const void* const src,
	std::size_t src_n,
	std::size_t* const dst
) {
	auto ptr = reinterpret_cast<const unsigned char*>(src) + src_n - 1;
	auto begin = reinterpret_cast<const unsigned char*>(src) - 1;

	for (; ptr != begin; --ptr) {
		if (*ptr) {
			*dst = ptr - (begin + 1);
			return ErrorCodes::OK;
		}
	}
	return ErrorCodes::DIVIDE_BY_ZERO;
}

void bit_shift_left_fast(
	void* const dst,
	const std::size_t& dst_n,
	unsigned char by_bits
) {
	auto dst_ptr = reinterpret_cast<unsigned char*>(dst) + dst_n;

	unsigned char buffer = 0;

	if (dst_n / (BASE256UMATH_ARCHITECTURE / 8)) {
		// This is to tell the loop when to start adding the shifted bits to the previous size_t
		// which is after the 1st iteration.
		bool toggle = false;
		for (std::size_t i = 0; i < dst_n / (BASE256UMATH_ARCHITECTURE / 8) - 1; i++) {
			if (toggle) {
				*dst_ptr |= buffer;
			}

			*(reinterpret_cast<std::size_t*>(dst_ptr) - 1) <<= by_bits;

			buffer = *(dst_ptr - 1);
			buffer &= 255 << (8 - by_bits);
			buffer >>= 8 - by_bits;

			toggle = true;

			dst_ptr -= sizeof(std::size_t);
		}
	}

#if BASE256UMATH_ARCHITECTURE > 64
#error Hard coding may be required. Look at the comments for additional information on this error.
	// If you're getting this error, then you either:
	// - Are from the future, and have a 128 bit processor and haven't gotten to this fix yet. 
	// - You put the incorrect number for the BASE256UMATH_ARCHITECTURE macro (in which case, I got nothing for you).

	// if in the case of the former, then to make this function work you should:
	// - Copy and paste the 64 bit block of code (including the preprocessor directive) above the 64 bit block.
	// - Make the appropriate changes (ie changing 64 to 128, changing 0b1000 to 0b10000).
	// - If you did everything correctly, then it should work.
#endif

	bool b;
#if BASE256UMATH_ARCHITECTURE >= 64
	// 64 bit
	b = dst_n & 0b1000;
	*dst_ptr |= b * (dst_ptr - reinterpret_cast<uint8_t*>(dst) < dst_n) * buffer;
	*(reinterpret_cast<uint64_t*>(dst_ptr) - 1) <<= b * by_bits;
	buffer = (buffer * !b) | *(dst_ptr - sizeof(uint64_t) - 1) * b;
	buffer &= 255 << (8 - by_bits) * b;
	buffer >>= (8 - by_bits) * b;
	dst_ptr -= sizeof(uint64_t) * b;
#endif

	// 32 bit
	b = dst_n & 0b100;
	*dst_ptr |= b * (dst_ptr - reinterpret_cast<uint8_t*>(dst) < dst_n) * buffer;
	*(reinterpret_cast<uint32_t*>(dst_ptr) - 1) <<= b * by_bits;
	buffer = (buffer * !b) | *(dst_ptr - sizeof(uint32_t) - 1) * b;
	buffer &= 255 << (8 - by_bits) * b;
	buffer >>= (8 - by_bits) * b;
	dst_ptr -= sizeof(uint32_t) * b;

	// 16 bit
	b = dst_n & 0b10;
	*dst_ptr |= b * (dst_ptr - reinterpret_cast<uint8_t*>(dst) < dst_n) * buffer;
	*(reinterpret_cast<uint16_t*>(dst_ptr) - 1) <<= b * by_bits;
	buffer = (buffer * !b) | *(dst_ptr - sizeof(uint16_t) - 1) * b;
	buffer &= 255 << (8 - by_bits) * b;
	buffer >>= (8 - by_bits) * b;
	dst_ptr -= sizeof(uint16_t) * b;

	// 8 bit
	b = dst_n & 0b1;
	*dst_ptr |= b * (dst_ptr - reinterpret_cast<uint8_t*>(dst) < dst_n) * buffer;
	*(dst_ptr - 1) <<= b * by_bits;
}

int Base256uMath::bit_shift_left(
	const void* const src,
	std::size_t src_n,
	const void* const by,
	std::size_t by_n,
	void* const dst,
	std::size_t dst_n
) {
	// dividing 'by' by 8 and getting the number of bytes to shift

	std::size_t bytes = *reinterpret_cast<const std::size_t*>(by) * bool(by_n);
	bytes >>= 3;
	if (by_n > sizeof(std::size_t))
		bytes |= *(reinterpret_cast<const unsigned char*>(by) + sizeof(std::size_t)) & (unsigned char)0b111;

	auto code = byte_shift_left(src, src_n, bytes, dst, dst_n);

	// shifting by the number of bits we missed with byte shifting.

	unsigned char n_bits = (*reinterpret_cast<const unsigned char*>(by) * bool(by_n)) & (unsigned char)0b111;

	if (n_bits) {
#if defined(BASE256UMATH_FAST_OPERATORS) && !defined(__CUDACC__)
		bit_shift_left_fast(dst, dst_n, n_bits);
#else
		// Slower alternative, but is reliable. No fancy manipulation of pointers here.
		// Iterating through the number, char by char, and bit shifting.

		auto dst_ptr = reinterpret_cast<unsigned char*>(dst) + dst_n;
		unsigned char mask = ~(((unsigned char)1 << (8 - n_bits)) - 1);
		while (--dst_ptr >= dst) {
			*dst_ptr <<= n_bits;
			*dst_ptr |= ((*(dst_ptr - 1) * (dst_ptr - 1 >= dst)) & mask) >> (8 - n_bits);
		}
#endif
	}

#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (!code && (src_n > dst_n))
		return ErrorCodes::TRUNCATED;
#endif
	return code;
}

int Base256uMath::bit_shift_left(
	void* const src,
	std::size_t src_n,
	const void* const by,
	std::size_t by_n
) {
	// dividing 'by' by 8 and getting the number of bytes to shift

	std::size_t bytes = *reinterpret_cast<const std::size_t*>(by) * bool(by_n);
	bytes >>= 3;
	if (by_n > sizeof(std::size_t))
		bytes |= *(reinterpret_cast<const unsigned char*>(by) + sizeof(std::size_t)) & (unsigned char)0b111;

	auto code = byte_shift_left(src, src_n, bytes);

	// shifting by the number of bits we missed with byte shifting.

	unsigned char n_bits = (*reinterpret_cast<const unsigned char*>(by) * bool(by_n)) & (unsigned char)0b111;

	if (n_bits) {
#if defined(BASE256UMATH_FAST_OPERATORS) && !defined(__CUDACC__)
		bit_shift_left_fast(src, src_n, n_bits);
#else
		auto src_ptr = reinterpret_cast<unsigned char*>(src) + src_n;
		unsigned char mask = ~(((unsigned char)1 << (8 - n_bits)) - 1);
		while (--src_ptr >= src) {
			*src_ptr <<= n_bits;
			*src_ptr |= ((*(src_ptr - 1) * (src_ptr - 1 >= src)) & mask) >> (8 - n_bits);
		}
#endif
	}
	return code;
}

int Base256uMath::bit_shift_left(
	const void* const src,
	std::size_t src_n,
	std::size_t by,
	void* const dst,
	std::size_t dst_n
) {
	// the 'by' parameter is capped at a number that is 8 times less than the highest index,
	// thus its highest value (for bytes) is 8 times less than std::size_t's highest value.

	auto code = byte_shift_left(src, src_n, by >> 3, dst, dst_n); // by / 8
#if BASE256UMATH_ARCHITECTURE == 64
	unsigned char n_bits = 0b111 & by;
#elif BASE256UMATH_ARCHITECTURE == 32
	unsigned char n_bits = 0b11 & by;
#endif

	if (n_bits) {
#if defined(BASE256UMATH_FAST_OPERATORS) && !defined(__CUDACC__)
		bit_shift_left_fast(dst, dst_n, n_bits);
#else
		auto dst_ptr = reinterpret_cast<unsigned char*>(dst) + dst_n;
		unsigned char mask = ~(((unsigned char)1 << (8 - n_bits)) - 1);
		while (--dst_ptr >= dst) {
			*dst_ptr <<= n_bits;
			*dst_ptr |= ((*(dst_ptr - 1) * (dst_ptr - 1 >= dst)) & mask) >> (8 - n_bits);
		}
#endif
	}
	return code;
}

int Base256uMath::bit_shift_left(
	void* const src,
	std::size_t src_n,
	std::size_t by
) {
	auto code = byte_shift_left(src, src_n, by >> 3);
	unsigned char n_bits = 0b111 & by;
	
	if (n_bits) {
#if defined(BASE256UMATH_FAST_OPERATORS) && !defined(__CUDACC__)
		bit_shift_left_fast(src, src_n, n_bits);
#else
		auto src_ptr = reinterpret_cast<unsigned char*>(src) + src_n;
		unsigned char mask = ~(((unsigned char)1 << (8 - n_bits)) - 1);
		while (--src_ptr >= src) {
			*src_ptr <<= n_bits;
			*src_ptr |= ((*(src_ptr - 1) * (src_ptr - 1 >= src)) & mask) >> (8 - n_bits);
		}
#endif
	}
	return code;
}


__host__ __device__
void bit_shift_right_fast(
	void* const dst,
	const std::size_t& dst_n,
	unsigned char by_bits
) {
	auto dst_ptr = reinterpret_cast<unsigned char*>(dst);

	unsigned char buffer = 0;

	if (dst_n / (BASE256UMATH_ARCHITECTURE / 8)) {

		// This is to tell the loop when to start adding the shifted bits to the previous size_t
		// which is after the 1st iteration.
		bool toggle = false;
		for (std::size_t i = 0; i < dst_n / (BASE256UMATH_ARCHITECTURE / 8) - 1; i++) {
			if (toggle) {
				*(dst_ptr - 1) |= buffer;
			}

			*reinterpret_cast<std::size_t*>(dst_ptr) >>= by_bits;
			buffer = *(dst_ptr + sizeof(std::size_t));
			buffer &= ~(255 << by_bits);
			buffer <<= 8 - by_bits;

			toggle = true;

			dst_ptr += sizeof(std::size_t);
		}
	}
#if BASE256UMATH_ARCHITECTURE > 64
#error Hard coding may be required. Look at the comments for additional information on this error.
// If you're getting this error, then you either:
// - Are from the future, and have a 128 bit processor and haven't gotten to this fix yet. 
// - You put the incorrect number for the BASE256UMATH_ARCHITECTURE macro (in which case, I got nothing for you).

// if in the case of the former, then to make this function work you should:
// - Copy and paste the 64 bit block of code (including the preprocessor directive) above the 64 bit block.
// - Make the appropriate changes (ie changing 64 to 128, changing 0b1000 to 0b10000).
// - If you did everything correctly, then it should work.
#endif

#if BASE256UMATH_ARCHITECTURE >= 64
	// 64 bit
	if (dst_n & 0b1000) { // checking divisibility 
		if (dst_ptr - reinterpret_cast<unsigned char*>(dst) > 0) // checking if this is the very first number in the bigger number
			*(dst_ptr - 1) |= buffer;

		*reinterpret_cast<uint64_t*>(dst_ptr) >>= by_bits;

		buffer = *(dst_ptr + sizeof(uint64_t));
		buffer &= ~(255 << by_bits);
		buffer <<= 8 - by_bits;

		dst_ptr += sizeof(uint64_t);
	}
#endif

	// 32 bit
	if (dst_n & 0b100) { // checking divisibility 
		if (dst_ptr - reinterpret_cast<unsigned char*>(dst) > 0) // checking if this is the very first number in the bigger number
			*(dst_ptr - 1) |= buffer;

		*reinterpret_cast<uint32_t*>(dst_ptr) >>= by_bits;

		buffer = *(dst_ptr + sizeof(uint32_t));
		buffer &= ~(255 << by_bits);
		buffer <<= 8 - by_bits;

		dst_ptr += sizeof(uint32_t);
	}

	// 16 bit
	if (dst_n & 0b10) { // checking divisibility 
		if (dst_ptr - reinterpret_cast<unsigned char*>(dst) > 0) // checking if this is the very first number in the bigger number
			*(dst_ptr - 1) |= buffer;

		*reinterpret_cast<uint16_t*>(dst_ptr) >>= by_bits;

		buffer = *(dst_ptr + sizeof(uint16_t));
		buffer &= ~(255 << by_bits);
		buffer <<= 8 - by_bits;

		dst_ptr += sizeof(uint16_t);
	}

	// 8 bit
	if (dst_n & 0b1) {
		if (dst_ptr - reinterpret_cast<unsigned char*>(dst) > 0)
			*(dst_ptr - 1) |= buffer;

		*dst_ptr >>= by_bits;
	}

}

int Base256uMath::bit_shift_right(
	const void* const src,
	std::size_t src_n,
	const void* const by,
	std::size_t by_n,
	void* const dst,
	std::size_t dst_n
) {
	// dividing 'by' by 8 and getting the number of bytes to shift

	std::size_t bytes = *reinterpret_cast<const std::size_t*>(by) * bool(by_n);
	bytes >>= 3;
	if (by_n > sizeof(std::size_t))
		bytes |= *(reinterpret_cast<const unsigned char*>(by) + sizeof(std::size_t)) & (unsigned char)0b111;

	auto code = byte_shift_right(src, src_n, bytes, dst, dst_n);

	// shifting by the number of bits we missed with byte shifting.

	unsigned char n_bits = (*reinterpret_cast<const unsigned char*>(by) * bool(by_n)) & (unsigned char)0b111;

	if (n_bits) {
#if defined(BASE256UMATH_FAST_OPERATORS) && !defined(__CUDACC__)
		bit_shift_right_fast(dst, dst_n, n_bits);
#else
		auto dst_ptr = reinterpret_cast<unsigned char*>(dst) - 1;
		auto dst_end = reinterpret_cast<unsigned char*>(dst) + dst_n;
		unsigned char mask = (((unsigned char)1 << n_bits) - 1);
		while (++dst_ptr < dst_end) {
			*dst_ptr >>= n_bits;
			*dst_ptr |= ((*(dst_ptr + 1) * (dst_ptr + 1 < dst_end)) & mask) << (8 - n_bits);
		}
#endif
	}

#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (!code && (src_n > dst_n))
		return ErrorCodes::TRUNCATED;
#endif

	return code;
}

int Base256uMath::bit_shift_right(
	const void* const src,
	std::size_t src_n,
	std::size_t by,
	void* const dst,
	std::size_t dst_n
) {
	// the 'by' parameter is capped at a number that is 8 times less than the highest index,
	// thus its highest value (for bytes) is 8 times less than std::size_t's highest value.

	auto code = byte_shift_right(src, src_n, by >> 3, dst, dst_n); // by / 8

	unsigned char n_bits = 0b111 & by;

	if (n_bits) {
#if defined(BASE256UMATH_FAST_OPERATORS) && !defined(__CUDACC__)
		bit_shift_right_fast(dst, dst_n, n_bits);
#else
		auto dst_ptr = reinterpret_cast<unsigned char*>(dst) - 1;
		auto dst_end = reinterpret_cast<unsigned char*>(dst) + dst_n;
		unsigned char mask = (((unsigned char)1 << n_bits) - 1);
		while (++dst_ptr < dst_end) {
			*dst_ptr >>= n_bits;
			*dst_ptr |= ((*(dst_ptr + 1) * (dst_ptr + 1 < dst_end)) & mask) << (8 - n_bits);
		}
#endif
	}
	return code;
}

int Base256uMath::bit_shift_right(
	void* const src,
	std::size_t src_n,
	std::size_t by
) {
	auto code = byte_shift_right(src, src_n, by >> 3);

	unsigned char n_bits = by & (unsigned char)0b111;

	if (n_bits) {
#if defined(BASE256UMATH_FAST_OPERATORS) && !defined(__CUDACC__)
		bit_shift_right_fast(src, src_n, by);
#else
		auto src_ptr = reinterpret_cast<unsigned char*>(src);
		unsigned char mask = (((unsigned char)1 << n_bits) - 1);
		while (src_ptr - src_n < src) {
			*src_ptr >>= n_bits;
			*src_ptr |= ((*(src_ptr + 1) * (1 + src_ptr - src_n < src)) & mask) << (8 - n_bits);
			src_ptr++;
		}
#endif
	}
	return 0;
}

int Base256uMath::bit_shift_right(
	void* const src,
	std::size_t src_n,
	const void* const by,
	std::size_t by_n
) {
	// dividing 'by' by 8 and getting the number of bytes to shift

	std::size_t bytes = *reinterpret_cast<const std::size_t*>(by) * bool(by_n);
	bytes >>= 3;
	if (by_n > sizeof(std::size_t))
		bytes |= *(reinterpret_cast<const unsigned char*>(by) + sizeof(std::size_t)) & (unsigned char)0b111;

	auto code = byte_shift_right(src, src_n, bytes);

	// shifting by the number of bits we missed with byte shifting.

	unsigned char n_bits = (*reinterpret_cast<const unsigned char*>(by) * bool(by_n)) & (unsigned char)0b111;

	if (n_bits) {
#if defined(BASE256UMATH_FAST_OPERATORS) && !defined(__CUDACC__)
		bit_shift_right_fast(src, src_n, n_bits);
#else
		// Fall back to just doing byte by byte

		auto src_ptr = reinterpret_cast<unsigned char*>(src);
		auto src_end = reinterpret_cast<unsigned char*>(src) + src_n;

		unsigned char mask = (((unsigned char)1 << n_bits) - 1);
		for (; src_ptr < src_end; src_ptr++) {
			*src_ptr >>= n_bits;
			*src_ptr |= ((*(src_ptr + 1) * (src_ptr + 1 < src_end)) & mask) << (8 - n_bits);
		}
#endif
	}
	return code;
}

__host__ __device__
inline void bitwise_and_fast(
	void* const left,
	const std::size_t& left_n,
	const void* const right,
	const std::size_t& right_n
) {
	auto l_ptr = reinterpret_cast<uint8_t*>(left);
	auto r_ptr = reinterpret_cast<const uint8_t*>(right);
#if BASE256UMATH_ARCHITECTURE == 64
	auto l_end = reinterpret_cast<uint8_t*>(left) + (left_n & ~(std::size_t)0b111);
	auto r_end = reinterpret_cast<const uint8_t*>(right) + (right_n & ~(std::size_t)0b111);
#elif BASE256UMATH_ARCHITECTURE == 32
	auto l_end = reinterpret_cast<uint8_t*>(left) + (left_n & ~(std::size_t)0b11);
	auto r_end = reinterpret_cast<const uint8_t*>(right) + (right_n & ~(std::size_t)0b11);
#endif

	while (l_ptr != l_end) {
		*reinterpret_cast<std::size_t*>(l_ptr) &= 
			*reinterpret_cast<const std::size_t*>(r_ptr) * (r_ptr != r_end);
		l_ptr += sizeof(std::size_t);
		r_ptr += sizeof(std::size_t) * (r_ptr != r_end);
	}

	r_end = reinterpret_cast<const uint8_t*>(right) + right_n;

	bool b;
#if BASE256UMATH_ARCHITECTURE >= 64
	b = left_n & 0b100;
	*reinterpret_cast<uint32_t*>(l_ptr) &=
		(*reinterpret_cast<uint32_t*>(l_ptr) * !b) |
			convert_to_size_t(r_ptr, MIN((r_end - r_ptr) * b, sizeof(uint32_t)));

	l_ptr += sizeof(uint32_t) * b;
	r_ptr += sizeof(uint32_t) * b;
#endif

	b = left_n & 0b10;
	*reinterpret_cast<uint16_t*>(l_ptr) &=
		(*reinterpret_cast<uint16_t*>(l_ptr) * !b) |
		convert_to_size_t(r_ptr, MIN((r_end - r_ptr) * b, sizeof(uint16_t)));

	l_ptr += sizeof(uint16_t) * b;
	r_ptr += sizeof(uint16_t) * b;

	b = left_n & 0b1;
	*l_ptr &=
		(*l_ptr * !b) |
		convert_to_size_t(r_ptr, MIN((r_end - r_ptr) * b, 1));
}

int Base256uMath::bitwise_and(
	const void* const left,
	std::size_t left_n,
	const void* const right,
	std::size_t right_n,
	void* const dst,
	std::size_t dst_n
) {
#if defined(BASE256UMATH_FAST_OPERATORS) && !defined(__CUDACC__)
	memset(dst, 0, dst_n);
	memcpy(dst, left, MIN(dst_n, left_n));
	bitwise_and_fast(dst, dst_n, right, right_n);
#else
	auto l_ptr = reinterpret_cast<const unsigned char*>(left);
	auto r_ptr = reinterpret_cast<const unsigned char*>(right);
	auto dst_ptr = reinterpret_cast<unsigned char*>(dst);
	for (std::size_t i = 0; i < dst_n; i++) {
		dst_ptr[i] = (l_ptr[i] * (i < left_n)) & (r_ptr[i] * (i < right_n));
	}
#endif

#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (dst_n < MIN(left_n, right_n))
		return ErrorCodes::TRUNCATED;
#endif
	return ErrorCodes::OK;
}

int Base256uMath::bitwise_and(
	void* const left,
	std::size_t left_n,
	const void* const right,
	std::size_t right_n
) {
#if defined(BASE256UMATH_FAST_OPERATORS) && !defined(__CUDACC__)
	bitwise_and_fast(left, left_n, right, right_n);
#else
	auto l_ptr = reinterpret_cast<unsigned char*>(left);
	auto r_ptr = reinterpret_cast<const unsigned char*>(right);
	for (std::size_t i = 0; i < left_n; i++) {
		l_ptr[i] &= (r_ptr[i] * (i < right_n));
	}
#endif
	return ErrorCodes::OK;
}

int Base256uMath::bitwise_or(
	const void* const left,
	std::size_t left_n,
	const void* const right,
	std::size_t right_n,
	void* const dst,
	std::size_t dst_n
) {
	auto l_ptr = reinterpret_cast<const unsigned char*>(left);
	auto r_ptr = reinterpret_cast<const unsigned char*>(right);
	auto dst_ptr = reinterpret_cast<unsigned char*>(dst);
	for (std::size_t i = 0; i < dst_n; i++) {
		dst_ptr[i] = (l_ptr[i] * (i < left_n)) | (r_ptr[i] * (i < right_n));
	}

#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (dst_n < MIN(left_n, right_n))
		return ErrorCodes::TRUNCATED;
#endif
	return ErrorCodes::OK;
}

int Base256uMath::bitwise_or(
	void* const left,
	std::size_t left_n,
	const void* const right,
	std::size_t right_n
) {
	auto l_ptr = reinterpret_cast<unsigned char*>(left);
	auto r_ptr = reinterpret_cast<const unsigned char*>(right);
	for (std::size_t i = 0; i < left_n; i++) {
		l_ptr[i] |= r_ptr[i] * (i < right_n);
	}
	return ErrorCodes::OK;
}

int Base256uMath::bitwise_xor(
	const void* const left,
	std::size_t left_n,
	const void* const right,
	std::size_t right_n,
	void* const dst,
	std::size_t dst_n
) {
	auto l_ptr = reinterpret_cast<const unsigned char*>(left);
	auto r_ptr = reinterpret_cast<const unsigned char*>(right);
	auto dst_ptr = reinterpret_cast<unsigned char*>(dst);
	for (std::size_t i = 0; i < dst_n; i++) {
		dst_ptr[i] = (l_ptr[i] * (i < left_n)) ^ (r_ptr[i] * (i < right_n));
	}
#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (dst_n < MIN(left_n, right_n))
		return ErrorCodes::TRUNCATED;
#endif
	return ErrorCodes::OK;
}

int Base256uMath::bitwise_xor(
	void* const left,
	std::size_t left_n,
	const void* const right,
	std::size_t right_n
) {
	auto l_ptr = reinterpret_cast<unsigned char*>(left);
	auto r_ptr = reinterpret_cast<const unsigned char*>(right);
	for (std::size_t i = 0; i < left_n; i++) {
		l_ptr[i] ^= r_ptr[i] * (i < right_n);
	}
	return ErrorCodes::OK;
}

__host__ __device__
void bitwise_not_fast(void* src, const std::size_t& src_n) {
	auto src_ptr = reinterpret_cast<unsigned char*>(src);

	if (src_n / (BASE256UMATH_ARCHITECTURE / 8)) {
		for (std::size_t i = 0; i < src_n / (BASE256UMATH_ARCHITECTURE / 8) - 1; i++) {
			*reinterpret_cast<std::size_t*>(src_ptr) = ~(*reinterpret_cast<std::size_t*>(src_ptr));
			src_ptr += sizeof(std::size_t);
		}
	}

#if BASE256UMATH_ARCHITECTURE >= 64
	if (src_n & 0b1000) {
		*reinterpret_cast<uint64_t*>(src_ptr) = ~(*reinterpret_cast<uint64_t*>(src_ptr));
		src_ptr += sizeof(uint64_t);
	}
#endif

	if (src_n & 0b100) {
		*reinterpret_cast<uint32_t*>(src_ptr) = ~(*reinterpret_cast<uint32_t*>(src_ptr));
		src_ptr += sizeof(uint32_t);
	}

	if (src_n & 0b10) {
		*reinterpret_cast<uint16_t*>(src_ptr) = ~(*reinterpret_cast<uint16_t*>(src_ptr));
		src_ptr += sizeof(uint16_t);
	}

	if (src_n & 0b1) {
		*reinterpret_cast<uint8_t*>(src_ptr) = ~(*reinterpret_cast<uint8_t*>(src_ptr));
	}
}

int Base256uMath::bitwise_not(
	void* const src,
	std::size_t src_n
) {
	auto ptr = reinterpret_cast<unsigned char*>(src) + src_n;
	while (--ptr != reinterpret_cast<unsigned char*>(src) - 1) {
		*ptr = ~(*ptr);
	}
	return ErrorCodes::OK;
}

// backend implementation
inline void byte_shift_left_impl(
	const void* const src,
	const std::size_t& src_n,
	const std::size_t& by,
	void* const dst,
	const std::size_t& dst_n
) {
	memcpy(
		reinterpret_cast<unsigned char*>(dst) + by * bool(dst_n),
		src,
		MIN(src_n, dst_n) - by * bool(dst_n)
	);
}

int Base256uMath::byte_shift_left(
	const void* const src,
	std::size_t src_n,
	std::size_t by,
	void* const dst,
	std::size_t dst_n
) {
	memset(dst, 0, dst_n);
	if (src_n <= by)
		return ErrorCodes::OK;
	 
	// src_n > by and dst_n > by is true from this point on

	byte_shift_left_impl(src, src_n, by, dst, dst_n);

#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (src_n > dst_n + by)
		return ErrorCodes::TRUNCATED;
#endif
	return ErrorCodes::OK;
}

__host__ __device__
void _memmove(void* dst, void* src, std::size_t n) {
	if (!n)
		return;
	if (src > dst) {
		for (std::size_t i = 0; i < n; i++) {
			reinterpret_cast<unsigned char*>(dst)[i] = reinterpret_cast<unsigned char*>(src)[i];
		}
	}
	else if (src < dst) {
		for (std::size_t i = n - 1; i < n; i--) {
			reinterpret_cast<unsigned char*>(dst)[i] = reinterpret_cast<unsigned char*>(src)[i];
		}
	}
}

// backend implementation
inline void byte_shift_left_impl(
	void* const src,
	const std::size_t& src_n,
	const std::size_t& by
) {
#ifndef __CUDACC__
	memmove(reinterpret_cast<unsigned char*>(src) + by, src, src_n - by);
#else
	_memmove(reinterpret_cast<unsigned char*>(src) + by, src, src_n - by);
#endif
	memset(reinterpret_cast<unsigned char*>(src), 0, by);
}

int Base256uMath::byte_shift_left(
	void* const src,
	std::size_t src_n,
	std::size_t by
) {
	if (src_n <= by) {
		memset(src, 0, src_n);
		return ErrorCodes::OK;
	}

	// src_n > by is true from this point on

	byte_shift_left_impl(src, src_n, by);

	return ErrorCodes::OK;
}

int Base256uMath::byte_shift_right(
	const void* const src,
	std::size_t src_n,
	std::size_t by,
	void* const dst,
	std::size_t dst_n
) {
	memset(dst, 0, dst_n);
	if (src_n <= by)
		return ErrorCodes::OK;

	memcpy(dst, reinterpret_cast<const unsigned char*>(src) + by, MIN(src_n - by, dst_n));

#if !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
	if (src_n - by > dst_n)
		return ErrorCodes::TRUNCATED;
#endif
	return ErrorCodes::OK;
}

int Base256uMath::byte_shift_right(
	void* const src,
	std::size_t src_n,
	std::size_t by
) {
	if (src_n <= by) {
		memset(src, 0, src_n);
		return ErrorCodes::OK;
	}

	// src_n > by is true from this point on

#ifndef __CUDACC__
	memmove(src, reinterpret_cast<unsigned char*>(src) + by, src_n - by);
#else
	_memmove(src, reinterpret_cast<unsigned char*>(src) + by, src_n - by);
#endif
	memset(reinterpret_cast<unsigned char*>(src) + (src_n - by), 0, by);

	return ErrorCodes::OK;
}

