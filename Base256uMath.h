/* Base256uMath.h
Author: Grayson Spidle

Houses the Base256uMath namespace and all its function declarations.

These functions are designed with NVIDIA CUDA in mind. Meaning it only uses
typedefs, functions, etc that are native to nvcc. Additionally, none of these
functions throw	exceptions and instead return error codes. This is also able to
be compiled on msvc or g++. Other C++ compilers outside of nvcc, msvc, and g++ 
are not taken into account and may bring unexpected results or compatibility issues.

Targeted for the C++14 standard and big endian machines.
*/

#ifndef __BASE256UMATH_H__
#define __BASE256UMATH_H__

#ifndef __NVCC__
#include <cstdlib> // std::size_t
#endif

/* 
Namespace that houses functions to satisfy most operators that primitive numbers have.
Primarily designed to work with numbers bigger than the computer's register size, but it
can work with small numbers as well.
*/
namespace Base256uMath {

	// a typedef to accomodate for the highest index of a bit. Mainly used for the log2 function.
	using bit_size_t = unsigned char[sizeof(std::size_t) + 1];

	/*  Enumeration of all error codes that all the functions in this namespace (that return error codes) can return.
	Warnings (non fatal errors) are all positive numbers and fatal errors are all negative numbers.

	You can suppress the TRUNCATED warning code by defining the macro BASE256UMATH_SUPPRESS_TRUNCATED_CODE as 1.
	By default, it is not suppressed. Even if the macro is not defined.
	I did this because the warning code doesn't really come into play that much. Most of the time it's warning you
	of nothing. However, when debugging, it is actually kind of useful *sometimes*. So there's that.
	*/
	enum ErrorCodes {

		OOM = -2, // Out of memory
		DIVIDE_BY_ZERO = -1, // Division by zero
		OK = 0, // Nothing went wrong
		FLOW = 1, // Overflow/Underflow warning
#if !defined(BASE256UMATH_SUPPRESS_TRUNCATED_CODE) || !BASE256UMATH_SUPPRESS_TRUNCATED_CODE
		TRUNCATED = 2 // Output data was truncated due to its size not being adequate enough to accomodate
#endif
	};

	/* Essentially, this is the opposite of the bool() operator.
	Parameters
	* src : pointer to the number to be evaluated. Read only.
	* src_n : the size of the number in bytes.
	Returns
	* true : if all bytes in src are 0.
	* false : if any byte in src is not 0.
	*/
	bool is_zero(
		const void* const src,
		std::size_t src_n
	);

	/* Compares two given numbers.
	Parameters:
	* left : pointer to the first number. Read only.
	* left_n : the size of the first number in bytes.
	* right : pointer to the second number. Read only.
	* right_n : the size of the second number in bytes.
	
	Returns:
	* 1 : if left is greater than right.
	* 0 : if left is equal to right.
	* -1 : if left is less than right.
	
	Unlike memcmp, you can rely on the return values being 1, 0, or -1.
	So it is safe to use in a switch-case block.
	*/
	int compare(
		const void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n
	);

	/* Compares two given numbers.
	Parameters:
	* left : pointer to the first number. Read only.
	* left_n : the size of the first number in bytes.
	* right : the second number. 

	Returns:
	* 1 : if left is greater than right.
	* 0 : if left is equal to right.
	* -1 : if left is less than right.

	Unlike memcmp, you can rely on the return values being 1, 0, or -1.
	So it is safe to use in a switch-case block.
	*/
	int compare(
		const void* const left,
		std::size_t left_n,
		std::size_t right
	);

	/* Given two values, finds the greatest value.
	Parameters:
	* left : pointer to the first number. Read only.
	* left_n : the size of the first number in bytes.
	* right : pointer to the second number. Read only.
	* right_n : the size of the second number in bytes.
	
	Returns:
	* right : if right is greater than left.
	* left : if left is greater than right.
	* left : if left is equal to right.
	*/
	const void* const max(
		const void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n
	);

	// This is the non-const version, it behaves just like the const version.
	void* const max(
		void* const left,
		std::size_t left_n,
		void* const right,
		std::size_t right_n
	);

	/* Given two values, finds the least value.
	Parameters:
	* left : pointer to the first number. Read only.
	* left_n : the size of the first number in bytes.
	* right : pointer to the second number. Read only.
	* right_n : the size of the second number in bytes.

	Returns:
	* left : if left is less than right.
	* right : if right is less than left.
	* left : if left is equal to right.
	*/
	const void* const min(
		const void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n
	);

	// This is the non-const version, it behaves just like the const version.
	void* const min(
		void* const left,
		std::size_t left_n,
		void* const right,
		std::size_t right_n
	);

	/* In-place addition of 1 to a given number.
	Parameters:
	* block : pointer to the number. Will not reassign where it points.
	* n : the size of the number in bytes.
	Returns an error code, here are the possible errors it can return:
	* OK : everything went well.
	* FLOW : integer overflow warning. Not fatal.
	*/
	int increment(
		void* const block,
		std::size_t n
	);

	/* In-place subtraction of 1 from a given number.
	Parameters:
	* block : pointer to the number. Will not reassign where it points.
	* n : the size of the number in bytes.
	Returns an error code, here are the possible error codes it can return:
	* OK : everything went well.
	* FLOW : integer underflow warning. Not fatal.
	*/
	int decrement(
		void* const block,
		std::size_t n
	);

	/* Adds two given numbers and stores the sum in another number.
	Parameters:
	* left : pointer to the first addend. Read only.
	* left_n : the size of the first addend in bytes.
	* right : pointer to the second addend. Read only.
	* right_n : the size of the second addend in bytes.
	* dst : pointer to the sum. Pointer is read only.
	* dst_n : the size of the sum in bytes.
	
	Returns an error code, here are the possible error codes it can return:
	* OK : everything went well.
	* FLOW : integer overflow warning. Not fatal.
	* TRUNCATED : when dst_n < MAX(left_n, right_n). Not fatal.
	*/
	int add(
		const void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n,
		void* const dst,
		std::size_t dst_n
	);

	/* Adds two given numbers and stores the sum in another number.
	Parameters:
	* left : pointer to the first addend. Read only.
	* left_n : the size of the first addend in bytes.
	* right : the second addend.
	* dst : pointer to the sum. Pointer is read only.
	* dst_n : the size of the sum in bytes.

	Returns an error code, here are the possible error codes it can return:
	* OK : everything went well.
	* FLOW : integer overflow warning. Not fatal.
	* TRUNCATED : when dst_n < MAX(left_n, sizeof(right)). Not fatal.
	*/
	int add(
		const void* const left,
		std::size_t left_n,
		std::size_t right,
		void* const dst,
		std::size_t dst_n
	);

	/* Adds two given numbers and stores the sum into the first addend. 
	This is commonly referred to as "in place" addition.
	Parameters:
	* left : pointer to the first addend. Pointer is read only.
	* left_n : the size of the first addend in bytes.
	* right : pointer to the second addend. Read only.
	* right_n : the size of the second addend in bytes.

	Returns an error code, here are the possible error codes it can return:
	* OK : everything went well.
	* FLOW : integer overflow warning. Not fatal.
	* OOM : required additional memory but was denied. No modifications have occurred.
	*/
	int add(
		void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n
	);

	int add(
		void* const left,
		std::size_t left_n,
		std::size_t right
	);

	/* Subtracts two given numbers and stores the difference into another number.
	Parameters:
	* left : pointer to the minuend. Read only.
	* left_n : the size of the minuend in bytes.
	* right : pointer to the subtrahend. Read only.
	* right_n : the size of the subtrahend in bytes.
	* dst : pointer to the difference. Pointer is read only.
	* dst_n : the size of the difference in bytes.

	Returns an error code, here are the possible error codes it can return:
	* OK : everything went well.
	* FLOW : integer underflow warning. Not fatal.
	* TRUNCATED : when dst_n < MAX(left_n, right_n). Not fatal.
	*/
	int subtract(
		const void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n,
		void* const dst,
		std::size_t dst_n
	);

	int subtract(
		const void* const left,
		std::size_t left_n,
		std::size_t right,
		void* const dst,
		std::size_t dst_n
	);

	/* Subtracts two given numbers and stores the difference into the minuend.
	This is commonly referred to as "in place" subtraction.
	Parameters:
	* left : pointer to the minuend. Pointer is read only.
	* left_n : the size of the minuend in bytes.
	* right : pointer to the subtrahend. Read only.
	* right_n : the size of the subtrahend in bytes.

	Returns an error code, here are the possible error codes it can return:
	* OK : everything went well.
	* FLOW : integer underflow warning. Not fatal.
	* TRUNCATED : when left_n < right_n. Not fatal.
	*/
	int subtract(
		void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n
	);

	/* Subtracts two given numbers and stores the difference into the minuend.
	This is commonly referred to as "in place" subtraction.
	Parameters:
	* left : pointer to the minuend. Pointer is read only.
	* left_n : the size of the minuend in bytes.
	* right : the subtrahend.

	Returns an error code, here are the possible error codes it can return:
	* OK : everything went well.
	* FLOW : integer underflow warning. Not fatal.
	* TRUNCATED : when left_n < sizeof(right). Not fatal.
	*/
	int subtract(
		void* const left,
		std::size_t left_n,
		std::size_t right
	);

	/* Multiplies two given numbers and stores the product into another number.
	Parameters:
	* left : pointer to the multiplicand. Read only.
	* left_n : the size of the multiplicand in bytes.
	* right : pointer to the multiplier. Read only.
	* right_n : the size of the multiplier in bytes.
	* dst : pointer to the product. Pointer is read only.
	* dst_n : the size of the product in bytes.
	
	Returns an error code, here are the possible error codes it can return:
	* OK : everything went well.
	* TRUNCATED : when !bool(dst_n) || dst_n < MIN(left_n, right_n). Not fatal.
	* OOM : required additional memory but was denied. No modifications have occurred.
	*/
	int multiply(
		const void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n,
		void* const dst,
		std::size_t dst_n
	);

	/* Multiplies two given numbers and stores the product into another number.
	Parameters:
	* left : pointer to the multiplicand. Read only.
	* left_n : the size of the multiplicand in bytes.
	* right : the multiplier.
	* dst : pointer to the product. Pointer is read only.
	* dst_n : the size of the product in bytes.

	Returns an error code, here are the possible error codes it can return:
	* OK : everything went well.
	* TRUNCATED : when !bool(dst_n) || dst_n < MIN(left_n, sizeof(right)). Not fatal.
	* OOM : required additional memory but was denied. No modifications have occurred.
	*/
	int multiply(
		const void* const left,
		std::size_t left_n,
		std::size_t right,
		void* const dst,
		std::size_t dst_n
	);

	/* Multiplies two given numbers and stores the product into the multiplicand.
	This is commonly referred to as "in place" multiplication.
	Parameters:
	* left : pointer to the multiplicand. Pointer is read only.
	* left_n : the size of the multiplicand in bytes.
	* right : pointer to the multiplier. Read only.
	* right_n : the size of the multiplier in bytes.

	Returns an error code, here are the possible error codes it can return:
	* OK : everything went well.
	* TRUNCATED : when !bool(dst_n) || dst_n < MIN(left_n, right_n). Not fatal.
	* OOM : required additional memory but was denied. No modifications have occurred.
	*/
	int multiply(
		void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n
	);

	int multiply(
		void* const left,
		std::size_t left_n,
		std::size_t right
	);

	/* Divides two given numbers and stores the quotient and remainder into their own separate numbers.
	Parameters:
	* left : pointer to the dividend. Read only.
	* left_n : the size of the dividend in bytes.
	* right : pointer to the divisor. Read only.
	* right_n : the size of the divisor in bytes.
	* dst : pointer to the quotient. Pointer is read only.
	* dst_n : the size of the quotient in bytes (should be of identical size to left_n).
	* remainder : pointer to the remainder. Pointer is read only.
	* remainder_n : the size of the remainder in bytes (should be of identical size to left_n).
	
	Returns an error code, here are the possible error codes it can return:
	* OK : everything went well.
	* DIVIDE_BY_ZERO : you tried to divide by zero. No modifications have occurred.
	* OOM : required additional memory but was denied.
	*/
	int divide(
		const void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n,
		void* const dst,
		std::size_t dst_n,
		void* const remainder,
		std::size_t remainder_n
	);

	int divide(
		const void* const left,
		std::size_t left_n,
		std::size_t right,
		void* const dst,
		std::size_t dst_n,
		void* const remainder,
		std::size_t remainder_n
	);

	int divide(
		void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n,
		void* const remainder,
		std::size_t remainder_n
	);

	int divide(
		void* const left,
		std::size_t left_n,
		std::size_t right,
		void* const remainder,
		std::size_t remainder_n
	);

	/* Divides two given numbers and stores the quotient in another number and discards the remainder (but still must allocate memory for it).
	Parameters:
	* left : pointer to the dividend. Read only.
	* left_n : the size of the dividend in bytes.
	* right : pointer to the divisor. Read only.
	* right_n : the size of the divisor in bytes.
	* dst : pointer to the quotient. Pointer is read only.
	* dst_n : the size of the quotient in bytes (should be of identical size to left_n).

	Returns an error code, here are the possible error codes it can return:
	* OK : everything went well.
	* DIVIDE_BY_ZERO : you tried to divide by zero. No modifications have occurred.
	* OOM : required additional memory but was denied.
	*/
	int divide_no_mod(
		const void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n,
		void* const dst,
		std::size_t dst_n
	);

	int divide_no_mod(
		const void* const left,
		std::size_t left_n,
		std::size_t right,
		void* const dst,
		std::size_t dst_n
	);

	int divide_no_mod(
		void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n
	);

	int divide_no_mod(
		void* const left,
		std::size_t left_n,
		std::size_t right
	);

	/* Divides two given numbers and stores the remainder in another number and discards the quotient (but still must allocate memory for it?).
	Parameters:
	* left : pointer to the dividend. Read only.
	* left_n : the size of the dividend in bytes.
	* right : pointer to the divisor. Read only.
	* right_n : the size of the divisor in bytes.
	* dst : pointer to the remainder. Pointer is read only.
	* dst_n : the size of the remainder in bytes (should be of identical size to left_n).

	Returns an error code, here are the possible error codes it can return:
	* OK : everything went well.
	* DIVIDE_BY_ZERO : you tried to divide by zero. No modifications have occurred.
	* OOM : required additional memory but was denied.
	*/
	int mod(
		const void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n,
		void* const dst,
		std::size_t dst_n
	);

	int mod(
		void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n
	);

	/* Finds the most significant bit in a given number and stores its index in another number.
	Parameters:
	* src : pointer to the number in which to find the most significant bit. Read only.
	* src_n : the size of the number in bytes.
	* dst : pointer to the number where the index will be stored. Pointer is read only.
	* dst_n : the size of the destination number in bytes.
	
	Returns an error code, here are the possible error codes it can return:
	* OK : everything went well.
	* DIVIDE_BY_ZERO : you tried to take the log of zero. No modifications have occurred?
	* OOM?
	*/
	int log2(
		const void* const src,
		std::size_t src_n,
		void* const dst,
		std::size_t dst_n
	);

	int log2(
		const void* const src,
		std::size_t src_n,
		bit_size_t dst
	);

	/* Finds the most significant byte in a given number and stores its index in another number.
	The destination is a size_t because it is physically impossible for the answer to be larger than that, since the computer would not be able to see that memory.
	Parameters:
	* src : pointer to the number in which to find the most significant byte. Read only.
	* src_n : the size of the number in bytes.
	* dst : pointer to the largest primitive unsigned integer where the index will be stored. Pointer is read only.
	
	Returns an error code, here are the possible error codes it can return:
	* OK : everything went well.
	* DIVIDE_BY_ZERO : you tried to take the log of zero. No modifications have occurred.
	*/
	int log256(
		const void* const src,
		std::size_t src_n,
		std::size_t* const dst
	);

	int bitwise_and(
		const void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n,
		void* const dst,
		std::size_t dst_n
	);

	int bitwise_and(
		void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n
	);

	int bitwise_or(
		const void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n,
		void* const dst,
		std::size_t dst_n
	);

	int bitwise_or(
		void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n
	);

	int bitwise_xor(
		const void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n,
		void* const dst,
		std::size_t dst_n
	);

	int bitwise_xor(
		void* const left,
		std::size_t left_n,
		const void* const right,
		std::size_t right_n
	);

	int bitwise_not(
		void* const src,
		std::size_t src_n
	);

	int bit_shift_left(
		const void* const src,
		std::size_t src_n,
		const void* const by,
		std::size_t by_n,
		void* const dst,
		std::size_t dst_n
	);

	// in-place
	int bit_shift_left(
		void* const src,
		std::size_t src_n,
		const void* const by,
		std::size_t by_n
	);

	int bit_shift_left(
		const void* const src,
		std::size_t src_n,
		std::size_t by,
		void* const dst,
		std::size_t dst_n
	);

	// in-place
	int bit_shift_left(
		void* const src,
		std::size_t src_n,
		std::size_t by
	);

	int bit_shift_right(
		const void* const src,
		std::size_t src_n,
		const void* const by,
		std::size_t by_n,
		void* const dst,
		std::size_t dst_n
	);

	int bit_shift_right(
		const void* const src,
		std::size_t src_n,
		std::size_t by,
		void* const dst,
		std::size_t dst_n
	);

	// in-place
	int bit_shift_right(
		void* const src,
		std::size_t src_n,
		const void* const by,
		std::size_t by_n
	);

	// in-place
	int bit_shift_right(
		void* const src,
		std::size_t src_n,
		std::size_t by
	);

	int byte_shift_left(
		const void* const src,
		std::size_t src_n,
		std::size_t by,
		void* const dst,
		std::size_t dst_n
	);

	// in-place
	int byte_shift_left(
		void* const src,
		std::size_t src_n,
		std::size_t by
	);

	int byte_shift_right(
		const void* const src,
		std::size_t src_n,
		std::size_t by,
		void* const dst,
		std::size_t dst_n
	);

	// in-place
	int byte_shift_right(
		void* const src,
		std::size_t src_n,
		std::size_t by
	);
}


#endif // __BASE256UMATH_H__