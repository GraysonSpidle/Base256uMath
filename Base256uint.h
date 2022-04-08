/* Base256uint.h
Author: Grayson Spidle

Declares the Base256uint struct for the people that
prefer the object oriented approach. I made it so 
you don't have to.

Due to the limitations of operators in C++, the division
operator does not give (in addition to the quotient)
the remainder, which forces you to do 1 more division
than if you didn't go with the object oriented approach. 
A sacrifice that must be made for the cleaner code this
approach brings. Hopefully the sacrifice is worth it to you.

I'm gonna be honest here, I have not tested this. I just wrote
this in case other people wanted to use this library and preferred OOP.

Also, don't use this with CUDA, it isn't designed for that.
*/

#ifndef __BASE256UINT_H__
#define __BASE256UINT_H__

#include <cstdlib> // std::size_t
#include "Base256uMath.h"

/* Base256uint object.
Essentially the OOP version of the regular library.

The documentation here is pretty sparse because all the
documentation is in the core files, so just go there.

The only functions documented are ones that don't appear
in the core library.
*/
struct Base256uint {
public:
	typedef Base256uMath::ErrorCodes ErrorCodes;

	static const Base256uint max(const Base256uint& left, const Base256uint& right);
	static Base256uint max(Base256uint& left, Base256uint& right);
	static const Base256uint min(const Base256uint& left, const Base256uint& right);
	static Base256uint min(Base256uint& left, Base256uint& right);

	static Base256uint log2(const Base256uint& other);
	static std::size_t log256(const Base256uint& other);

	std::size_t size; // the size of raw
	unsigned char* raw = nullptr; // the raw bytes
	int error = ErrorCodes::OK; // error code

	// alignment
#if BASE256UMATH_ARCHITECTURE == 64
	// 8 + 8 + 4 = 20
	// for alignment. DO NOT RELY ON THIS TO HOLD DATA.
	unsigned char unused[4]; 
#elif BASE256UMATH_ARCHITECTURE == 32
	// 4 + 4 + 2 = 10
	// for alignment. DO NOT RELY ON THIS TO HOLD DATA.
	unsigned char unused[6];
#endif

	Base256uint() = delete; // No default constructor.

	Base256uint(std::size_t num);
	// if you want to make the previous constructor explicit, then go right ahead.
	// I wrote the implementations so that you don't have to do anything extra. 
	// You're welcome :)

	Base256uint(unsigned char nums[]);
	// That wasn't variadic because I hate how the variadic thing looks for this.

	// copy constructor
	Base256uint(const Base256uint& other);

	// move constructor
	Base256uint(Base256uint&& other);
	// wrapper constructor (do not use on stack objects)
	Base256uint(void* raw, std::size_t size);

	~Base256uint();

	// Kinda like the String.trim() method from Java. Trims unused bytes from the big endian.
	void resize();

	// Comparison Operators

	operator bool() const;

	bool operator==(const Base256uint& other) const;
	bool operator==(std::size_t other) const;

	bool operator!=(const Base256uint& other) const;
	bool operator!=(std::size_t other) const;

	bool operator<(const Base256uint& other) const;
	bool operator<(std::size_t other) const;

	bool operator<=(const Base256uint& other) const;
	bool operator<=(std::size_t other) const;

	bool operator>(const Base256uint& other) const;
	bool operator>(std::size_t other) const;

	bool operator>=(const Base256uint& other) const;
	bool operator>=(std::size_t other) const;

	// Bitwise Operators

	Base256uint operator&(const Base256uint& other) const;
	Base256uint operator|(const Base256uint& other) const;
	Base256uint operator^(const Base256uint& other) const;

	void operator&=(const Base256uint& other);
	void operator|=(const Base256uint& other);
	void operator^=(const Base256uint& other);
	void operator~();

	Base256uint operator<<(const Base256uint& by) const;
	Base256uint operator<<(std::size_t by) const;
	Base256uint operator>>(const Base256uint& by) const;
	Base256uint operator>>(std::size_t by) const;

	void operator<<=(const Base256uint& by);
	void operator<<=(std::size_t by);
	void operator>>=(const Base256uint& by);
	void operator>>=(std::size_t by);

	// Arithmetic Operators

	void operator++();
	void operator--();

	Base256uint operator+(const Base256uint& other) const;
	Base256uint operator+(std::size_t other) const;

	Base256uint operator-(const Base256uint& other) const;
	Base256uint operator-(std::size_t other) const;

	Base256uint operator*(const Base256uint& other) const;
	Base256uint operator*(std::size_t other) const;

	Base256uint operator/(const Base256uint& other) const;
	Base256uint operator/(std::size_t other) const;

	Base256uint operator%(const Base256uint& other) const;
	Base256uint operator%(std::size_t other) const;

	void operator+=(const Base256uint& other);
	void operator+=(std::size_t other);

	void operator-=(const Base256uint& other);
	void operator-=(std::size_t other);

	void operator*=(const Base256uint& other);
	void operator*=(std::size_t other);

	void operator/=(const Base256uint& other);
	void operator/=(std::size_t other);

	void operator%=(const Base256uint& other);
	void operator%=(std::size_t other);
};

#endif // __BASE256UINT_H__