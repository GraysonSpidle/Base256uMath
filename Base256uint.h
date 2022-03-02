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

*/

#ifndef __BASE256UINT_H__
#define __BASE256UINT_H__

#include <cstdlib> // std::size_t
#include "Base256uMath.h"

/* 


*/
struct Base256uint {
public:
	typedef Base256uMath::ErrorCodes ErrorCodes;

	static const Base256uint max(const Base256uint& left, const Base256uint& right);
	static const Base256uint max(const Base256uint& left, std::size_t right);
	static const Base256uint min(const Base256uint& left, const Base256uint& right);
	static const Base256uint min(const Base256uint& left, std::size_t right);

	static Base256uint log2(const Base256uint& other);
	static std::size_t log256(const Base256uint& other);

	int error = ErrorCodes::OK;
	std::size_t size;
	unsigned char* raw;

	Base256uint() = delete; // No default constructor.

	Base256uint(std::size_t num);
	Base256uint(unsigned char nums[]);
	Base256uint(const Base256uint& other);
private:
	Base256uint(void* raw, std::size_t size);
public:
	~Base256uint();

	void resize(); // Kinda like the String.trim() method from Java. Trims unused bytes from the big endian.

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