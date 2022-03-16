#include "Base256uint.h"
#include <cstring>

#ifndef MIN(a,b)
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX(a,b)
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

Base256uint::Base256uint(std::size_t num) : size(sizeof(std::size_t)) {
	raw = reinterpret_cast<unsigned char*>(calloc(sizeof(std::size_t), 1));
	if (!raw) {
		error = ErrorCodes::OOM;
		return;
	}
	memcpy(raw, &num, sizeof(std::size_t));
}

Base256uint::Base256uint(unsigned char nums[]) {
	raw = nums;
	size = sizeof(nums);
}

Base256uint::Base256uint(const Base256uint& other) : size(other.size) {
	if (!raw) {
		raw = reinterpret_cast<unsigned char*>(malloc(other.size));
	}
	else {
		void* temp = realloc(raw, other.size);
		if (temp) {
			raw = reinterpret_cast<unsigned char*>(temp);
		}
		else {
			error = ErrorCodes::OOM;
			return;
		}
	}
	memcpy(raw, other.raw, other.size); // raw isn't 0, shut up compiler.
}

Base256uint::Base256uint(void* raw, std::size_t size) : size(size) {
	this->raw = reinterpret_cast<unsigned char*>(raw);
	error = ErrorCodes::OK;
}

Base256uint::~Base256uint() {
	if (raw)
		free(raw);
}

void Base256uint::resize() {
	std::size_t sig_byte;
	if (
		!Base256uMath::log256(raw, size, &sig_byte) &&
		sig_byte < size - 1
	) {
		void* temp = realloc(raw, sig_byte + 1);
		if (temp)
			raw = reinterpret_cast<unsigned char*>(temp);
		else
			error = Base256uMath::ErrorCodes::OOM;
	}
	error = Base256uMath::ErrorCodes::OK;
}

Base256uint::operator bool() const {
	return !Base256uMath::is_zero(raw, size);
}

bool Base256uint::operator==(const Base256uint& other) const {
	return Base256uMath::compare(raw, size, other.raw, other.size) == 0;
}

bool Base256uint::operator==(std::size_t other) const {
	return Base256uMath::compare(raw, size, &other, sizeof(other)) == 0;
}

bool Base256uint::operator!=(const Base256uint& other) const {
	return !operator==(other);
}

bool Base256uint::operator!=(std::size_t other) const {
	return !operator==(other);
}

bool Base256uint::operator<(const Base256uint& other) const {
	return Base256uMath::compare(raw, size, other.raw, other.size) < 0;
}

bool Base256uint::operator<(std::size_t other) const {
	return Base256uMath::compare(raw, size, &other, sizeof(other)) < 0;
}

bool Base256uint::operator<=(const Base256uint& other) const {
	return operator<(other) || operator==(other);
}

bool Base256uint::operator<=(std::size_t other) const {
	return operator<(other) || operator==(other);
}

bool Base256uint::operator>(const Base256uint& other) const {
	return Base256uMath::compare(raw, size, other.raw, other.size) < 0;
}

bool Base256uint::operator>(std::size_t other) const {
	return Base256uMath::compare(raw, size, &other, sizeof(other)) < 0;
}

bool Base256uint::operator>=(const Base256uint& other) const {
	return operator<(other) || operator==(other);
}

bool Base256uint::operator>=(std::size_t other) const {
	return operator<(other) || operator==(other);
}

Base256uint Base256uint::operator&(const Base256uint& other) const {
	Base256uint output { MIN(size, other.size) };
	output.error = Base256uMath::bitwise_and(raw, size, other.raw, other.size, output.raw, output.size);
	return output;
}

Base256uint Base256uint::operator|(const Base256uint& other) const {
	Base256uint output { MAX(size, other.size) };
	output.error = Base256uMath::bitwise_or(raw, size, other.raw, other.size, output.raw, output.size);
	return output;
}

Base256uint Base256uint::operator^(const Base256uint& other) const {
	Base256uint output { MIN(size, other.size) };
	output.error = Base256uMath::bitwise_xor(raw, size, other.raw, other.size, output.raw, output.size);
	return output;
}

void Base256uint::operator&=(const Base256uint& other) {
	error = Base256uMath::bitwise_and(raw, size, other.raw, other.size);
}

void Base256uint::operator|=(const Base256uint& other) {
	error = Base256uMath::bitwise_or(raw, size, other.raw, other.size);
}

void Base256uint::operator^=(const Base256uint& other) {
	error = Base256uMath::bitwise_xor(raw, size, other.raw, other.size);
}

void Base256uint::operator~() {
	error = Base256uMath::bitwise_not(raw, size);
}

Base256uint Base256uint::operator<<(const Base256uint& by) const {
	Base256uint output { size };
	output.error = Base256uMath::bit_shift_left(raw, size, by.raw, by.size, output.raw, output.size);	
	return output;
}

Base256uint Base256uint::operator<<(std::size_t by) const {
	Base256uint output { size };
	output.error = Base256uMath::bit_shift_left(raw, size, by, output.raw, output.size);
	return output;
}

Base256uint Base256uint::operator>>(const Base256uint& by) const {
	Base256uint output{ size };
	output.error = Base256uMath::bit_shift_right(raw, size, by.raw, by.size, output.raw, output.size);
	return output;
}

Base256uint Base256uint::operator>>(std::size_t by) const {
	Base256uint output{ size };
	output.error = Base256uMath::bit_shift_right(raw, size, by, output.raw, output.size);
	return output;
}

void Base256uint::operator<<=(const Base256uint& by) {
	error = Base256uMath::bit_shift_left(raw, size, by.raw, by.size);
}

void Base256uint::operator<<=(std::size_t by) {
	error = Base256uMath::bit_shift_left(raw, size, by);
}

void Base256uint::operator>>=(const Base256uint& by) {
	error = Base256uMath::bit_shift_right(raw, size, by.raw, by.size);
}

void Base256uint::operator>>=(std::size_t by) {
	error = Base256uMath::bit_shift_right(raw, size, by);
}

void Base256uint::operator++() {
	error = Base256uMath::increment(raw, size);	
}

void Base256uint::operator--() {
	error = Base256uMath::decrement(raw, size);
}

Base256uint Base256uint::operator+(const Base256uint& other) const {
	Base256uint output { MAX(size, other.size) };

	output.error = Base256uMath::add(
		raw, size,
		other.raw, other.size,
		output.raw, output.size
	);

	return output;
}

Base256uint Base256uint::operator+(std::size_t other) const {
	Base256uint output{ MAX(size, sizeof(other)) };

	output.error = Base256uMath::add(
		raw, size,
		&other, sizeof(other),
		output.raw, output.size
	);

	return output;
}

Base256uint Base256uint::operator-(const Base256uint& other) const {
	Base256uint output{ MAX(size, other.size) };

	output.error = Base256uMath::subtract(
		raw, size,
		other.raw, other.size,
		output.raw, output.size
	);

	return output;
}

Base256uint Base256uint::operator-(std::size_t other) const {
	Base256uint output{ MAX(size, sizeof(other)) };

	output.error = Base256uMath::subtract(
		raw, size,
		&other, sizeof(other),
		output.raw, output.size
	);

	return output;
}

Base256uint Base256uint::operator*(const Base256uint& other) const {
	Base256uint output{ MAX(size, other.size) };

	output.error = Base256uMath::multiply(
		raw, size,
		other.raw, other.size,
		output.raw, output.size
	);

	return output;
}

Base256uint Base256uint::operator*(std::size_t other) const {
	Base256uint output{ MAX(size, sizeof(other)) };

	output.error = Base256uMath::multiply(
		raw, size,
		&other, sizeof(other),
		output.raw, output.size
	);

	return output;
}

Base256uint Base256uint::operator/(const Base256uint& other) const {
	Base256uint output{ MAX(size, other.size) };

	output.error = Base256uMath::divide(
		raw, size,
		other.raw, other.size,
		output.raw, output.size
	);

	return output;
}

Base256uint Base256uint::operator/(std::size_t other) const {
	Base256uint output{ MAX(size, sizeof(other)) };

	output.error = Base256uMath::divide(
		raw, size,
		&other, sizeof(other),
		output.raw, output.size
	);

	return output;
}

Base256uint Base256uint::operator%(const Base256uint& other) const {
	Base256uint output{ MAX(size, other.size) };

	output.error = Base256uMath::mod(
		raw, size,
		other.raw, other.size,
		output.raw, output.size
	);

	return output;
}

Base256uint Base256uint::operator%(std::size_t other) const {
	Base256uint output{ MAX(size, sizeof(other)) };

	output.error = Base256uMath::mod(
		raw, size,
		&other, sizeof(other),
		output.raw, output.size
	);

	return output;
}

void Base256uint::operator+=(const Base256uint& other) {
	error = Base256uMath::add(raw, size, other.raw, other.size);
}

void Base256uint::operator+=(std::size_t other) {
	error = Base256uMath::add(raw, size, &other, sizeof(other));
}

void Base256uint::operator-=(const Base256uint& other) {
	error = Base256uMath::subtract(raw, size, other.raw, other.size);
}

void Base256uint::operator-=(std::size_t other) {
	error = Base256uMath::subtract(raw, size, &other, sizeof(other));
}

void Base256uint::operator*=(const Base256uint& other) {
	error = Base256uMath::multiply(raw, size, other.raw, other.size);
}

void Base256uint::operator*=(std::size_t other) {
	error = Base256uMath::multiply(raw, size, &other, sizeof(other));
}

void Base256uint::operator/=(const Base256uint& other) {
	error = Base256uMath::divide_no_mod(raw, size, other.raw, other.size);
}

void Base256uint::operator/=(std::size_t other) {
	error = Base256uMath::divide_no_mod(raw, size, &other, sizeof(other));
}

void Base256uint::operator%=(const Base256uint& other) {
	error = Base256uMath::mod(raw, size, other.raw, other.size);
}

void Base256uint::operator%=(std::size_t other) {
	error = Base256uMath::mod(raw, size, &other, sizeof(other));
}

Base256uint Base256uint::log2(const Base256uint& other) {
	Base256uint output { sizeof(std::size_t) + 1 };
	output.error = Base256uMath::log2(other.raw, other.size, output.raw, output.size);
	return output;
}

std::size_t Base256uint::log256(const Base256uint& other) {
	std::size_t output = 0;
	Base256uMath::log256(other.raw, other.size, &output);
	return output;
}

const Base256uint Base256uint::max(
	const Base256uint& left,
	const Base256uint& right
) {
	int cmp = Base256uMath::compare(left.raw, left.size, right.raw, right.size);
	if (cmp < 0)
		return right;
	else
		return left;
}

const Base256uint Base256uint::max(
	const Base256uint& left,
	std::size_t right
) {
	int cmp = Base256uMath::compare(left.raw, left.size, &right, sizeof(right));
	if (cmp < 0)
		return Base256uint(&right, sizeof(right));
	else
		return left;
}

const Base256uint Base256uint::min(
	const Base256uint& left,
	const Base256uint& right
) {
	int cmp = Base256uMath::compare(left.raw, left.size, right.raw, right.size);
	if (cmp > 0)
		return right;
	else
		return left;
}

const Base256uint Base256uint::min(
	const Base256uint& left,
	std::size_t right
) {
	int cmp = Base256uMath::compare(left.raw, left.size, &right, sizeof(right));
	if (cmp > 0)
		return Base256uint(&right, sizeof(right));
	else
		return left;
}
