#include "UnitTests.h"
#include "Base256uMath.h"

#include <cassert>
#include <string>

#if BASE256UMATH_ARCHITECTURE == 64
typedef uint32_t half_size_t;
#elif BASE256UMATH_ARCHITECTURE == 32
typedef uint16_t half_size_t;
#endif

void Base256uMathTests::test_unit_tests() {
	is_zero();
	compare();
	max();
	min();
	bitwise_and();
	bitwise_or();
	bitwise_xor();
	bitwise_not();
	byte_shift_left();
	byte_shift_right();
	bit_shift_left();
	bit_shift_right();
	increment();
	decrement();
	add();
	subtract();
	log2();
	log256();
	multiply();
	divide();
	divide_no_mod();
	mod();
}

Base256uMathTests::compare::compare() {
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
void Base256uMathTests::compare::ideal_case_equal() {
	// Here the two numbers will be equal sizes.
	// And they will both be small.

	std::size_t l = 156,
		r = 156;
	assert(sizeof(l) == sizeof(r));
	int cmp = Base256uMath::compare(
		&l, sizeof(l),
		&r, sizeof(r)
	);
	assert(cmp == 0);
}
void Base256uMathTests::compare::ideal_case_greater() {
	// Here the two numbers will be equal sizes.
	// And they will both be small.

	std::size_t l = 420,
		r = 69;
	assert(sizeof(l) == sizeof(r));
	int cmp = Base256uMath::compare(
		&l, sizeof(l),
		&r, sizeof(r)
	);
	assert(cmp > 0);
}
void Base256uMathTests::compare::ideal_case_less() {
	// Here the two numbers will be equal sizes.
	// And they will both be small.

	std::size_t l = 69,
		r = 420;
	assert(sizeof(l) == sizeof(r));
	int cmp = Base256uMath::compare(
		&l, sizeof(l),
		&r, sizeof(r)
	);
	assert(cmp < 0);
}
void Base256uMathTests::compare::big_ideal_case_equal() {
	// Here the two numbers will be equal sizes.
	// And they will both be big.

	uint8_t l[] = { 183, 79, 180, 87, 57, 45, 214, 45, 189 };
	uint8_t r[] = { 183, 79, 180, 87, 57, 45, 214, 45, 189 };
	assert(sizeof(l) == sizeof(r));
	int cmp = Base256uMath::compare(
		l, sizeof(l),
		r, sizeof(r)
	);
	assert(cmp == 0);
}
void Base256uMathTests::compare::big_ideal_case_greater() {
	// Here the two numbers will be equal sizes.
	// And they will both be big.

	uint8_t l[] = { 186, 153, 248, 144, 124, 225, 100, 21, 186 };
	uint8_t r[] = { 125, 225, 204, 133, 182, 137, 171, 180, 105 };
	assert(sizeof(l) == sizeof(r));
	int cmp = Base256uMath::compare(
		l, sizeof(l),
		r, sizeof(r)
	);
	assert(cmp > 0);
}
void Base256uMathTests::compare::big_ideal_case_less() {
	// Here the two numbers will be equal sizes.
	// And they will both be big.
	
	uint8_t l[] = { 15, 100, 121, 37, 114, 241, 99, 246, 155 };
	uint8_t r[] = { 97, 197, 235, 80, 143, 160, 4, 88, 188 };
	assert(sizeof(l) == sizeof(r));
	int cmp = Base256uMath::compare(
		l, sizeof(l),
		r, sizeof(r)
	);
	assert(cmp < 0);
}
void Base256uMathTests::compare::l_bigger_equal() {
	// l is going to be bigger in size, but they will evaluate to be equal

#if BASE256UMATH_ARCHITECTURE == 64
	std::size_t l = 8008135;
	half_size_t r = 8008135;
#elif BASE256UMATH_ARCHITECTURE == 32
	std::size_t l = 1337;
	half_size_t r = 1337;
#endif
	assert(sizeof(l) > sizeof(r));
	int cmp = Base256uMath::compare(
		&l, sizeof(l),
		&r, sizeof(r)
	);
	assert(cmp == 0);
}
void Base256uMathTests::compare::l_smaller_equal() {
	// l is going to be smaller in size, but they will evaluate to be equal

#if BASE256UMATH_ARCHITECTURE == 64
	half_size_t l = 8008135;
	std::size_t r = 8008135;
#elif BASE256UMATH_ARCHITECTURE == 32
	half_size_t l = 1337;
	std::size_t r = 1337;
#endif
	assert(sizeof(l) < sizeof(r));
	int cmp = Base256uMath::compare(
		&l, sizeof(l),
		&r, sizeof(r)
	);
	assert(cmp == 0);
}
void Base256uMathTests::compare::big_l_bigger_equal() {
	uint8_t l[] = { 202, 14, 146, 155, 72, 7, 240, 198, 40, 0, 0 };
	uint8_t r[] = { 202, 14, 146, 155, 72, 7, 240, 198, 40 };
	assert(sizeof(l) > sizeof(r));
	int cmp = Base256uMath::compare(
		l, sizeof(l),
		r, sizeof(r)
	);
	assert(cmp == 0);
}
void Base256uMathTests::compare::big_l_smaller_equal() {
	uint8_t l[] = { 252, 95, 221, 19, 91, 22, 144, 72, 99 };
	uint8_t r[] = { 252, 95, 221, 19, 91, 22, 144, 72, 99, 0, 0 };
	assert(sizeof(l) < sizeof(r));
	int cmp = Base256uMath::compare(
		l, sizeof(l),
		r, sizeof(r)
	);
	assert(cmp == 0);
}
void Base256uMathTests::compare::l_bigger_greater() {
	// l is going to be bigger in size, but l will be greater

	std::size_t l = 144;
#if BASE256UMATH_ARCHITECTURE == 64
	half_size_t r = 25;
#elif BASE256UMATH_ARCHITECTURE == 32
	half_size_t r = 25;
#endif
	assert(sizeof(l) > sizeof(r));
	int cmp = Base256uMath::compare(
		&l, sizeof(l),
		&r, sizeof(r)
	);
	assert(cmp > 0);
}
void Base256uMathTests::compare::l_smaller_greater() {
	// l is going to be smaller in size, but l will be greater

#if BASE256UMATH_ARCHITECTURE == 64
	half_size_t l = 1026;
#elif BASE256UMATH_ARCHITECTURE == 32
	half_size_t l = 1026;
#endif
	std::size_t r = 55;
	assert(sizeof(l) < sizeof(r));
	int cmp = Base256uMath::compare(
		&l, sizeof(l),
		&r, sizeof(r)
	);
	assert(cmp > 0);
}
void Base256uMathTests::compare::big_l_bigger_greater() {
	uint8_t l[] = { 147, 199, 111, 216, 79, 139, 236, 53, 116, 0, 0 };
	uint8_t r[] = { 142, 99, 1, 230, 35, 170, 69, 133, 22 };

	assert(sizeof(l) > sizeof(r));
	int cmp = Base256uMath::compare(
		l, sizeof(l),
		r, sizeof(r)
	);
	assert(cmp > 0);
}
void Base256uMathTests::compare::big_l_smaller_greater() {
	uint8_t l[] = { 245, 206, 105, 71, 234, 204, 105, 6, 220 };
	uint8_t r[] = { 172, 253, 57, 29, 149, 255, 208, 108, 3, 0, 0 };

	assert(sizeof(l) < sizeof(r));
	int cmp = Base256uMath::compare(
		l, sizeof(l),
		r, sizeof(r)
	);
	assert(cmp > 0);
}
void Base256uMathTests::compare::l_bigger_less() {
	// l is going to be bigger in size, but l will be less

	std::size_t l = 55;
#if BASE256UMATH_ARCHITECTURE == 64
	half_size_t r = 98;
#elif BASE256UMATH_ARCHITECTURE == 32
	half_size_t r = 98;
#endif

	assert(sizeof(l) > sizeof(r));
	int cmp = Base256uMath::compare(
		&l, sizeof(l),
		&r, sizeof(r)
	);
	assert(cmp < 0);
}
void Base256uMathTests::compare::l_smaller_less() {
	// l is going to be smaller in size, but l will be less

	half_size_t l = 18;
	std::size_t r = 2173;

	assert(sizeof(l) < sizeof(r));
	int cmp = Base256uMath::compare(
		&l, sizeof(l),
		&r, sizeof(r)
	);
	assert(cmp < 0);
}
void Base256uMathTests::compare::big_l_bigger_less() {
	uint8_t l[] = { 170, 30, 170, 121, 65, 171, 74, 245, 197, 0, 0 };
	uint8_t r[] = { 172, 253, 57, 29, 149, 255, 208, 108, 220 };

	assert(sizeof(l) > sizeof(r));
	int cmp = Base256uMath::compare(
		l, sizeof(l),
		r, sizeof(r)
	);
	assert(cmp < 0);
}
void Base256uMathTests::compare::big_l_smaller_less() {
	uint8_t l[] = { 8, 14, 171, 56, 247, 85, 145, 105, 219 };
	uint8_t r[] = { 35, 47, 187, 90, 199, 73, 141, 94, 241, 0, 0 };

	assert(sizeof(l) < sizeof(r));
	int cmp = Base256uMath::compare(
		l, sizeof(l),
		r, sizeof(r)
	);
	assert(cmp < 0);
}
void Base256uMathTests::compare::left_n_zero() {
	// if left_n is zero, then left is assumed to be 0.

	std::size_t left = 5839010;
	std::size_t right = 199931;
	assert(left > right);
	int cmp = Base256uMath::compare(&left, 0, &right, sizeof(right));
	assert(cmp < 0);
	cmp = Base256uMath::compare(&right, 0, &left, sizeof(left));
	assert(cmp < 0);
	cmp = Base256uMath::compare(&left, 0, &left, sizeof(left));
	assert(cmp < 0);
	right = 0;
	cmp = Base256uMath::compare(&left, 0, &right, sizeof(right));
	assert(cmp == 0);
}
void Base256uMathTests::compare::right_n_zero() {
	// if right_n is zero, then left is assumed to be 0.
	
	std::size_t left = 199931;
	std::size_t right = 5839010;
	assert(left < right);
	int cmp = Base256uMath::compare(&left, sizeof(left), &right, 0);
	assert(cmp > 0);
	cmp = Base256uMath::compare(&right, sizeof(right), &left, 0);
	assert(cmp > 0);
	cmp = Base256uMath::compare(&left, sizeof(left), &left, 0);
	assert(cmp > 0);
	left = 0;
	cmp = Base256uMath::compare(&left, sizeof(left), &right, 0);
	assert(cmp == 0);
}

Base256uMathTests::is_zero::is_zero() {
	ideal_case();
	big_ideal_case();
	not_zero();
	big_not_zero();
	src_n_zero();
}
void Base256uMathTests::is_zero::ideal_case() {
	std::size_t num = 0;
	assert(Base256uMath::is_zero(&num, sizeof(num)));
}
void Base256uMathTests::is_zero::big_ideal_case() {
	uint8_t num[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	assert(Base256uMath::is_zero(num, sizeof(num)));
}
void Base256uMathTests::is_zero::not_zero() {
	std::size_t num = 1;
	num <<= (sizeof(std::size_t) << 3) - 1;
	assert(!Base256uMath::is_zero(&num, sizeof(num)));
}
void Base256uMathTests::is_zero::big_not_zero() {
	uint8_t num[] = { 0, 0, 0, 0, 0, 0, 0, 0, 1 };
	assert(!Base256uMath::is_zero(num, sizeof(num)));
}
void Base256uMathTests::is_zero::src_n_zero() {
	// if src_n is zero, then it is assumed to be zero.
	// In other words, it should always return true.

	half_size_t num = 1990671886;
	assert(Base256uMath::is_zero(&num, 0));
}

Base256uMathTests::max::max() {
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

	left_n_zero();
	right_n_zero();
}
void Base256uMathTests::max::ideal_case_left() {
	std::size_t left = 500,
		right = 320;
	assert(sizeof(left) == sizeof(right));
	auto ptr = reinterpret_cast<const std::size_t*>(
		Base256uMath::max(&left, sizeof(left), &right, sizeof(right))
	);
	assert(ptr == &left);
}
void Base256uMathTests::max::ideal_case_right() {
	std::size_t left = 13,
		right = 1337;
	assert(sizeof(left) == sizeof(right));
	auto ptr = reinterpret_cast<const std::size_t*>(
		Base256uMath::max(&left, sizeof(left), &right, sizeof(right))
	);
	assert(ptr == &right);
}
void Base256uMathTests::max::big_ideal_case_left() {
	const uint8_t left[] = { 156, 247, 183, 55, 60, 119, 65, 37, 175 };
	const uint8_t right[] = { 239, 55, 236, 133, 175, 168, 253, 237, 57 };

	assert(sizeof(left) == sizeof(right));
	auto ptr = reinterpret_cast<const uint8_t*>(
		Base256uMath::max(left, sizeof(left), right, sizeof(right))
	);
	assert(ptr == left);
}
void Base256uMathTests::max::big_ideal_case_right() {
	const uint8_t left[] = { 220, 165, 118, 130, 251, 82, 50, 81, 178 };
	const uint8_t right[] = { 177, 246, 145, 224, 167, 216, 180, 173, 186 };

	assert(sizeof(left) == sizeof(right));
	auto ptr = reinterpret_cast<const uint8_t*>(
		Base256uMath::max(left, sizeof(left), right, sizeof(right))
	);
	assert(ptr == right);
}
void Base256uMathTests::max::left_bigger_left() {
	std::size_t left = 696969696;
	half_size_t right = 21360;
	assert(sizeof(left) > sizeof(right));
	auto ptr = Base256uMath::max(&left, sizeof(left), &right, sizeof(right));
	assert(ptr == &left);
}
void Base256uMathTests::max::left_smaller_left() {
	half_size_t left = 35459;
	std::size_t right = 3819;
	assert(sizeof(left) < sizeof(right));
	auto ptr = Base256uMath::max(&left, sizeof(left), &right, sizeof(right));
	assert(ptr == &left);
}
void Base256uMathTests::max::left_bigger_right() {
	std::size_t left = 13264;
	half_size_t right = 19894;
	assert(sizeof(left) > sizeof(right));
	auto ptr = Base256uMath::max(&left, sizeof(left), &right, sizeof(right));
	assert(ptr == &right);
}
void Base256uMathTests::max::left_smaller_right() {
	half_size_t left = 4548;
	std::size_t right = 30923;
	assert(sizeof(left) < sizeof(right));
	auto ptr = Base256uMath::max(&left, sizeof(left), &right, sizeof(right));
	assert(ptr == &right);
}
void Base256uMathTests::max::big_left_bigger_left() {
	uint8_t left[] = { 65, 128, 36, 71, 126, 195, 52, 194, 176, 0, 0 };
	uint8_t right[] = { 108, 128, 45, 116, 237, 77, 15, 158, 89 };
	assert(sizeof(left) > sizeof(right));
	auto ptr = Base256uMath::max(left, sizeof(left), right, sizeof(right));
	assert(ptr == left);
}
void Base256uMathTests::max::big_left_smaller_left() {
	uint8_t left[] = { 180, 67, 35, 216, 106, 3, 28, 187, 155 };
	uint8_t right[] = { 149, 169, 152, 146, 14, 240, 4, 241, 95, 0, 0 };
	assert(sizeof(left) < sizeof(right));
	auto ptr = Base256uMath::max(left, sizeof(left), right, sizeof(right));
	assert(ptr == left);
}
void Base256uMathTests::max::big_left_bigger_right() {
	uint8_t left[] = { 216, 116, 109, 138, 103, 52, 127, 58, 65, 0, 0 };
	uint8_t right[] = { 119, 78, 117, 53, 63, 130, 146, 168, 219 };
	assert(sizeof(left) > sizeof(right));
	auto ptr = Base256uMath::max(left, sizeof(left), right, sizeof(right));
	assert(ptr == right);
}
void Base256uMathTests::max::big_left_smaller_right() {
	uint8_t left[] = { 164, 254, 202, 93, 102, 155, 170, 243, 234 };
	uint8_t right[] = { 163, 24, 36, 50, 205, 211, 146, 12, 238, 0, 0 };
	assert(sizeof(left) < sizeof(right));
	auto ptr = Base256uMath::max(left, sizeof(left), right, sizeof(right));
	assert(ptr == right);
}
void Base256uMathTests::max::left_n_zero() {
	std::size_t left = 1337;
	std::size_t right = 69;
	auto ptr = Base256uMath::max(&left, 0, &right, sizeof(right));
	assert(ptr == &right);
}
void Base256uMathTests::max::right_n_zero() {
	std::size_t left = 1337;
	std::size_t right = 69;
	auto ptr = Base256uMath::max(&left, sizeof(left), &right, 0);
	assert(ptr == &left);
}

Base256uMathTests::min::min() {
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

	left_n_zero();
	right_n_zero();
}
void Base256uMathTests::min::ideal_case_left() {
	std::size_t left = 8969;
	std::size_t right = 11219;
	assert(sizeof(left) == sizeof(right));
	auto ptr = Base256uMath::min(&left, sizeof(left), &right, sizeof(right));
	assert(ptr == &left);
}
void Base256uMathTests::min::ideal_case_right() {
	std::size_t left = 34063;
	std::size_t right = 16197;
	assert(sizeof(left) == sizeof(right));
	auto ptr = Base256uMath::min(&left, sizeof(left), &right, sizeof(right));
	assert(ptr == &right);
}
void Base256uMathTests::min::big_ideal_case_left() {
	uint8_t left[] = { 29, 236, 239, 48, 243, 6, 109, 228, 82 };
	uint8_t right[] = { 153, 65, 158, 142, 123, 85, 44, 225, 162 };
	assert(sizeof(left) == sizeof(right));
	auto ptr = Base256uMath::min(left, sizeof(left), right, sizeof(right));
	assert(ptr == left);
}
void Base256uMathTests::min::big_ideal_case_right() {
	uint8_t left[] = { 83, 167, 5, 136, 162, 1, 249, 140, 156 };
	uint8_t right[] = { 102, 251, 89, 166, 213, 231, 56, 54, 20 };
	assert(sizeof(left) == sizeof(right));
	auto ptr = Base256uMath::min(left, sizeof(left), right, sizeof(right));
	assert(ptr == right);
}
void Base256uMathTests::min::left_bigger_left() {
	std::size_t left = 28606;
	half_size_t right = 34288;
	assert(sizeof(left) > sizeof(right));
	auto ptr = Base256uMath::min(&left, sizeof(left), &right, sizeof(right));
	assert(ptr == &left);
}
void Base256uMathTests::min::left_smaller_left() {
	half_size_t left = 43810;
	std::size_t right = 47275;
	assert(sizeof(left) < sizeof(right));
	auto ptr = Base256uMath::min(&left, sizeof(left), &right, sizeof(right));
	assert(ptr == &left);
}
void Base256uMathTests::min::left_bigger_right() {
	std::size_t left = 49660;
	half_size_t right = 7010;
	assert(sizeof(left) > sizeof(right));
	auto ptr = Base256uMath::min(&left, sizeof(left), &right, sizeof(right));
	assert(ptr == &right);
}
void Base256uMathTests::min::left_smaller_right() {
	half_size_t left = 63729;
	std::size_t right = 46223;
	assert(sizeof(left) < sizeof(right));
	auto ptr = Base256uMath::min(&left, sizeof(left), &right, sizeof(right));
	assert(ptr == &right);
}
void Base256uMathTests::min::big_left_bigger_left() {
	uint8_t left[] = { 123, 68, 215, 46, 186, 97, 149, 27, 149, 0, 0 };
	uint8_t right[] = { 120, 114, 238, 213, 227, 7, 228, 47, 159 };
	assert(sizeof(left) > sizeof(right));
	auto ptr = Base256uMath::min(left, sizeof(left), right, sizeof(right));
	assert(ptr == left);
}
void Base256uMathTests::min::big_left_smaller_left() {
	uint8_t left[] = { 253, 37, 145, 49, 69, 19, 171, 189, 27 };
	uint8_t right[] = { 67, 228, 217, 39, 59, 24, 249, 194, 55, 0, 0};
	assert(sizeof(left) < sizeof(right));
	auto ptr = Base256uMath::min(left, sizeof(left), right, sizeof(right));
	assert(ptr == left);
}
void Base256uMathTests::min::big_left_bigger_right() {
	uint8_t left[] = { 49, 27, 111, 206, 109, 89, 42, 220, 227, 0, 0 };
	uint8_t right[] = { 93, 22, 212, 80, 84, 184, 37, 130, 194 };
	assert(sizeof(left) > sizeof(right));
	auto ptr = Base256uMath::min(left, sizeof(left), right, sizeof(right));
	assert(ptr == right);
}
void Base256uMathTests::min::big_left_smaller_right() {
	uint8_t left[] = { 87, 220, 65, 201, 73, 117, 94, 29, 173 };
	uint8_t right[] = { 91, 247, 82, 39, 62, 19, 90, 174, 118, 0, 0 };
	assert(sizeof(left) < sizeof(right));
	auto ptr = Base256uMath::min(left, sizeof(left), right, sizeof(right));
	assert(ptr == right);
}
void Base256uMathTests::min::left_n_zero() {
	unsigned int left = 1337;
	unsigned int right = 69;
	auto ptr = Base256uMath::min(&left, 0, &right, sizeof(right));
	assert(ptr == &left);
}
void Base256uMathTests::min::right_n_zero() {
	unsigned int left = 69;
	unsigned int right = 1337;
	auto ptr = Base256uMath::min(&left, sizeof(left), &right, 0);
	assert(ptr == &right);
}

Base256uMathTests::increment::increment() {
	ideal_case();
	big_ideal_case();
	overflow();
	big_overflow();
	block_n_zero();
}
void Base256uMathTests::increment::ideal_case() {
	unsigned int num = 14703;
	auto code = Base256uMath::increment(&num, sizeof(num));
	assert(num == 14704);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::increment::big_ideal_case() {
	uint8_t num[] = { 72, 202, 187, 220, 23, 141, 160, 38, 41 };
	auto code = Base256uMath::increment(num, sizeof(num));
	uint8_t answer[] = { 73, 202, 187, 220, 23, 141, 160, 38, 41 };
	for (uint8_t i = 0; i < sizeof(num); i++) {
		assert(num[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::increment::overflow() {
	std::size_t num = -1;
	auto code = Base256uMath::increment(&num, sizeof(num));
	assert(num == 0);
	assert(code == Base256uMath::ErrorCodes::FLOW);
}
void Base256uMathTests::increment::big_overflow() {
	uint8_t num[] = { 255, 255, 255, 255, 255, 255, 255, 255, 255 };
	auto code = Base256uMath::increment(num, sizeof(num));
	for (uint8_t i = 0; i < sizeof(num); i++) {
		assert(num[i] == 0);
	}
	assert(code == Base256uMath::ErrorCodes::FLOW);
}
void Base256uMathTests::increment::block_n_zero() {
	unsigned int num = 12345;
	auto code = Base256uMath::increment(&num, 0);
	assert(num == 12345);
	assert(code == Base256uMath::ErrorCodes::OK);
}

Base256uMathTests::decrement::decrement() {
	ideal_case();
	big_ideal_case();
	underflow();
	big_underflow();
	block_n_zero();
}
void Base256uMathTests::decrement::ideal_case() {
	unsigned int num = 47157;
	auto code = Base256uMath::decrement(&num, sizeof(num));
	assert(num == 47156);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::decrement::big_ideal_case() {
	uint8_t num[] = { 82, 130, 64, 83, 78, 107, 211, 34, 158 };
	auto code = Base256uMath::decrement(num, sizeof(num));
	uint8_t answer[] = { 81, 130, 64, 83, 78, 107, 211, 34, 158 };
	for (uint8_t i = 0; i < sizeof(num); i++) {
		assert(num[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::decrement::underflow() {
	std::size_t num = 0;
	auto code = Base256uMath::decrement(&num, sizeof(num));
	assert(num == (std::size_t)-1);
	assert(code == Base256uMath::ErrorCodes::FLOW);
}
void Base256uMathTests::decrement::big_underflow() {
	uint8_t num[] = { 0,0,0,0,0,0,0,0,0 };
	auto code = Base256uMath::decrement(num, sizeof(num));
	for (uint8_t i = 0; i < sizeof(num); i++) {
		assert(num[i] == 255);
	}
	assert(code == Base256uMath::ErrorCodes::FLOW);
}
void Base256uMathTests::decrement::block_n_zero() {
	unsigned int num = 12345;
	auto code = Base256uMath::decrement(&num, 0);
	assert(num == 12345);
	assert(code == Base256uMath::ErrorCodes::OK);
}

Base256uMathTests::add::add() {
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
void Base256uMathTests::add::ideal_case() {
	std::size_t left = 383991685,
		right = 2054577074,
		dst, answer = left + right;
	assert(sizeof(left) == sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::add::big_ideal_case() {
	uint8_t left[] = { 133, 141, 239, 45, 85, 113, 36, 5, 18 };
	uint8_t right[] = { 71, 61, 127, 205, 77, 38, 168, 183, 100 };
	uint8_t dst[] = { 0,0,0,0,0,0,0,0,0 };
	assert(sizeof(left) == sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::add(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 204, 202, 110, 251, 162, 151, 204, 188, 118 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::add::left_bigger() {
	std::size_t left = 8388997231;
	half_size_t right = 62557;
	std::size_t dst = 0;
	assert(sizeof(left) > sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	decltype(dst) answer = left + right;
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::add::left_smaller() {
	half_size_t left = 62557;
	std::size_t right = 8388997231;
	std::size_t dst = 0;
	assert(sizeof(left) < sizeof(right) && sizeof(right) == sizeof(dst));
	auto code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	decltype(dst) answer = left + right;
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::add::big_left_bigger() {
	uint8_t left[] = { 89, 41, 94, 204, 226, 89, 158, 240, 172, 184, 0, 248 };
	uint8_t right[] = { 175, 209, 133, 96, 128, 118, 74, 9, 212 };
	uint8_t dst[] = { 0,0,0,0,0,0,0,0,0,0,0,0 };
	assert(sizeof(left) > sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::add(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 8, 251, 227, 44, 99, 208, 232, 249, 128, 185, 0, 248 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::add::big_left_smaller() {
	uint8_t left[] = { 156, 140, 94, 99, 248, 185, 215, 241, 43 };
	uint8_t right[] = { 226, 149, 147, 57, 68, 129, 92, 115, 20, 129, 106, 73 };
	uint8_t dst[] = { 0,0,0,0,0,0,0,0,0,0,0,0 };
	assert(sizeof(left) < sizeof(right) && sizeof(right) == sizeof(dst));
	auto code = Base256uMath::add(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 126, 34, 242, 156, 60, 59, 52, 101, 64, 129, 106, 73 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::add::overflow() {
	std::size_t left = -1,
		right = 1,
		dst;
	auto code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == 0);
	assert(code == Base256uMath::ErrorCodes::FLOW);
}
void Base256uMathTests::add::big_overflow() {
	uint8_t left[] = { 255, 255, 255, 255, 255, 255, 255, 255, 255 };
	uint8_t right[] = { 1, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t dst[] = { 52, 81, 217, 207, 245, 155, 109, 25, 252 };
	auto code = Base256uMath::add(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == 0);
	}
	assert(code == Base256uMath::ErrorCodes::FLOW);	
}
void Base256uMathTests::add::dst_too_small() {
	// when dst is too small, it should truncate the answer and return the appropriate code.

	std::size_t left = 8024591321371708722,
		right = 64081;
	half_size_t dst = 0,
		answer = left + right;
	assert(sizeof(left) == sizeof(right) && sizeof(left) > sizeof(dst));
	auto code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::add::big_dst_too_small() {
	// when dst is too small, it should truncate the answer and return the appropriate code.

	uint8_t left[] = { 81, 161, 205, 28, 5, 231, 145, 223, 39, 28, 13, 92 };
	uint8_t right[] = { 250, 17, 104, 13, 192, 89, 177, 235, 10, 100 };
	uint8_t dst[] = { 0,0,0,0,0,0,0,0,0 };
	assert(sizeof(left) > sizeof(right) && sizeof(left) > sizeof(dst));
	auto code = Base256uMath::add(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 75, 179, 53, 42, 197, 64, 67, 203, 50, 128, 13, 92 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::add::zero_for_left_n() {
	// If left_n is 0, then it's the equivalent of adding 0 to right.

	half_size_t left = 1337;
	std::size_t right = -1;
	std::size_t dst = 0;
	auto code = Base256uMath::add(&left, 0, &right, sizeof(right), &dst, sizeof(dst));
	assert(right == dst);
	assert(code == Base256uMath::ErrorCodes::OK);

}
void Base256uMathTests::add::zero_for_right_n() {
	// If right_n is 0, then it's the equivalent of adding 0 to left.

	uint8_t left[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	unsigned short right = 1;
	uint8_t dst[9];
	auto code = Base256uMath::add(left, sizeof(left), &right, 0, dst, sizeof(dst));
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == left[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::add::zero_for_dst_n() {
	// If dst_n is 0, then adding won't do anything but return a truncated error code.

	uint8_t left[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	uint8_t right[] = { 19, 20, 21, 22, 23, 24, 25, 26, 27 };
	uint8_t dst[] = { 28, 29, 30 };
	auto code = Base256uMath::add(left, sizeof(left), right, sizeof(right), dst, 0);
	assert(dst[0] == 28 && dst[1] == 29 && dst[2] == 30);
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::add::in_place_ideal_case() {
	std::size_t left = 21048,
		right = 13196,
		answer = left + right;
	assert(sizeof(left) == sizeof(right));
	auto code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::add::in_place_big_ideal_case() {
	uint8_t left[] = { 133, 141, 239, 45, 85, 113, 36, 5, 18 };
	uint8_t right[] = { 71, 61, 127, 205, 77, 38, 168, 183, 100 };
	assert(sizeof(left) == sizeof(right));
	auto code = Base256uMath::add(left, sizeof(left), right, sizeof(right));
	uint8_t answer[] = { 204, 202, 110, 251, 162, 151, 204, 188, 118 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::add::in_place_left_bigger() {
	std::size_t left = 8388997231;
	half_size_t right = 62557;
	std::size_t answer = left + right;
	assert(sizeof(left) > sizeof(right));
	auto code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::add::in_place_left_smaller() {
	half_size_t left = 62557;
	std::size_t right = 8388936303;
	half_size_t answer = left + right;
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::add::in_place_big_left_bigger() {
	uint8_t left[] = { 89, 41, 94, 204, 226, 89, 158, 240, 172, 184, 0, 248 };
	uint8_t right[] = { 175, 209, 133, 96, 128, 118, 74, 9, 212 };
	assert(sizeof(left) > sizeof(right));
	auto code = Base256uMath::add(left, sizeof(left), right, sizeof(right));
	uint8_t answer[] = { 8, 251, 227, 44, 99, 208, 232, 249, 128, 185, 0, 248 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::add::in_place_big_left_smaller() {
	uint8_t left[] = { 156, 140, 94, 99, 248, 185, 215, 241, 43 };
	uint8_t right[] = { 226, 149, 147, 57, 68, 129, 92, 115, 20, 129, 106, 73 };
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::add(left, sizeof(left), right, sizeof(right));
	uint8_t answer[] = { 126, 34, 242, 156, 60, 59, 52, 101, 64, 129, 106, 73 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::add::in_place_overflow() {
	std::size_t left = -1,
		right = 1;
	auto code = Base256uMath::add(&left, sizeof(left), &right, sizeof(right));
	assert(left == 0);
	assert(code == Base256uMath::ErrorCodes::FLOW);
}
void Base256uMathTests::add::in_place_big_overflow() {
	uint8_t left[] = { 255, 255, 255, 255, 255, 255, 255, 255, 255 };
	uint8_t right[] = { 1, 0, 0, 0, 0, 0, 0, 0, 0 };
	auto code = Base256uMath::add(left, sizeof(left), right, sizeof(right));
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == 0);
	}
	assert(code == Base256uMath::ErrorCodes::FLOW);
}
void Base256uMathTests::add::in_place_zero_for_left_n() {
	// If left_n is 0, then adding won't do anything but return a truncated error code.

	uint8_t left[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	uint8_t right[] = { 19, 20, 21, 22, 23, 24, 25, 26, 27 };
	auto code = Base256uMath::add(left, 0, right, sizeof(right));
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == 10 + i);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::add::in_place_zero_for_right_n() {
	// If right_n is 0, then it's the equivalent of adding 0 to left.

	uint8_t left[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	unsigned short right = 1;
	auto code = Base256uMath::add(left, sizeof(left), &right, 0);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == i + 1);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}

Base256uMathTests::subtract::subtract() {
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
void Base256uMathTests::subtract::ideal_case() {
	std::size_t left = 47462,
		right = 36840,
		dst,
		answer = left - right;
	assert(sizeof(left) == sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::subtract::big_ideal_case() {
	uint8_t left[] = { 73, 120, 227, 232, 214, 48, 11, 250, 184 };
	uint8_t right[] = { 66, 115, 195, 196, 65, 141, 141, 8, 46 };
	uint8_t dst[9];
	assert(sizeof(left) == sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 7, 5, 32, 36, 149, 163, 125, 241, 138 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::subtract::left_bigger() {
	std::size_t left = 9886306996502392208;
	half_size_t right = 2536;
	std::size_t dst,
		answer = left - right;
	assert(sizeof(left) > sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::subtract::left_smaller() {
	half_size_t left = 27639;
	std::size_t right = 15223;
	std::size_t dst;
	assert(sizeof(left) < sizeof(right) && sizeof(right) == sizeof(dst));
	auto code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == 12416);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::subtract::big_left_bigger() {
	uint8_t left[] = { 144, 165, 47, 40, 109, 135, 246, 58, 243, 129, 123, 49 };
	uint8_t right[] = { 99, 52, 93, 254, 211, 44, 168, 77, 192 };
	uint8_t dst[12];
	assert(sizeof(left) > sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 45, 113, 210, 41, 153, 90, 78, 237, 50, 129, 123, 49 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::subtract::big_left_smaller() {
	uint8_t left[] = { 55, 110, 240, 162, 231, 119, 65, 145, 251 };
	uint8_t right[] = { 99, 148, 120, 205, 172, 215, 125, 26, 14, 0, 0, 0 };
	uint8_t dst[12];
	assert(sizeof(left) < sizeof(right) && sizeof(right) == sizeof(dst));
	auto code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 212, 217, 119, 213, 58, 160, 195, 118, 237, 0, 0, 0 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::subtract::underflow() {
	std::size_t left = 0,
		right = 60331,
		dst,
		answer = left - right;
	auto code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::FLOW);
}
void Base256uMathTests::subtract::big_underflow() {
	uint8_t left[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t right[] = { 111, 198, 135, 255, 25, 61, 175, 193, 75 };
	uint8_t dst[9];
	auto code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 145, 57, 120, 0, 230, 194, 80, 62, 180 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::FLOW);
}
void Base256uMathTests::subtract::dst_too_small() {
	std::size_t left = 3971310598,
		right = 3473639;
	half_size_t dst,
		answer = left - right;
	assert(sizeof(left) == sizeof(right) && sizeof(left) > sizeof(dst));
	auto code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::subtract::big_dst_too_small() {
	uint8_t left[] = { 220, 139, 205, 222, 88, 100, 192, 105, 59, 106, 0, 179 };
	uint8_t right[] = { 206, 72, 107, 123, 155, 149, 24, 175 };
	uint8_t dst[9];
	auto code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 14, 67, 98, 99, 189, 206, 167, 186, 58, 106, 0, 179 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(answer[i] == dst[i]);
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::subtract::zero_for_left_n() {
	// if left_n is zero, then it is the equivalent of doing 2's complement
	// and return a flow error

	std::size_t left = 1212121212121;
	unsigned short right = 29332;
	unsigned short dst = 0;
	auto code = Base256uMath::subtract(&left, 0, &right, sizeof(right), &dst, sizeof(dst));
	unsigned short answer = ~right + 1;
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::FLOW);
}
void Base256uMathTests::subtract::zero_for_right_n() {
	// if right_n is zero, then it is the equivalent of copying the left into dst

	std::size_t left = 85985313099138956,
		right = 1929328482,
		dst,
		answer = left;
	auto code = Base256uMath::subtract(&left, sizeof(left), &right, 0, &dst, sizeof(dst));
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::subtract::zero_for_dst_n() {
	// if dst_n is zero, then the function does nothing and returns the truncated code

	std::size_t left = 1937650443;
	half_size_t right = 5232;
	std::size_t dst = 69;
	auto code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right), &dst, 0);
	assert(dst == 69);
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::subtract::in_place_ideal_case() {
	std::size_t left = 47462,
		right = 36840,
		answer = left - right;
	assert(sizeof(left) == sizeof(right));
	auto code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::subtract::in_place_big_ideal_case() {
	uint8_t left[] = { 73, 120, 227, 232, 214, 48, 11, 250, 184 };
	uint8_t right[] = { 66, 115, 195, 196, 65, 141, 141, 8, 46 };
	uint8_t dst[9];
	assert(sizeof(left) == sizeof(right));
	auto code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right));
	uint8_t answer[] = { 7, 5, 32, 36, 149, 163, 125, 241, 138 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::subtract::in_place_left_bigger() {
	std::size_t left = 9886306996502392208;
	half_size_t right = 2536;
	std::size_t answer = left - right;
	assert(sizeof(left) > sizeof(right));
	auto code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::subtract::in_place_left_smaller() {
	half_size_t left = 27639;
	std::size_t right = 15223;
	decltype(left) answer = left - right;
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::subtract::in_place_big_left_bigger() {
	uint8_t left[] = { 144, 165, 47, 40, 109, 135, 246, 58, 243, 129, 123, 49 };
	uint8_t right[] = { 99, 52, 93, 254, 211, 44, 168, 77, 192 };
	assert(sizeof(left) > sizeof(right));
	auto code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right));
	uint8_t answer[] = { 45, 113, 210, 41, 153, 90, 78, 237, 50, 129, 123, 49 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::subtract::in_place_big_left_smaller() {
	uint8_t left[] = { 55, 110, 240, 162, 231, 119, 65, 145, 251 };
	uint8_t right[] = { 99, 148, 120, 205, 172, 215, 125, 26, 14, 0, 0, 0 };
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right));
	uint8_t answer[] = { 212, 217, 119, 213, 58, 160, 195, 118, 237, 0, 0, 0 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::subtract::in_place_underflow() {
	std::size_t left = 0,
		right = 60331,
		answer = left - right;
	auto code = Base256uMath::subtract(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::FLOW);
}
void Base256uMathTests::subtract::in_place_big_underflow() {
	uint8_t left[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t right[] = { 111, 198, 135, 255, 25, 61, 175, 193, 75 };
	auto code = Base256uMath::subtract(left, sizeof(left), right, sizeof(right));
	uint8_t answer[] = { 145, 57, 120, 0, 230, 194, 80, 62, 180 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::FLOW);
}
void Base256uMathTests::subtract::in_place_zero_for_left_n() {
	// if left_n is zero, then the function does nothing

	std::size_t left = 1212121212121;
	half_size_t right = 29332;
	decltype(left) answer = 1212121212121;
	auto code = Base256uMath::subtract(&left, 0, &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::subtract::in_place_zero_for_right_n() {
	// if right_n is zero, then it is the equivalent of subtracting by 0.

	std::size_t left = 85985313099138956;
	half_size_t right = 1929328482;
	decltype(left) answer = left;
	auto code = Base256uMath::subtract(&left, sizeof(left), &right, 0);
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}

Base256uMathTests::multiply::multiply() {
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
void Base256uMathTests::multiply::ideal_case() {
	half_size_t left = 25603,
		right = 6416;
	std::size_t dst,
		answer = left * right;;
	assert(sizeof(left) == sizeof(right) && sizeof(left) < sizeof(dst));
	auto code = Base256uMath::multiply(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::multiply::big_ideal_case() {
	uint8_t left[] = { 138, 204, 16, 163, 74, 68, 68, 39, 2 };
	uint8_t right[] = { 72, 80, 180, 160, 32, 125, 248, 160, 102 };
	uint8_t dst[18];
	assert(sizeof(left) == sizeof(right) && sizeof(left) < sizeof(dst));
	auto code = Base256uMath::multiply(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 208, 166, 172, 45, 217, 162, 152, 6, 155, 229, 217, 94, 0, 248, 212, 255, 220, 0 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::multiply::multiply_zero() {
	uint8_t left[] = { 253, 112, 1, 250, 242, 174, 77, 35, 242 };
	uint8_t right[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t dst[9];
	auto code = Base256uMath::multiply(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == 0);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::multiply::multiply_one() {
	uint8_t left[] = { 1, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t right[] = { 14, 92, 130, 38, 174, 216, 149, 5, 169 };
	uint8_t dst[9];
	auto code = Base256uMath::multiply(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == right[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::multiply::left_n_greater_than_right_n() {
	uint8_t left[] = { 217, 86, 117, 188, 220, 28, 236, 105, 119 };
	std::size_t right = 707070;
	uint8_t dst[sizeof(left) + sizeof(right)];
	assert(sizeof(left) > sizeof(right));
	auto code = Base256uMath::multiply(left, sizeof(left), &right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 78, 140, 22, 130, 144, 79, 141, 155, 222, 91, 8, 5, 0 };
	for (uint8_t i = 0; i < sizeof(answer); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::multiply::left_n_less_than_right_n() {
	std::size_t left = 707070;
	uint8_t right[] = { 217, 86, 117, 188, 220, 28, 236, 105, 119 };
	uint8_t dst[sizeof(left) + sizeof(right)];
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::multiply(&left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 78, 140, 22, 130, 144, 79, 141, 155, 222, 91, 8, 5, 0 };
	for (uint8_t i = 0; i < sizeof(answer); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::multiply::dst_n_less_than_both() {
	// if dst_n < left_n <=> right_n, then the answer will be truncated, and 
	// return the truncated error code

	uint8_t left[] = { 217, 86, 117, 188, 220, 28, 236, 105, 119 };
	std::size_t right = 707070;
	uint8_t dst[sizeof(right) - 1];
	assert(sizeof(left) > sizeof(right));
	auto code = Base256uMath::multiply(left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	uint8_t answer[] = { 78, 140, 22, 130, 144, 79, 141, 155, 222, 91, 8, 5 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::multiply::dst_n_zero() {
	// if dst_n is zero, then nothing will happen and the truncated error code will be returned

	uint8_t left[] = { 101, 165, 53, 155, 99, 101, 83, 23, 42 };
	half_size_t right = 35;
	uint8_t dst[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	auto code = Base256uMath::multiply(left, sizeof(left), &right, sizeof(right), dst, 0);
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == i);
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::multiply::in_place_ideal_case() {
	std::size_t left = 25603;
	half_size_t right = 6416;
	decltype(left) answer = left * right;
	auto code = Base256uMath::multiply(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::multiply::in_place_big_ideal_case() {
	uint8_t left[] = { 138, 204, 16, 163, 74, 68, 68, 39, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t right[] = { 72, 80, 180, 160, 32, 125, 248, 160, 102 };
	auto code = Base256uMath::multiply(left, sizeof(left), right, sizeof(right));
	uint8_t answer[] = { 208, 166, 172, 45, 217, 162, 152, 6, 155, 229, 217, 94, 0, 248, 212, 255, 220, 0 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::multiply::in_place_multiply_zero() {
	uint8_t left[] = { 253, 112, 1, 250, 242, 174, 77, 35, 242 };
	uint8_t right[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	auto code = Base256uMath::multiply(left, sizeof(left), right, sizeof(right));
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == 0);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::multiply::in_place_multiply_one() {
	uint8_t left[] = { 1, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t right[] = { 14, 92, 130, 38, 174, 216, 149, 5, 169 };
	auto code = Base256uMath::multiply(left, sizeof(left), right, sizeof(right));
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == right[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::multiply::in_place_left_n_greater_than_right_n() {
	uint8_t left[] = { 217, 86, 117, 188, 220, 28, 236, 105, 119, 0, 0, 0, 0 };
	std::size_t right = 707070;
	assert(sizeof(left) > sizeof(right));
	auto code = Base256uMath::multiply(left, sizeof(left), &right, sizeof(right));
	uint8_t answer[] = { 78, 140, 22, 130, 144, 79, 141, 155, 222, 91, 8, 5, 0 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::multiply::in_place_left_n_less_than_right_n() {
	uint8_t left[] = { 254, 201, 10 };
	uint8_t right[] = { 217, 86, 117, 188, 220, 28, 236, 105, 119 };
	auto code = Base256uMath::multiply(&left, sizeof(left), right, sizeof(right));
	uint8_t answer[] = { 78, 140, 22, 130, 144, 79, 141, 155, 222, 91, 8, 5, 0 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}

Base256uMathTests::divide::divide() {
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
void Base256uMathTests::divide::ideal_case() {
	std::size_t left = 0b11001000000000111111010,
		right = 0b1100100010000,
		dst, mod,
		answer = left / right,
		answer_mod = left % right;
	auto code = Base256uMath::divide(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst),
		&mod, sizeof(mod)
	);
	assert(dst == answer);
	assert(mod == answer_mod);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide::big_ideal_case() {
	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	uint8_t dst[10];
	uint8_t mod[10];
	uint8_t answer[] = { 91, 1 };
	uint8_t answer_mod[] = { 101, 110, 228, 117, 44, 39, 11, 193, 83 };
	auto code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	for (uint8_t i = 0; i < sizeof(answer); i++) {
		assert(dst[i] == answer[i]);
	}
	for (uint8_t i = 0; i < sizeof(answer_mod); i++) {
		assert(mod[i] == answer_mod[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide::left_is_zero() {
	// if left is zero, then dst and mod become all zeros

	uint8_t left[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	uint8_t dst[sizeof(left)];
	uint8_t mod[sizeof(left)];
	auto code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == 0);
		assert(mod[i] == 0);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide::right_is_zero() {
	// if right is zero, then nothing happens and a division by zero error code is returned

	uint8_t left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t right = 0;
	uint8_t dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	uint8_t mod[] = { 20, 21, 22, 23, 24, 25, 26, 27, 28 };
	auto code = Base256uMath::divide(
		left, sizeof(left),
		&right, sizeof(right),
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == i);
		assert(dst[i] == (i + 10));
		assert(mod[i] == (i + 20));
	}
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}
void Base256uMathTests::divide::right_n_zero() {
	// if right_n is zero, then right is assumed to be all zeros and the function
	// behaves as if right were zero.

	uint8_t left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t right = 5;
	uint8_t dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	uint8_t mod[] = { 20, 21, 22, 23, 24, 25, 26, 27, 28 };
	auto code = Base256uMath::divide(
		left, sizeof(left),
		&right, 0,
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == i);
		assert(dst[i] == (i + 10));
		assert(mod[i] == (i + 20));
	}
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}
void Base256uMathTests::divide::left_n_less() {
	// left > right, but 0 < left_n < right_n. 
	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 116 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 45, 0, 0 };
	uint8_t dst[sizeof(left)];
	uint8_t mod[sizeof(left)];
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	uint8_t answer[] = { 139, 2, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t answer_mod[] = { 69, 102, 238, 175, 91, 120, 162, 41, 0 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	for (uint8_t i = 0; i < sizeof(mod); i++) {
		assert(mod[i] == answer_mod[i]);
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::divide::left_n_zero() {
	// if left_n is zero, then it is assumed to be all zeros.
	// that means dst and mod will be all zeros

	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 116 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 45, 0, 0 };
	uint8_t dst[sizeof(left)];
	uint8_t mod[sizeof(left)];
	auto code = Base256uMath::divide(
		left, 0,
		right, sizeof(right),
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == 0);
		assert(mod[i] == 0);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide::dst_n_less() {
	// If dst_n is less than left_n, truncation might occur.
	// To guarantee no truncation, dst should be the same size as left.
	// If mod is of adequate size, then it should yield the correct answer.
	// In any case, truncation or no, the function should return the truncated warning code.

	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139, 0 };
	uint8_t dst[1]; // the answer has 2 significant characters, so this will demonstrate truncation
	uint8_t mod[sizeof(left)];
	auto code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	uint8_t answer[] = { 91, 1 };
	uint8_t answer_mod[] = { 101, 110, 228, 117, 44, 39, 11, 193, 83 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	for (uint8_t i = 0; i < sizeof(answer_mod); i++) {
		assert(mod[i] == answer_mod[i]);
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::divide::dst_n_zero() {
	// if dst_n is zero, truncation is guaranteed and nothing will happen.
	// Even if mod is of correct size, it will not be changed.
	// The function should return the truncated warning code.

	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	uint8_t dst[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	uint8_t mod[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
	auto code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		dst, 0,
		mod, sizeof(mod)
	);
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == i);
		assert(mod[i] == (10 + i));
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::divide::remainder_n_less() {
	// if remainder_n is less than left_n, then left_n is treated as if it were
	// of size remainder_n. Guarantees truncation.

	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 0, 0, 0, 0 }; //, 88, 139, 0 };
	uint8_t dst[sizeof(left)];
	uint8_t mod[7]; // the remainder has 9 significant chars
	auto code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst),
		mod, sizeof(mod)
	);
	uint8_t answer[] = { 250, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t answer_mod[] = { 91, 30, 23, 149, 189, 75, 0 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	for (uint8_t i = 0; i < sizeof(mod); i++) {
		assert(mod[i] == answer_mod[i]);
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::divide::remainder_n_zero() {
	// if remainder_n is zero, then the function behaves as if left_n
	// was zero. Returns a truncated error code.

	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	uint8_t dst[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	uint8_t mod[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
	auto code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst),
		mod, 0
	);
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == 0);
		assert(mod[i] == (10 + i));
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::divide::in_place_ideal_case() {
	std::size_t left = 0b11001000000000111111010,
		right = 0b1100100010000,
		mod,
		answer = left / right,
		answer_mod = left % right;
	auto code = Base256uMath::divide(
		&left, sizeof(left),
		&right, sizeof(right),
		&mod, sizeof(mod)
	);
	assert(left  == answer);
	assert(mod = answer_mod);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide::in_place_big_ideal_case() {
	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	uint8_t mod[10];
	uint8_t answer[] = { 91, 1 };
	uint8_t answer_mod[] = { 101, 110, 228, 117, 44, 39, 11, 193, 83 };
	auto code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		mod, sizeof(mod)
	);
	for (uint8_t i = 0; i < sizeof(answer); i++) {
		assert(left[i] == answer[i]);
	}
	for (uint8_t i = 0; i < sizeof(answer_mod); i++) {
		assert(mod[i] == answer_mod[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide::in_place_left_is_zero() {
	// if left is zero, then left will be all zeros and mod will be a copy of left.
	// 

	uint8_t left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	uint8_t mod[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	auto code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		mod, sizeof(mod)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == 0);
		assert(mod[i] == i);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide::in_place_left_n_zero() {
	// if left_n is zero, then left and mod are untouched.

	uint8_t left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t right[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	uint8_t mod[] = { 100, 101, 102, 103, 104, 105, 106, 107, 108 };
	auto code = Base256uMath::divide(
		left, 0,
		right, sizeof(right),
		mod, sizeof(mod)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == i);
		assert(mod[i] == (i + 100));
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide::in_place_right_is_zero() {
	// if right is zero, then nothing happens and a division by zero error code is returned

	uint8_t left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t right = 0;
	uint8_t mod[] = { 20, 21, 22, 23, 24, 25, 26, 27, 28 };
	auto code = Base256uMath::divide(
		left, sizeof(left),
		&right, sizeof(right),
		mod, sizeof(mod)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == i);
		assert(mod[i] == (i + 20));
	}
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}
void Base256uMathTests::divide::in_place_right_n_zero() {
	// if right_n is zero, then right is assumed to be all zeros and the function
	// behaves as if right were zero.

	uint8_t left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t right = 5;
	uint8_t mod[] = { 20, 21, 22, 23, 24, 25, 26, 27, 28 };
	auto code = Base256uMath::divide(
		left, sizeof(left),
		&right, 0,
		mod, sizeof(mod)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == i);
		assert(mod[i] == (i + 20));
	}
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}
void Base256uMathTests::divide::in_place_left_n_less() {
	// left > right, but 0 < left_n < right_n. 
	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 116 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 45, 0, 0 };
	uint8_t mod[sizeof(left)];
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		mod, sizeof(mod)
	);
	uint8_t answer[] = { 139, 2, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t answer_mod[] = { 69, 102, 238, 175, 91, 120, 162, 41, 0 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	for (uint8_t i = 0; i < sizeof(mod); i++) {
		assert(mod[i] == answer_mod[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide::in_place_remainder_n_less() {
	// if remainder_n is less than left_n, then left_n is treated as if it were
	// of size remainder_n.

	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 0, 0, 0, 0 }; //, 88, 139, 0 };
	uint8_t mod[7]; // the remainder has 9 significant chars
	auto code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		mod, sizeof(mod)
	);
	uint8_t answer[] = { 250, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t answer_mod[] = { 91, 30, 23, 149, 189, 75, 0 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	for (uint8_t i = 0; i < sizeof(mod); i++) {
		assert(mod[i] == answer_mod[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide::in_place_remainder_n_zero() {
	// if remainder_n is zero, then the function behaves as if left_n
	// was zero. 

	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	uint8_t mod[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
	auto code = Base256uMath::divide(
		left, sizeof(left),
		right, sizeof(right),
		mod, 0
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == 0);
		assert(mod[i] == (10 + i));
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}

Base256uMathTests::divide_no_mod::divide_no_mod() {
	ideal_case();
	big_ideal_case();
	left_is_zero();
	left_n_zero();
	right_is_zero();
	right_n_zero();
	left_n_less();
	dst_n_less();
	dst_n_zero();

	in_place_ideal_case();
	in_place_big_ideal_case();
	in_place_left_is_zero();
	in_place_left_n_zero();
	in_place_right_is_zero();
	in_place_right_n_zero();
	in_place_left_n_less();
}
void Base256uMathTests::divide_no_mod::ideal_case() {
	std::size_t left = 0b11001000000000111111010,
		right = 0b1100100010000,
		dst,
		answer = left / right;
	auto code = Base256uMath::divide_no_mod(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst)
	);
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide_no_mod::big_ideal_case() {
	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	uint8_t dst[10];
	uint8_t answer[] = { 91, 1 };
	auto code = Base256uMath::divide_no_mod(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	for (uint8_t i = 0; i < sizeof(answer); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide_no_mod::left_is_zero() {
	// if left is zero, then dst and mod become all zeros

	uint8_t left[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	uint8_t dst[sizeof(left)];
	auto code = Base256uMath::divide_no_mod(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == 0);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide_no_mod::right_is_zero() {
	// if right is zero, then nothing happens and a division by zero error code is returned

	uint8_t left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t right = 0;
	uint8_t dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	auto code = Base256uMath::divide_no_mod(
		left, sizeof(left),
		&right, sizeof(right),
		dst, sizeof(dst)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == i);
		assert(dst[i] == (i + 10));
	}
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}
void Base256uMathTests::divide_no_mod::right_n_zero() {
	// if right_n is zero, then right is assumed to be all zeros and the function
	// behaves as if right were zero.

	uint8_t left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t right = 5;
	uint8_t dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	auto code = Base256uMath::divide_no_mod(
		left, sizeof(left),
		&right, 0,
		dst, sizeof(dst)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == i);
		assert(dst[i] == (i + 10));
	}
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}
void Base256uMathTests::divide_no_mod::left_n_less() {
	// left > right, but 0 < left_n < right_n. 
	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 116 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 45, 0, 0 };
	uint8_t dst[sizeof(left)];
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::divide_no_mod(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	uint8_t answer[] = { 139, 2, 0, 0, 0, 0, 0, 0, 0 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::divide_no_mod::left_n_zero() {
	// if left_n is zero, then it is assumed to be all zeros.
	// that means dst and mod will be all zeros

	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 116 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 45, 0, 0 };
	uint8_t dst[sizeof(left)];
	auto code = Base256uMath::divide_no_mod(
		left, 0,
		right, sizeof(right),
		dst, sizeof(dst)
	);
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == 0);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide_no_mod::dst_n_less() {
	// If dst_n is less than left_n, truncation might occur.
	// To guarantee no truncation, dst should be the same size as left.
	// If mod is of adequate size, then it should yield the correct answer.
	// In any case, truncation or no, the function should return the truncated warning code.

	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139, 0 };
	uint8_t dst[1]; // the answer has 2 significant characters, so this will demonstrate truncation
	auto code = Base256uMath::divide_no_mod(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	uint8_t answer[] = { 91, 1 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::divide_no_mod::dst_n_zero() {
	// if dst_n is zero, truncation is guaranteed and nothing will happen.
	// Even if mod is of correct size, it will not be changed.
	// The function should return the truncated warning code.

	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	uint8_t dst[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	auto code = Base256uMath::divide_no_mod(
		left, sizeof(left),
		right, sizeof(right),
		dst, 0
	);
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == i);
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::divide_no_mod::in_place_ideal_case() {
	std::size_t left = 0b11001000000000111111010,
		right = 0b1100100010000,
		answer = left / right;
	auto code = Base256uMath::divide_no_mod(
		&left, sizeof(left),
		&right, sizeof(right)
	);
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide_no_mod::in_place_big_ideal_case() {
	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	uint8_t answer[] = { 91, 1 };
	auto code = Base256uMath::divide_no_mod(
		left, sizeof(left),
		right, sizeof(right)
	);
	for (uint8_t i = 0; i < sizeof(answer); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide_no_mod::in_place_left_is_zero() {
	// if left is zero, then dst and mod become all zeros

	uint8_t left[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	auto code = Base256uMath::divide_no_mod(
		left, sizeof(left),
		right, sizeof(right)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == 0);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide_no_mod::in_place_right_is_zero() {
	// if right is zero, then nothing happens and a division by zero error code is returned

	uint8_t left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t right = 0;
	auto code = Base256uMath::divide_no_mod(
		left, sizeof(left),
		&right, sizeof(right)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == i);
	}
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}
void Base256uMathTests::divide_no_mod::in_place_right_n_zero() {
	// if right_n is zero, then right is assumed to be all zeros and the function
	// behaves as if right were zero.

	uint8_t left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t right = 5;
	auto code = Base256uMath::divide_no_mod(
		left, sizeof(left),
		&right, 0
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == i);
	}
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}
void Base256uMathTests::divide_no_mod::in_place_left_n_less() {
	// left > right, but 0 < left_n < right_n. 
	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 116 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 45, 0, 0 };
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::divide_no_mod(
		left, sizeof(left),
		right, sizeof(right)
	);
	uint8_t answer[] = { 139, 2, 0, 0, 0, 0, 0, 0, 0 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::divide_no_mod::in_place_left_n_zero() {
	// if left_n is zero, then left is untouched.
	// that means dst and mod will be all zeros

	uint8_t left[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	uint8_t right[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	auto code = Base256uMath::divide_no_mod(
		left, 0,
		right, sizeof(right)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == (i + 10));
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}

Base256uMathTests::mod::mod() {
	ideal_case();
	big_ideal_case();
	left_is_zero();
	left_n_zero();
	right_is_zero();
	right_n_zero();
	left_n_less();
	dst_n_less();
	dst_n_zero();

	in_place_ideal_case();
	in_place_big_ideal_case();
	in_place_left_is_zero();
	in_place_left_n_zero();
	in_place_right_is_zero();
	in_place_right_n_zero();
	in_place_left_n_less();
}
void Base256uMathTests::mod::ideal_case() {
	std::size_t left = 0b11001000000000111111010,
		right = 0b1100100010000,
		dst,
		answer = left % right;
	auto code = Base256uMath::mod(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::mod::big_ideal_case() {
	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	uint8_t dst[10];
	uint8_t answer[] = { 101, 110, 228, 117, 44, 39, 11, 193, 83, 0 };
	auto code = Base256uMath::mod(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::mod::left_is_zero() {
	uint8_t left[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	uint8_t dst[sizeof(left)];
	auto code = Base256uMath::mod(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == 0);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::mod::left_n_zero() {
	// if left_n is zero, then it is assumed to be all zeros.
	// that means dst will be all zeros

	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 116 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 45, 0, 0 };
	uint8_t dst[sizeof(left)];
	auto code = Base256uMath::mod(
		left, 0,
		right, sizeof(right),
		dst, sizeof(dst)
	);
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == 0);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::mod::right_is_zero() {
	// if right is zero, then nothing happens and a division by zero error code is returned

	uint8_t left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t right = 0;
	uint8_t dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	auto code = Base256uMath::mod(
		left, sizeof(left),
		&right, sizeof(right),
		dst, sizeof(dst)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == i);
		assert(dst[i] == (i + 10));
	}
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}
void Base256uMathTests::mod::right_n_zero() {
	// if right_n is zero, then right is assumed to be all zeros and the function
	// behaves as if right were zero.

	uint8_t left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t right = 5;
	uint8_t dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	auto code = Base256uMath::mod(
		left, sizeof(left),
		&right, 0,
		dst, sizeof(dst)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == i);
		assert(dst[i] == (i + 10));
	}
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}
void Base256uMathTests::mod::left_n_less() {
	// left > right, but 0 < left_n < right_n. 
	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 116 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 45, 0, 0 };
	uint8_t dst[sizeof(left)];
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::mod(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	uint8_t answer[] = { 69, 102, 238, 175, 91, 120, 162, 41, 0 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::mod::dst_n_less() {
	// if dst_n is less than left_n, then left_n is treated as if it were
	// of size dst_n. Guarantees truncation.

	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 0, 0, 0, 0 };
	uint8_t dst[7]; // the remainder has 9 significant chars
	auto code = Base256uMath::mod(
		left, sizeof(left),
		right, sizeof(right),
		dst, sizeof(dst)
	);
	uint8_t answer[] = { 91, 30, 23, 149, 189, 75, 0 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::mod::dst_n_zero() {
	// if dst_n is zero, then nothing happens. Returns a truncated error code.

	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	uint8_t dst[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	auto code = Base256uMath::mod(
		left, sizeof(left),
		right, sizeof(right),
		dst, 0
	);
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == i);
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::mod::in_place_ideal_case() {
	std::size_t left = 0b11001000000000111111010,
		right = 0b1100100010000,
		answer = left % right;
	auto code = Base256uMath::mod(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::mod::in_place_big_ideal_case() {
	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 89, 189 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	uint8_t answer[] = { 101, 110, 228, 117, 44, 39, 11, 193, 83, 0 };
	auto code = Base256uMath::mod(left, sizeof(left), right, sizeof(right));
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::mod::in_place_left_is_zero() {
	uint8_t left[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 115, 139 };
	auto code = Base256uMath::mod(
		left, sizeof(left),
		right, sizeof(right)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == 0);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::mod::in_place_left_n_zero() {
	// if left_n is zero, then left will be untouched.

	uint8_t left[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	uint8_t right[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	auto code = Base256uMath::mod(
		left, 0,
		right, sizeof(right)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == (10 + i));
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::mod::in_place_right_is_zero() {
	// if right is zero, then nothing happens and a division by zero error code is returned

	uint8_t left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t right = 0;
	auto code = Base256uMath::mod(
		left, sizeof(left),
		&right, sizeof(right)
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == i);
	}
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}
void Base256uMathTests::mod::in_place_right_n_zero() {
	// if right_n is zero, then right is assumed to be all zeros and the function
	// behaves as if right were zero.

	uint8_t left[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t right = 5;
	auto code = Base256uMath::mod(
		left, sizeof(left),
		&right, 0
	);
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == i);
	}
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}
void Base256uMathTests::mod::in_place_left_n_less() {
	// left > right, but 0 < left_n < right_n. 
	uint8_t left[] = { 23, 0, 84, 101, 183, 110, 254, 208, 116 };
	uint8_t right[] = { 182, 193, 139, 54, 147, 128, 223, 45, 0, 0 };
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::mod(
		left, sizeof(left),
		right, sizeof(right)
	);
	uint8_t answer[] = { 69, 102, 238, 175, 91, 120, 162, 41, 0 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}

Base256uMathTests::log2::log2() {
	ideal_case();
	big_ideal_case();
	src_is_zero();
	src_n_zero();
	dst_n_zero();
}
void Base256uMathTests::log2::ideal_case() {
	half_size_t src = 0b00010000000100000;
	std::size_t dst = 0;
	assert(sizeof(src) <= sizeof(dst));
	auto code = Base256uMath::log2(&src, sizeof(src), &dst, sizeof(dst));
	assert(dst == 13);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::log2::big_ideal_case(){
	uint8_t src[] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
	uint8_t dst[sizeof(std::size_t) + 1];
	auto code = Base256uMath::log2(src, sizeof(src), dst, sizeof(dst));
	assert(dst[0] == 64);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::log2::src_is_zero() {
	// if *src is zero, then nothing happens to dst and divide by zero error is returned

	half_size_t src = 0;
	std::size_t dst = 1234567890;
	assert(src == 0);
	auto code = Base256uMath::log2(&src, sizeof(src), &dst, sizeof(dst));
	assert(dst == 1234567890);
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}
void Base256uMathTests::log2::src_n_zero() {
	// if src_n is zero, then nothing happens to dst and divide by zero error is returned

	half_size_t src = 1337;
	std::size_t dst = 1234567890;
	auto code = Base256uMath::log2(&src, 0, &dst, sizeof(dst));
	assert(dst == 1234567890);
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}
void Base256uMathTests::log2::dst_n_zero() {
	// if dst_n is zero, then nothing happens and truncated warning code is returned
	
	half_size_t src = 1337;
	std::size_t dst = 1234567890;
	auto code = Base256uMath::log2(&src, sizeof(src), &dst, 0);
	assert(dst == 1234567890);
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}

Base256uMathTests::log256::log256() {
	ideal_case();
	big_ideal_case();
	src_is_zero();
	src_n_zero();
}
void Base256uMathTests::log256::ideal_case() {
	std::size_t src = 16289501482060108362,
		dst;
	auto code = Base256uMath::log256(&src, sizeof(src), &dst);
	assert(dst == sizeof(src) - 1);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::log256::big_ideal_case() {
	uint8_t src[] = { 139, 46, 187, 204, 123, 55, 217, 147, 102, 0 };
	std::size_t dst;
	auto code = Base256uMath::log256(src, sizeof(src), &dst);
	assert(dst == 8);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::log256::src_is_zero() {
	// if src is zero, then return divide by zero error

	std::size_t zero = 0;
	std::size_t dst = 1467;
	auto code = Base256uMath::log256(&zero, sizeof(zero), &dst);
	assert(dst == 1467);
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}
void Base256uMathTests::log256::src_n_zero() {
	// if src_n is zero, then return divide by zero error

	std::size_t src = 230841808201;
	std::size_t dst = 1337;
	auto code = Base256uMath::log256(&src, 0, &dst);
	assert(dst == 1337); // nothing changed
	assert(code == Base256uMath::ErrorCodes::DIVIDE_BY_ZERO);
}

Base256uMathTests::bitwise_and::bitwise_and() {
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
void Base256uMathTests::bitwise_and::ideal_case() {
	half_size_t left = 49816,
		right = 13925,
		dst;
	assert(sizeof(left) == sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::bitwise_and(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == (left & right));
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_and::big_ideal_case() {
	uint8_t left[] = { 177, 191, 253, 203, 209, 95, 131, 49, 194 };
	uint8_t right[] = { 97, 78, 90, 246, 21, 127, 59, 186, 23 };
	uint8_t dst[9];
	assert(sizeof(left) == sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::bitwise_and(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == (left[i] & right[i]));
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_and::left_bigger() {
	std::size_t left = 3477483;
	half_size_t right = 16058;
	decltype(left) dst;
	assert(sizeof(left) > sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::bitwise_and(
		&left, sizeof(left),
		&right, sizeof(right),
		&dst, sizeof(dst)
	);
	assert(dst == (left & right));
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_and::left_smaller() {
	half_size_t left = 20968;
	std::size_t right = 226081,
		dst;
	assert(sizeof(left) < sizeof(right) && sizeof(right) == sizeof(dst));
	auto code = Base256uMath::bitwise_and(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == (left & right));
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_and::big_left_bigger() {
	uint8_t left[] = { 53, 138, 217, 66, 48, 69, 213, 139, 91, 163, 230, 8 };
	uint8_t right[] = { 16, 126, 35, 76, 137, 248, 62, 10, 11 };
	uint8_t dst[12];
	assert(sizeof(left) > sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::bitwise_and(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 16, 10, 1, 64, 0, 64, 20, 10, 11, 0, 0, 0 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_and::big_left_smaller() {
	uint8_t left[] = { 191, 126, 201, 200, 181, 4, 177, 127, 163 };
	uint8_t right[] = { 145, 53, 22, 29, 169, 149, 201, 111, 97, 66, 21, 27 };
	uint8_t dst[12];
	assert(sizeof(left) < sizeof(right) && sizeof(right) == sizeof(dst));
	auto code = Base256uMath::bitwise_and(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 145, 52, 0, 8, 161, 4, 129, 111, 33, 0, 0, 0 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_and::dst_too_small() {
	// if dst is too small to accomodate, then the results will be truncated and
	// the appropriate code will be returned

	std::size_t left = 61107471;
	half_size_t right = 186824;
	uint8_t dst;
	assert(sizeof(left) > sizeof(dst) || sizeof(right) > sizeof(dst));
	auto code = Base256uMath::bitwise_and(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	decltype(dst) answer = left & right;
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::bitwise_and::big_dst_too_small() {
	uint8_t left[] = { 31, 204, 101, 59, 181, 129, 29, 143, 123, 111 };
	uint8_t right[] = { 159, 29, 249, 164, 61, 95, 169, 60, 199, 5, 254 };
	uint8_t dst[9];
	assert(sizeof(left) > sizeof(dst) || sizeof(right) > sizeof(dst));
	auto code = Base256uMath::bitwise_and(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == (left[i] & right[i]));
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::bitwise_and::left_n_zero() {
	// if left_n is zero, then it is treated as all zeros

	std::size_t left = 2912879481, 
		right = -1,
		dst;
	assert(sizeof(right) == sizeof(dst));
	auto code = Base256uMath::bitwise_and(&left, 0, &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == 0);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_and::right_n_zero() {
	// if right_n is zero, then it is treated as all zeros

	std::size_t left = -1,
		right = 2912879481,
		dst;
	assert(sizeof(left) == sizeof(dst));
	auto code = Base256uMath::bitwise_and(&left, sizeof(left), &right, 0, &dst, sizeof(dst));
	assert(dst == 0);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_and::dst_n_zero() {
	// if dst_n is zero, then nothing happens but returns a truncated error code

	std::size_t left = 2739824923,
		right = 248020302;
	uint8_t dst = 223,
		answer = dst;
	auto code = Base256uMath::bitwise_and(&left, sizeof(left), &right, sizeof(right), &dst, 0);
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}

void Base256uMathTests::bitwise_and::in_place_ideal_case() {
	half_size_t left = 49816,
		right = 13925;
	decltype(left) answer = left & right;
	assert(sizeof(left) == sizeof(right));
	auto code = Base256uMath::bitwise_and(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_and::in_place_big_ideal_case() {
	uint8_t left[] = { 177, 191, 253, 203, 209, 95, 131, 49, 194 };
	uint8_t right[] = { 97, 78, 90, 246, 21, 127, 59, 186, 23 };
	uint8_t answer[] = { 33, 14, 88, 194, 17, 95, 3, 48, 2 };
	assert(sizeof(left) == sizeof(right));
	auto code = Base256uMath::bitwise_and(left, sizeof(left), right, sizeof(right));
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_and::in_place_left_bigger() {
	std::size_t left = 3477482;
	half_size_t right = 16058;
	decltype(left) answer = left & right;
	assert(sizeof(left) > sizeof(right));
	auto code = Base256uMath::bitwise_and(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_and::in_place_left_smaller() {
	half_size_t left = 20968;
	std::size_t right = 22673;
	decltype(left) answer = left & right;
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::bitwise_and(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_and::in_place_big_left_bigger() {
	uint8_t left[] = { 53, 138, 217, 66, 48, 69, 213, 139, 91, 163, 230, 8 };
	uint8_t right[] = { 16, 126, 35, 76, 137, 248, 62, 10, 11 };
	assert(sizeof(left) > sizeof(right));
	auto code = Base256uMath::bitwise_and(left, sizeof(left), right, sizeof(right));
	uint8_t answer[] = { 16, 10, 1, 64, 0, 64, 20, 10, 11, 0, 0, 0 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_and::in_place_big_left_smaller() {
	uint8_t left[] = { 191, 126, 201, 200, 181, 4, 177, 127, 163 };
	uint8_t right[] = { 145, 53, 22, 29, 169, 149, 201, 111, 97, 66, 21, 27 };
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::bitwise_and(left, sizeof(left), right, sizeof(right));
	uint8_t answer[] = { 145, 52, 0, 8, 161, 4, 129, 111, 33, 0, 0, 0 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_and::in_place_left_n_zero() {
	// if left_n is zero, then nothing happens

	std::size_t left = 2912879481;
	half_size_t right = -1;
	decltype(left) answer = left;
	auto code = Base256uMath::bitwise_and(&left, 0, &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_and::in_place_right_n_zero() {
	// if right_n is zero, then it is treated as all zeros

	half_size_t left = -1;
	std::size_t right = 2912879481;
	decltype(left) answer = 0;
	auto code = Base256uMath::bitwise_and(&left, sizeof(left), &right, 0);
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}

Base256uMathTests::bitwise_or::bitwise_or() {
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
void Base256uMathTests::bitwise_or::ideal_case() {
	unsigned short left = 49816;
	unsigned short right = 13925;
	unsigned short dst;
	assert(sizeof(left) == sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::bitwise_or(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == (left | right));
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_or::big_ideal_case() {
	uint8_t left[] = { 177, 191, 253, 203, 209, 95, 131, 49, 194 };
	uint8_t right[] = { 97, 78, 90, 246, 21, 127, 59, 186, 23 };
	uint8_t dst[9];
	assert(sizeof(left) == sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::bitwise_or(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == (left[i] | right[i]));
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_or::left_bigger() {
	std::size_t left = 347748386;
	half_size_t right = 16058;
	decltype(left) dst;
	assert(sizeof(left) > sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::bitwise_or(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	decltype(left) answer = left | right;
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_or::left_smaller() {
	half_size_t left = 20968;
	std::size_t right = 227385081;
	decltype(right) dst;
	assert(sizeof(left) < sizeof(right) && sizeof(right) == sizeof(dst));
	auto code = Base256uMath::bitwise_or(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == (left | right));
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_or::big_left_bigger() {
	uint8_t left[] = { 53, 138, 217, 66, 48, 69, 213, 139, 91, 163, 230, 8 };
	uint8_t right[] = { 16, 126, 35, 76, 137, 248, 62, 10, 11 };
	uint8_t dst[12];
	assert(sizeof(left) > sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::bitwise_or(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 53, 254, 251, 78, 185, 253, 255, 139, 91, 163, 230, 8 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_or::big_left_smaller() {
	uint8_t left[] = { 191, 126, 201, 200, 181, 4, 177, 127, 163 };
	uint8_t right[] = { 145, 53, 22, 29, 169, 149, 201, 111, 97, 66, 21, 27 };
	uint8_t dst[12];
	assert(sizeof(left) < sizeof(right) && sizeof(right) == sizeof(dst));
	auto code = Base256uMath::bitwise_or(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 191, 127, 223, 221, 189, 149, 249, 127, 227, 66, 21, 27 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_or::dst_too_small() {
	// if dst is too small to accomodate, then the results will be truncated and
	// the appropriate code will be returned

	std::size_t left = 61133978;
	std::size_t right = 11834;
	half_size_t dst;
	assert(sizeof(left) > sizeof(dst) || sizeof(right) > sizeof(dst));
	auto code = Base256uMath::bitwise_or(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	decltype(dst) answer = left | right;
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::bitwise_or::big_dst_too_small() {
	uint8_t left[] = { 31, 204, 101, 59, 181, 129, 29, 143, 123, 111 };
	uint8_t right[] = { 159, 29, 249, 164, 61, 95, 169, 60, 199, 5, 254 };
	uint8_t dst[9];
	assert(sizeof(left) > sizeof(dst) || sizeof(right) > sizeof(dst));
	auto code = Base256uMath::bitwise_or(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == (left[i] | right[i]));
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::bitwise_or::left_n_zero() {
	// if left_n is zero, then it is treated as all zeros

	std::size_t left = 2912879481;
	std::size_t right = -1;
	decltype(right) dst;
	assert(sizeof(right) == sizeof(dst));
	auto code = Base256uMath::bitwise_or(&left, 0, &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == right);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_or::right_n_zero() {
	// if right_n is zero, then it is treated as all zeros

	unsigned int left = -1;
	std::size_t right = 291287941;
	unsigned int dst = -1;
	assert(sizeof(left) == sizeof(dst));
	auto code = Base256uMath::bitwise_or(&left, sizeof(left), &right, 0, &dst, sizeof(dst));
	assert(dst == left);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_or::dst_n_zero() {
	// if dst_n is zero, then nothing happens but returns a truncated error code

	std::size_t left = 273983;
	std::size_t right = 24885;
	uint8_t dst = 223;
	auto code = Base256uMath::bitwise_or(&left, sizeof(left), &right, sizeof(right), &dst, 0);
	assert(dst == 223);
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::bitwise_or::in_place_ideal_case() {
	unsigned short left = 49816;
	unsigned short right = 13925;
	decltype(left) answer = left | right;
	assert(sizeof(left) == sizeof(right));
	auto code = Base256uMath::bitwise_or(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_or::in_place_big_ideal_case() {
	uint8_t left[] = { 177, 191, 253, 203, 209, 95, 131, 49, 194 };
	uint8_t right[] = { 97, 78, 90, 246, 21, 127, 59, 186, 23 };
	uint8_t answer[] = { 241, 255, 255, 255, 213, 127, 187, 187, 215 };
	assert(sizeof(left) == sizeof(right));
	auto code = Base256uMath::bitwise_or(left, sizeof(left), right, sizeof(right));
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_or::in_place_left_bigger() {
	unsigned int left = 347748382;
	unsigned short right = 16058;
	decltype(left) answer = left | right;
	assert(sizeof(left) > sizeof(right));
	auto code = Base256uMath::bitwise_or(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_or::in_place_left_smaller() {
	unsigned short left = 20968;
	unsigned int right = 226738508;
	decltype(left) answer = left | right;
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::bitwise_or(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_or::in_place_big_left_bigger() {
	uint8_t left[] = { 53, 138, 217, 66, 48, 69, 213, 139, 91, 163, 230, 8 };
	uint8_t right[] = { 16, 126, 35, 76, 137, 248, 62, 10, 11 };
	assert(sizeof(left) > sizeof(right));
	auto code = Base256uMath::bitwise_or(left, sizeof(left), right, sizeof(right));
	uint8_t answer[] = { 53, 254, 251, 78, 185, 253, 255, 139, 91, 163, 230, 8 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_or::in_place_big_left_smaller() {
	uint8_t left[] = { 191, 126, 201, 200, 181, 4, 177, 127, 163 };
	uint8_t right[] = { 145, 53, 22, 29, 169, 149, 201, 111, 97, 66, 21, 27 };
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::bitwise_or(left, sizeof(left), right, sizeof(right));
	uint8_t answer[] = { 191, 127, 223, 221, 189, 149, 249, 127, 227, 66, 21, 27 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_or::in_place_left_n_zero() {
	// if left_n is zero, then nothing happens

	std::size_t left = 29128481;
	half_size_t right = -1;
	decltype(left) answer = 29128481;
	auto code = Base256uMath::bitwise_or(&left, 0, &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_or::in_place_right_n_zero() {
	// if right_n is zero, then it is treated as all zeros

	half_size_t left = -1;
	std::size_t right = 2912879481;
	decltype(left) answer = -1;
	auto code = Base256uMath::bitwise_or(&left, sizeof(left), &right, 0);
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}

Base256uMathTests::bitwise_xor::bitwise_xor() {
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
void Base256uMathTests::bitwise_xor::ideal_case() {
	unsigned short left = 49816;
	unsigned short right = 13925;
	unsigned short dst;
	assert(sizeof(left) == sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == (left ^ right));
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_xor::big_ideal_case() {
	uint8_t left[] = { 177, 191, 253, 203, 209, 95, 131, 49, 194 };
	uint8_t right[] = { 97, 78, 90, 246, 21, 127, 59, 186, 23 };
	uint8_t dst[9];
	assert(sizeof(left) == sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::bitwise_xor(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == (left[i] ^ right[i]));
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_xor::left_bigger() {
	unsigned int left = 347748382;
	unsigned short right = 16058;
	unsigned int dst;
	assert(sizeof(left) > sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	decltype(dst) answer = left ^ right;
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_xor::left_smaller() {
	unsigned short left = 20968;
	unsigned int right = 226738581;
	unsigned int dst;
	assert(sizeof(left) < sizeof(right) && sizeof(right) == sizeof(dst));
	auto code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == (left ^ right));
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_xor::big_left_bigger() {
	uint8_t left[] = { 53, 138, 217, 66, 48, 69, 213, 139, 91, 163, 230, 8 };
	uint8_t right[] = { 16, 126, 35, 76, 137, 248, 62, 10, 11 };
	uint8_t dst[12];
	assert(sizeof(left) > sizeof(right) && sizeof(left) == sizeof(dst));
	auto code = Base256uMath::bitwise_xor(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 37, 244, 250, 14, 185, 189, 235, 129, 80, 163, 230, 8 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_xor::big_left_smaller() {
	uint8_t left[] = { 191, 126, 201, 200, 181, 4, 177, 127, 163 };
	uint8_t right[] = { 145, 53, 22, 29, 169, 149, 201, 111, 97, 66, 21, 27 };
	uint8_t dst[12];
	assert(sizeof(left) < sizeof(right) && sizeof(right) == sizeof(dst));
	auto code = Base256uMath::bitwise_xor(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	uint8_t answer[] = { 46, 75, 223, 213, 28, 145, 120, 16, 194, 66, 21, 27 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_xor::dst_too_small() {
	// if dst is too small to accomodate, then the results will be truncated and
	// the appropriate code will be returned

	std::size_t left = 6113397841731107471;
	half_size_t right = 1868241834;
	half_size_t dst;
	assert(sizeof(left) > sizeof(dst) || sizeof(right) > sizeof(dst));
	auto code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, sizeof(right), &dst, sizeof(dst));
	decltype(dst) answer = left ^ right;
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::bitwise_xor::big_dst_too_small() {
	uint8_t left[] = { 31, 204, 101, 59, 181, 129, 29, 143, 123, 111 };
	uint8_t right[] = { 159, 29, 249, 164, 61, 95, 169, 60, 199, 5, 254 };
	uint8_t dst[9];
	assert(sizeof(left) > sizeof(dst) || sizeof(right) > sizeof(dst));
	auto code = Base256uMath::bitwise_xor(left, sizeof(left), right, sizeof(right), dst, sizeof(dst));
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == (left[i] ^ right[i]));
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::bitwise_xor::left_n_zero() {
	// if left_n is zero, then it is treated as all zeros

	std::size_t left = 2912879481;
	unsigned int right = -1,
		dst = 5978137491;
	assert(sizeof(right) == sizeof(dst));
	auto code = Base256uMath::bitwise_xor(&left, 0, &right, sizeof(right), &dst, sizeof(dst));
	assert(dst == right);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_xor::right_n_zero() {
	// if right_n is zero, then it is treated as all zeros

	unsigned int left = -1;
	std::size_t right = 2912879481;
	unsigned int dst = 5978137491;
	assert(sizeof(left) == sizeof(dst));
	auto code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, 0, &dst, sizeof(dst));
	assert(dst == left);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_xor::dst_n_zero() {
	// if dst_n is zero, then nothing happens but returns a truncated error code

	std::size_t left = 2739824923;
	std::size_t right = 248020302;
	uint8_t dst = 223;
	auto code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, sizeof(right), &dst, 0);
	assert(dst == 223);
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::bitwise_xor::in_place_ideal_case() {
	unsigned short left = 49816;
	unsigned short right = 13925;
	decltype(left) answer = left ^ right;
	assert(sizeof(left) == sizeof(right));
	auto code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_xor::in_place_big_ideal_case() {
	uint8_t left[] = { 177, 191, 253, 203, 209, 95, 131, 49, 194 };
	uint8_t right[] = { 97, 78, 90, 246, 21, 127, 59, 186, 23 };
	uint8_t answer[] = { 208, 241, 167, 61, 196, 32, 184, 139, 213 };
	assert(sizeof(left) == sizeof(right));
	auto code = Base256uMath::bitwise_xor(left, sizeof(left), right, sizeof(right));
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_xor::in_place_left_bigger() {
	unsigned int left = 3477483862;
	unsigned short right = 16058;
	decltype(left) answer = left ^ right;
	assert(sizeof(left) > sizeof(right));
	auto code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_xor::in_place_left_smaller() {
	unsigned short left = 20968;
	unsigned int right = 2267385081;
	decltype(left) answer = left ^ right;
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_xor::in_place_big_left_bigger() {
	uint8_t left[] = { 53, 138, 217, 66, 48, 69, 213, 139, 91, 163, 230, 8 };
	uint8_t right[] = { 16, 126, 35, 76, 137, 248, 62, 10, 11 };
	assert(sizeof(left) > sizeof(right));
	auto code = Base256uMath::bitwise_xor(left, sizeof(left), right, sizeof(right));
	uint8_t answer[] = { 37, 244, 250, 14, 185, 189, 235, 129, 80, 163, 230, 8 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_xor::in_place_big_left_smaller() {
	uint8_t left[] = { 191, 126, 201, 200, 181, 4, 177, 127, 163 };
	uint8_t right[] = { 145, 53, 22, 29, 169, 149, 201, 111, 97, 66, 21, 27 };
	assert(sizeof(left) < sizeof(right));
	auto code = Base256uMath::bitwise_xor(left, sizeof(left), right, sizeof(right));
	uint8_t answer[] = { 46, 75, 223, 213, 28, 145, 120, 16, 194, 66, 21, 27 };
	for (uint8_t i = 0; i < sizeof(left); i++) {
		assert(left[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_xor::in_place_left_n_zero() {
	// if left_n is zero, then nothing happens

	std::size_t left = 29128481;
	half_size_t right = -1;
	decltype(left) answer = 29128481;
	auto code = Base256uMath::bitwise_xor(&left, 0, &right, sizeof(right));
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_xor::in_place_right_n_zero() {
	// if right_n is zero, then it is treated as all zeros

	half_size_t left = -1;
	std::size_t right = 2919481;
	decltype(left) answer = -1;
	auto code = Base256uMath::bitwise_xor(&left, sizeof(left), &right, 0);
	assert(left == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}

Base256uMathTests::bitwise_not::bitwise_not() {
	ideal_case();
	big_ideal_case();
	src_n_zero();
}
void Base256uMathTests::bitwise_not::ideal_case() {
	std::size_t src = 2493050980,
		answer = ~src;
	auto code = Base256uMath::bitwise_not(&src, sizeof(src));
	assert(src == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_not::big_ideal_case() {
	uint8_t src[] = { 180, 127, 35, 146, 158, 174, 69, 249, 147 };
	auto code = Base256uMath::bitwise_not(src, sizeof(src));
	uint8_t answer[] = { 75, 128, 220, 109, 97, 81, 186, 6, 108 };
	for (uint8_t i = 0; i < sizeof(src); i++) {
		assert(src[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bitwise_not::src_n_zero() {
	// if src_n is zero, then nothing will happen and no error will be returned.

	uint8_t src[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	auto code = Base256uMath::bitwise_not(src, 0);
	for (uint8_t i = 0; i < sizeof(src); i++) {
		assert(src[i] == i);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}

Base256uMathTests::bit_shift_left::bit_shift_left() {
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
void Base256uMathTests::bit_shift_left::ideal_case() {
	std::size_t src = 14687480,
		dst, answer;
	int code;
	for (std::size_t by = 1; by < sizeof(src) * 8; by++) {
		answer = src << by;
		code = Base256uMath::bit_shift_left(
			&src, sizeof(src),
			&by, sizeof(by),
			&dst, sizeof(dst)
		);
		assert(dst == answer);
		assert(code == Base256uMath::ErrorCodes::OK);
	}
}
void Base256uMathTests::bit_shift_left::big_ideal_case() {
	uint8_t src[] = { 215, 204, 79, 242, 35, 94, 138, 32, 147 };
	uint8_t dst[9];
	std::size_t by = 25;
	assert(sizeof(src) == sizeof(dst));
	uint8_t answer[] = { 0, 0, 0, 174, 153, 159, 228, 71, 188, 20, 65, 38, 1 };
	auto code = Base256uMath::bit_shift_left(src, sizeof(src), &by, sizeof(by), dst, sizeof(dst));
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bit_shift_left::src_n_less_than_by() {
	// if src_n * 8 < by, then dst is all zeros

	std::size_t src = 182389,
		dst = 42070131,
		by = sizeof(src) * 8 + 1;
	auto code = Base256uMath::bit_shift_left(
		&src, sizeof(src), 
		&by, sizeof(by),
		&dst, sizeof(dst)
	);
	assert(dst == 0);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bit_shift_left::src_n_greater_than_dst_n() {
	// if src_n > dst_n, then the answer will be truncated and return the truncated error code

	std::size_t src = 16559212640052646418;
	half_size_t dst, answer;
	int code;
	for (std::size_t by = 1; by < sizeof(src); by++) {
		answer = src << by;
		assert(sizeof(src) > sizeof(dst));
		code = Base256uMath::bit_shift_left(
			&src, sizeof(src),
			&by, sizeof(by),
			&dst, sizeof(dst)
		);
		assert(dst == answer);
		assert(code == Base256uMath::ErrorCodes::TRUNCATED);
	}
}
void Base256uMathTests::bit_shift_left::src_n_less_than_dst_n() {
	// if src_n < dst_n, then the answer will be copied into dst and the excess bytes will
	// become zero.

	half_size_t src = 39816;
	std::size_t dst = 7168245,
		answer;
	int code;
	for (std::size_t by = sizeof(src) + 1; by < sizeof(dst); by++) {
		answer = src << by;
		code = Base256uMath::bit_shift_left(
			&src, sizeof(src),
			&by, sizeof(by),
			&dst, sizeof(dst)
		);
		assert(dst == answer);
		assert(code == Base256uMath::ErrorCodes::OK);
	}
}
void Base256uMathTests::bit_shift_left::src_n_zero() {
	// if src_n is zero, then src is evaluated as all zeros which makes dst all zeros.

	std::size_t src = 2423423,
		dst = 42831231,
		by = 15;
	auto code = Base256uMath::bit_shift_left(&src, 0, &by, sizeof(by), &dst, sizeof(dst));
	assert(dst == 0);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bit_shift_left::dst_n_zero() {
	// if dst_n is zero, then nothing happens and the truncated error code should be returned.

	std::size_t src = 1234567890,
		dst = 987654321,
		by = 10,
		answer = dst;
	auto code = Base256uMath::bit_shift_left(&src, sizeof(src), &by, sizeof(by), &dst, 0);
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::bit_shift_left::by_n_zero() {
	// if by_n is zero, then by is treated as 0, which means src is effectively copied into dst.

	std::size_t src = 246810,
		dst = 1357911,
		by = 25;
	auto code = Base256uMath::bit_shift_left(&src, sizeof(src), &by, 0, &dst, sizeof(dst));
	assert(dst == src);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bit_shift_left::in_place_ideal_case() {
	std::size_t src = 146146596,
		original = src,
		answer;
	int code;
	for (std::size_t by = 1; by < sizeof(src); by++) {
		src = original;
		answer = src << by;
		code = Base256uMath::bit_shift_left(
			&src, sizeof(src),
			&by, sizeof(by)
		);
		assert(src == answer);
		assert(code == Base256uMath::ErrorCodes::OK);
	}
}
void Base256uMathTests::bit_shift_left::in_place_big_ideal_case() {
	uint8_t src[] = { 215, 204, 79, 242, 35, 94, 138, 32, 147 };
	std::size_t by = 25;
	auto code = Base256uMath::bit_shift_left(src, sizeof(src), &by, sizeof(by));
	uint8_t answer[] = { 0, 0, 0, 174, 153, 159, 228, 71, 188, 20, 65, 38, 1 };
	for (uint8_t i = 0; i < sizeof(src); i++) {
		assert(src[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bit_shift_left::in_place_src_n_less_than_by() {
	// if src_n * 8 < by, then src is all zeros
	std::size_t src = 1823827429,
		answer = 0;
	std::size_t by = sizeof(src) * 8 + 1;
	auto code = Base256uMath::bit_shift_left(&src, sizeof(src), &by, sizeof(by));
	assert(src == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bit_shift_left::in_place_src_n_zero() {
	// if src_n is zero, then nothing happens
	std::size_t src = 24234233,
		answer = src;
	std::size_t by = 15;
	auto code = Base256uMath::bit_shift_left(&src, 0, &by, sizeof(by));
	assert(src == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bit_shift_left::in_place_by_n_zero() {
	// if by_n is zero, then nothing happens.

	std::size_t src = 246810,
		answer = src;
	std::size_t by = 25;
	auto code = Base256uMath::bit_shift_left(&src, sizeof(src), &by, 0);
	assert(src == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}

Base256uMathTests::bit_shift_right::bit_shift_right() {
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
void Base256uMathTests::bit_shift_right::ideal_case() {
	std::size_t src = 14687480300692146596,
		dst = 0, answer;
	int code;
	for (std::size_t by = 1; by < sizeof(std::size_t) * 8; by++) {
		assert(sizeof(src) == sizeof(dst) && sizeof(src) * 8 > by);
		answer = src >> by;
		code = Base256uMath::bit_shift_right(&src, sizeof(src), &by, sizeof(by), &dst, sizeof(dst));
		assert(dst == answer);
		assert(code == Base256uMath::ErrorCodes::OK);
	}
}
void Base256uMathTests::bit_shift_right::big_ideal_case() {
	uint8_t src[] = { 215, 204, 79, 242, 35, 94, 138, 32, 147 };
	uint8_t dst[9];
	std::size_t by = 45;
	assert(sizeof(src) == sizeof(dst));
	auto code = Base256uMath::bit_shift_right(src, sizeof(src), &by, sizeof(by), dst, sizeof(dst));
	uint8_t answer[] = { 82, 4, 153, 4, 0, 0, 0, 0, 0 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bit_shift_right::src_n_less_than_by() {
	// if src_n * 8 < by, then dst is all zeros and returns the flow error code

	std::size_t src = 1823827429;
	std::size_t dst = 420735172730131;
	std::size_t by = sizeof(src) * 8 + 1;
	auto code = Base256uMath::bit_shift_right(&src, sizeof(src), &by, sizeof(by), &dst, sizeof(dst));
	assert(dst == 0);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bit_shift_right::src_n_greater_than_dst_n() {
	// if src_n > dst_n, then *sometimes* the answer will be truncated
	// and return the truncated error code

	std::size_t src = 16559212640052646418;
	half_size_t dst, answer;
	int code;
	for (std::size_t by = 1; by < sizeof(dst) * 8; by++) {
		answer = src >> by;
		code = Base256uMath::bit_shift_right(
			&src, sizeof(src),
			&by, sizeof(by),
			&dst, sizeof(dst)
		);
		assert(dst == answer);
		assert(code == Base256uMath::ErrorCodes::TRUNCATED);
	}
}
void Base256uMathTests::bit_shift_right::src_n_less_than_dst_n() {
	// if src_n < dst_n, then the answer will be copied into dst and the excess bytes will
	// become zero.

	half_size_t src = 3256569816;
	std::size_t dst = 7654851108216826745,
		answer;
	int code;
	for (std::size_t by = 1; by < sizeof(dst); by++) {
		answer = src >> by;
		code = Base256uMath::bit_shift_right(
			&src, sizeof(src),
			&by, sizeof(by),
			&dst, sizeof(dst)
		);
		assert(dst == answer);
		assert(code == Base256uMath::ErrorCodes::OK);
	}
}
void Base256uMathTests::bit_shift_right::src_n_zero() {
	// if src_n is zero, then src is evaluated as all zeros which makes dst all zeros.

	half_size_t src = 2423423423,
		dst = 428198231231;
	int code;
	for (std::size_t by = 1; by < sizeof(std::size_t); by++) {
		dst = 428198231231;
		code = Base256uMath::bit_shift_right(
			&src, 0,
			&by, sizeof(by),
			&dst, sizeof(dst)
		);
		assert(dst == 0);
		assert(code == Base256uMath::ErrorCodes::OK);
	}
}
void Base256uMathTests::bit_shift_right::dst_n_zero() {
	// if dst_n is zero, then nothing happens and the truncated error code should be returned.

	std::size_t src = 1234567890;
	half_size_t dst = 987654321,
		answer = dst;
	int code;
	for (std::size_t by = 1; by < sizeof(std::size_t); by++) {
		code = Base256uMath::bit_shift_right(
			&src, sizeof(src),
			&by, sizeof(by),
			&dst, 0
		);
		assert(dst == answer);
		assert(code == Base256uMath::ErrorCodes::TRUNCATED);
	}
}
void Base256uMathTests::bit_shift_right::by_n_zero() {
	// if by_n is zero, then by is treated as 0, which means src is effectively copied into dst.

	std::size_t src = 246810;
	std::size_t dst = 1357911;
	std::size_t by = 25;
	auto code = Base256uMath::bit_shift_right(&src, sizeof(src), &by, 0, &dst, sizeof(dst));
	assert(dst == src);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bit_shift_right::in_place_ideal_case() {
	std::size_t num = 14687480300692146596,
		src, answer;
	int code;
	for (std::size_t by = 1; by < sizeof(src) * 8; by++) {
		src = num;
		answer = num >> by;
		code = Base256uMath::bit_shift_right(&src, sizeof(src), &by, sizeof(by));
		assert(src == answer);
		assert(code == Base256uMath::ErrorCodes::OK);
	}
}
void Base256uMathTests::bit_shift_right::in_place_big_ideal_case() {
	uint8_t src[] = { 215, 204, 79, 242, 35, 94, 138, 32, 147 };
	std::size_t by = 45;
	auto code = Base256uMath::bit_shift_right(src, sizeof(src), &by, sizeof(by));
	uint8_t answer[] = { 82, 4, 153, 4, 0, 0, 0, 0, 0 };
	for (uint8_t i = 0; i < sizeof(src); i++) {
		assert(src[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bit_shift_right::in_place_src_n_less_than_by() {
	// if src_n * 8 < by, then src is all zeros

	std::size_t src = 1823827429,
		answer = 0;
	std::size_t by = sizeof(src) * 8 + 1;
	auto code = Base256uMath::bit_shift_right(&src, sizeof(src), &by, sizeof(by));
	assert(src == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bit_shift_right::in_place_src_n_zero() {
	// if src_n is zero, then nothing happens

	unsigned int src = 2423423423,
		answer = src;
	std::size_t by = 15;
	auto code = Base256uMath::bit_shift_right(&src, 0, &by, sizeof(by));
	assert(src == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::bit_shift_right::in_place_by_n_zero() {
	// if by_n is zero, then nothing happens.

	unsigned int src = 246810,
		answer = src;
	std::size_t by = 25;
	auto code = Base256uMath::bit_shift_right(&src, sizeof(src), &by, 0);
	assert(src == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}

Base256uMathTests::byte_shift_left::byte_shift_left() {
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
void Base256uMathTests::byte_shift_left::ideal_case() {
	std::size_t src = 1305258424,
		dst, answer;
	int code;
	for (std::size_t by = 1; by < sizeof(std::size_t); by++) {
		assert(sizeof(src) > by && by > 0 && sizeof(src) == sizeof(dst));
		answer = src << (by * 8);
		code = Base256uMath::byte_shift_left(&src, sizeof(src), by, &dst, sizeof(dst));
		assert(dst == answer);
		assert(code == Base256uMath::ErrorCodes::OK);
	}
}
void Base256uMathTests::byte_shift_left::big_ideal_case() {
	uint8_t src[] = { 223, 192, 7, 188, 111, 229, 33, 55, 8 };
	std::size_t by = 3;
	uint8_t dst[9];
	assert(sizeof(src) > by && by > 0 && sizeof(src) == sizeof(dst));
	auto code = Base256uMath::byte_shift_left(src, sizeof(src), by, dst, sizeof(dst));
	uint8_t answer[] = { 0, 0, 0, 223, 192, 7, 188, 111, 229 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_left::src_n_less_than_by() {
	// means the result is all zeros

	half_size_t src = 1337;
	std::size_t by = sizeof(src) + 1;
	half_size_t dst = 394021884;
	assert(sizeof(src) == sizeof(dst) && sizeof(src) < by);
	auto code = Base256uMath::byte_shift_left(&src, sizeof(src), by, &dst, sizeof(dst));
	assert(dst == 0);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_left::src_n_greater_than_dst_n() {
	// if src_n > dst_n and by < dst_n, then the answer will be truncated

	unsigned int src = 39158;
	std::size_t by = 1;
	unsigned short dst;
	assert(sizeof(src) > sizeof(dst) && by < sizeof(dst));
	auto code = Base256uMath::byte_shift_left(&src, sizeof(src), by, &dst, sizeof(dst));
	decltype(dst) answer = src << (by * 8);
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::byte_shift_left::src_n_less_than_dst_n() {
	// if src_n < dst_n and by < src_n, then the answer will be copied into dst
	// and the rest of dst will be zero'd out.

	uint8_t src[] = { 244, 184, 73, 236, 228, 182, 41, 107, 81 };
	uint8_t dst[] = { 159, 188, 20, 222, 209, 85, 173, 112, 72, 73, 40, 123 };
	std::size_t by = 3;
	assert(sizeof(src) < sizeof(dst) && sizeof(src) > by);
	auto code = Base256uMath::byte_shift_left(src, sizeof(src), by, dst, sizeof(dst));
	uint8_t answer[] = { 0, 0, 0, 244, 184, 73, 236, 228, 182, 0, 0, 0 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_left::src_n_zero() {
	// if src_n is zero, then src is assumed to be all zeros and makes dst all zeros

	uint8_t src[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	auto code = Base256uMath::byte_shift_left(src, 0, 3, dst, sizeof(dst));
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == 0);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_left::dst_n_zero() {
	// if dst_n is zero, then nothing happens and returns the truncated error code.

	uint8_t src[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	auto code = Base256uMath::byte_shift_left(src, sizeof(src), 3, dst, 0);
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == i + 10);
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::byte_shift_left::by_is_zero() {
	// if by is zero, then it effectively copies src into dst.

	std::size_t src = 1334,
		dst = 39301;
	auto code = Base256uMath::byte_shift_left(&src, sizeof(src), 0, &dst, sizeof(dst));
	assert(dst == 1334);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_left::in_place_ideal_case() {
	std::size_t src = 258424;
	std::size_t by = 3;
	decltype(src) answer = src << by * 8;
	assert(sizeof(src) > by && by > 0);
	auto code = Base256uMath::byte_shift_left(&src, sizeof(src), by);
	assert(src == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_left::in_place_big_ideal_case() {
	uint8_t src[] = { 223, 192, 7, 188, 111, 229, 33, 55, 8 };
	std::size_t by = 3;
	assert(sizeof(src) > by && by > 0);
	auto code = Base256uMath::byte_shift_left(src, sizeof(src), by);
	uint8_t answer[] = { 0, 0, 0, 223, 192, 7, 188, 111, 229 };
	for (uint8_t i = 0; i < sizeof(src); i++) {
		assert(src[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_left::in_place_src_n_less_than_by() {
	// means the result is all zeros

	unsigned short src = 1337;
	std::size_t by = sizeof(src) + 1;
	assert(sizeof(src) < by);
	auto code = Base256uMath::byte_shift_left(&src, sizeof(src), by);
	assert(src == 0);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_left::in_place_src_n_zero() {
	// if src_n is zero, then nothing happens

	uint8_t src[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	auto code = Base256uMath::byte_shift_left(src, 0, 3);
	for (uint8_t i = 0; i < sizeof(src); i++) {
		assert(src[i] == i);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_left::in_place_by_is_zero() {
	// if by is zero, then nothing happens.
	unsigned int src = 133804;
	decltype(src) answer = src;
	auto code = Base256uMath::byte_shift_left(&src, sizeof(src), 0);
	assert(src == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}

Base256uMathTests::byte_shift_right::byte_shift_right() {
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
void Base256uMathTests::byte_shift_right::ideal_case() {
	std::size_t src = 13056761402769258424,
		dst, answer;
	int code;
	for (std::size_t by = 1; by * 8 < sizeof(std::size_t); by++) {
		assert(sizeof(src) > by && by > 0 && sizeof(src) == sizeof(dst));
		answer = src >> (by * 8);
		code = Base256uMath::byte_shift_right(&src, sizeof(src), by, &dst, sizeof(dst));
		assert(dst == answer);
		assert(code == Base256uMath::ErrorCodes::OK);
	}
}
void Base256uMathTests::byte_shift_right::big_ideal_case() {
	uint8_t src[] = { 223, 192, 7, 188, 111, 229, 33, 55, 8 };
	std::size_t by = 3;
	uint8_t dst[9];
	assert(sizeof(src) > by && by > 0 && sizeof(src) == sizeof(dst));
	auto code = Base256uMath::byte_shift_right(src, sizeof(src), by, dst, sizeof(dst));
	uint8_t answer[] = { 188, 111, 229, 33, 55, 8, 0, 0, 0 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_right::src_n_less_than_by() {
	// if src_n < by, then the result is all zeros

	unsigned short src = 1337;
	std::size_t by = sizeof(src) + 1;
	unsigned short dst = 394021884;
	assert(sizeof(src) == sizeof(dst) && sizeof(src) < by);
	auto code = Base256uMath::byte_shift_right(&src, sizeof(src), by, &dst, sizeof(dst));
	assert(dst == 0);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_right::src_n_greater_than_dst_n() {
	// if src_n > dst_n and by < dst_n, then the answer will be truncated

	std::size_t src = 16559212640052646418;
	std::size_t by = 1;
	half_size_t dst;
	assert(sizeof(src) > sizeof(dst) && by < sizeof(dst));
	auto code = Base256uMath::byte_shift_right(&src, sizeof(src), by, &dst, sizeof(dst));
	decltype(dst) answer = src >> (by * 8);
	assert(dst == answer);
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::byte_shift_right::src_n_less_than_dst_n() {
	// if src_n < dst_n and by < src_n, then the answer will be copied into dst
	// and the rest of dst will be zero'd out.

	uint8_t src[] = { 244, 184, 73, 236, 228, 182, 41, 107, 81 };
	uint8_t dst[] = { 159, 188, 20, 222, 209, 85, 173, 112, 72, 73, 40, 123 };
	std::size_t by = 3;
	assert(sizeof(src) < sizeof(dst) && sizeof(src) > by);
	auto code = Base256uMath::byte_shift_right(src, sizeof(src), by, dst, sizeof(dst));
	uint8_t answer[] = { 236, 228, 182, 41, 107, 81, 0, 0, 0, 0, 0, 0 };
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_right::src_n_zero() {
	// if src_n is zero, then src is assumed to be all zeros and makes dst all zeros.

	uint8_t src[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	auto code = Base256uMath::byte_shift_right(src, 0, 3, dst, sizeof(dst));
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == 0);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_right::dst_n_zero() {
	// if dst_n is zero, then nothing happens and the truncated error code is returned.

	uint8_t src[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	uint8_t dst[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
	auto code = Base256uMath::byte_shift_right(src, sizeof(src), 3, dst, 0);
	for (uint8_t i = 0; i < sizeof(dst); i++) {
		assert(dst[i] == i + 10);
	}
	assert(code == Base256uMath::ErrorCodes::TRUNCATED);
}
void Base256uMathTests::byte_shift_right::by_is_zero() {
	// if by is zero, then it effectively copies src into dst.

	unsigned int src = 133804,
		dst = 3939101;
	auto code = Base256uMath::byte_shift_right(&src, sizeof(src), 0, &dst, sizeof(dst));
	assert(dst == src);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_right::in_place_ideal_case() {
	std::size_t src = 13056761402769258424;
	std::size_t by = 3;
	decltype(src) answer = src >> by * 8;
	assert(sizeof(src) > by && by > 0);
	auto code = Base256uMath::byte_shift_right(&src, sizeof(src), by);
	assert(src == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_right::in_place_big_ideal_case() {
	uint8_t src[] = { 223, 192, 7, 188, 111, 229, 33, 55, 8 };
	std::size_t by = 3;
	assert(sizeof(src) > by && by > 0);
	auto code = Base256uMath::byte_shift_right(src, sizeof(src), by);
	uint8_t answer[] = { 188, 111, 229, 33, 55, 8, 0, 0, 0 };
	for (uint8_t i = 0; i < sizeof(src); i++) {
		assert(src[i] == answer[i]);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_right::in_place_src_n_less_than_by() {
	// means the result is all zeros

	unsigned short src = 1337;
	std::size_t by = sizeof(src) + 1;
	assert(sizeof(src) < by);
	auto code = Base256uMath::byte_shift_right(&src, sizeof(src), by);
	assert(src == 0);
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_right::in_place_src_n_zero() {
	// if src_n is zero, then nothing happens and returns the truncated error code.

	uint8_t src[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	auto code = Base256uMath::byte_shift_right(src, 0, 3);
	for (uint8_t i = 0; i < sizeof(src); i++) {
		assert(src[i] == i);
	}
	assert(code == Base256uMath::ErrorCodes::OK);
}
void Base256uMathTests::byte_shift_right::in_place_by_is_zero() {
	// if by is zero, then nothing happens.

	unsigned int src = 133804,
		answer = src;
	auto code = Base256uMath::byte_shift_right(&src, sizeof(src), 0);
	assert(src == answer);
	assert(code == Base256uMath::ErrorCodes::OK);
}
