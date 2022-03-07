/* UnitTests.h
Author: Grayson Spidle

Definitions for all my unit tests. I decided to write my own instead of using a framework,
because I didn't want to have to deal with the hassle of the framework not working in other
environments.

I tried to be comprehensive in my testing, but writing unit tests is boring so I might miss a few things.
*/

#ifndef __BASE256UMATH_UNIT_TESTS_H__
#define __BASE256UMATH_UNIT_TESTS_H__

/* TODO
max:
	left_n_zero()
	right_n_zero()
min
	left_n_zero()
	right_n_zero()
*/

namespace Base256uMathTests {
	// tests all of them
	void test_unit_tests();

	struct is_zero {
		// Given any sized number, tells you if all bytes are 0.
		// Returns true if all bytes are 0.
		// Returns false if not all bytes are 0.

		is_zero();

		void ideal_case();
		void big_ideal_case();

		void not_zero();
		void big_not_zero();

		void src_n_zero();
	};
	struct compare {
		// Given any sized number, the function should return:
		//	1 : if left is greater than right.
		//	0  : if left is equal to right.
		//	-1 : if left is less than right.
		// If the numbers are of different sizes, then we treat the smallest
		// sized one as if we added bytes with zeros to the front in an attempt
		// to make it the same size.

		compare(); // tests everything

		void ideal_case_equal();
		void ideal_case_greater();
		void ideal_case_less();

		void big_ideal_case_equal();
		void big_ideal_case_greater();
		void big_ideal_case_less();

		void l_bigger_equal();
		void l_smaller_equal();
		void big_l_bigger_equal();
		void big_l_smaller_equal();

		void l_bigger_greater();
		void l_smaller_greater();
		void big_l_bigger_greater();
		void big_l_smaller_greater();

		void l_bigger_less();
		void l_smaller_less();
		void big_l_bigger_less();
		void big_l_smaller_less();

		void left_n_zero();
		void right_n_zero();
		
	};
	struct max {
		// Given any sized number, return the pointer to the larger one.

		max();

		void ideal_case_left();
		void ideal_case_right();
		void big_ideal_case_left();
		void big_ideal_case_right();

		void left_bigger_left();
		void left_smaller_left();
		void left_bigger_right();
		void left_smaller_right();

		void big_left_bigger_left();
		void big_left_smaller_left();
		void big_left_bigger_right();
		void big_left_smaller_right();
	};
	struct min {
		min();

		void ideal_case_left();
		void ideal_case_right();
		void big_ideal_case_left();
		void big_ideal_case_right();

		void left_bigger_left();
		void left_smaller_left();
		void left_bigger_right();
		void left_smaller_right();

		void big_left_bigger_left();
		void big_left_smaller_left();
		void big_left_bigger_right();
		void big_left_smaller_right();
	};
	struct increment {
		increment();

		void ideal_case();
		void big_ideal_case();
		void overflow();
		void big_overflow();
	};
	struct decrement {
		decrement();

		void ideal_case();
		void big_ideal_case();
		void underflow();
		void big_underflow();
	};
	struct add {
		add();

		void ideal_case();
		void big_ideal_case();
		void left_bigger();
		void left_smaller();
		void big_left_bigger();
		void big_left_smaller();
		void overflow();
		void big_overflow();
		void dst_too_small();
		void big_dst_too_small();
		void zero_for_left_n();
		void zero_for_right_n();
		void zero_for_dst_n();

		void in_place_ideal_case();
		void in_place_big_ideal_case();
		void in_place_left_bigger();
		void in_place_left_smaller();
		void in_place_big_left_bigger();
		void in_place_big_left_smaller();
		void in_place_overflow();
		void in_place_big_overflow();
		void in_place_zero_for_left_n();
		void in_place_zero_for_right_n();
	};
	struct subtract {
		subtract();

		void ideal_case();
		void big_ideal_case();
		void left_bigger();
		void left_smaller();
		void big_left_bigger();
		void big_left_smaller();
		void underflow();
		void big_underflow();
		void dst_too_small();
		void big_dst_too_small();
		void zero_for_left_n();
		void zero_for_right_n();
		void zero_for_dst_n();

		void in_place_ideal_case();
		void in_place_big_ideal_case();
		void in_place_left_bigger();
		void in_place_left_smaller();
		void in_place_big_left_bigger();
		void in_place_big_left_smaller();
		void in_place_underflow();
		void in_place_big_underflow();
		void in_place_zero_for_left_n();
		void in_place_zero_for_right_n();
	};
	struct multiply {
		multiply();

		void ideal_case();
		void big_ideal_case();
		void multiply_zero();
		void multiply_one();
		void left_n_greater_than_right_n();
		void left_n_less_than_right_n();
		void dst_n_less_than_both();
		void dst_n_zero();

		void in_place_ideal_case();
		void in_place_big_ideal_case();
		void in_place_multiply_zero();
		void in_place_multiply_one();
		void in_place_left_n_greater_than_right_n();
		void in_place_left_n_less_than_right_n();
	};
	struct divide {
		divide();

		void ideal_case();
		void big_ideal_case();
		void left_is_zero();
		void left_n_zero();
		void right_is_zero();
		void right_n_zero();
		void left_n_less();
		void dst_n_less();
		void dst_n_zero();
		void remainder_n_less();
		void remainder_n_zero();

		void in_place_ideal_case();
		void in_place_big_ideal_case();
		void in_place_left_is_zero();
		void in_place_left_n_zero();
		void in_place_right_is_zero();
		void in_place_right_n_zero();
		void in_place_left_n_less();
		void in_place_remainder_n_less();
		void in_place_remainder_n_zero();
	};
	struct divide_no_mod {
		divide_no_mod();

		void ideal_case();
		void big_ideal_case();
		void left_is_zero();
		void left_n_zero();
		void right_is_zero();
		void right_n_zero();
		void left_n_less();
		void dst_n_less();
		void dst_n_zero();

		void in_place_ideal_case();
		void in_place_big_ideal_case();
		void in_place_left_is_zero();
		void in_place_left_n_zero();
		void in_place_right_is_zero();
		void in_place_right_n_zero();
		void in_place_left_n_less();
	};
	struct mod {
		mod();

		void ideal_case();
		void big_ideal_case();
		void left_is_zero();
		void left_n_zero();
		void right_is_zero();
		void right_n_zero();
		void left_n_less();
		void dst_n_less();
		void dst_n_zero();

		void in_place_ideal_case();
		void in_place_big_ideal_case();
		void in_place_left_is_zero();
		void in_place_left_n_zero();
		void in_place_right_is_zero();
		void in_place_right_n_zero();
		void in_place_left_n_less();
	};
	struct log2 {
		log2();

		void ideal_case();
		void big_ideal_case();
		void src_is_zero();
		void src_n_zero();
		void dst_n_zero();
	};
	struct log256 {
		log256();
		
		void ideal_case();
		void big_ideal_case();
		void src_is_zero();
		void src_n_zero();
	};
	struct bitwise_and {
		bitwise_and();

		void ideal_case();
		void big_ideal_case();
		void left_bigger();
		void left_smaller();
		void big_left_bigger();
		void big_left_smaller();
		void dst_too_small();
		void big_dst_too_small();
		void left_n_zero();
		void right_n_zero();
		void dst_n_zero();

		void in_place_ideal_case();
		void in_place_big_ideal_case();
		void in_place_left_bigger();
		void in_place_left_smaller();
		void in_place_big_left_bigger();
		void in_place_big_left_smaller();
		void in_place_left_n_zero();
		void in_place_right_n_zero();
	};
	struct bitwise_or {
		bitwise_or();
		void ideal_case();
		void big_ideal_case();
		void left_bigger();
		void left_smaller();
		void big_left_bigger();
		void big_left_smaller();
		void dst_too_small();
		void big_dst_too_small();
		void left_n_zero();
		void right_n_zero();
		void dst_n_zero();

		void in_place_ideal_case();
		void in_place_big_ideal_case();
		void in_place_left_bigger();
		void in_place_left_smaller();
		void in_place_big_left_bigger();
		void in_place_big_left_smaller();
		void in_place_left_n_zero();
		void in_place_right_n_zero();
	};
	struct bitwise_xor {
		bitwise_xor();
		void ideal_case();
		void big_ideal_case();
		void left_bigger();
		void left_smaller();
		void big_left_bigger();
		void big_left_smaller();
		void dst_too_small();
		void big_dst_too_small();
		void left_n_zero();
		void right_n_zero();
		void dst_n_zero();

		void in_place_ideal_case();
		void in_place_big_ideal_case();
		void in_place_left_bigger();
		void in_place_left_smaller();
		void in_place_big_left_bigger();
		void in_place_big_left_smaller();
		void in_place_left_n_zero();
		void in_place_right_n_zero();
	};
	struct bitwise_not {
		bitwise_not();

		void ideal_case();
		void big_ideal_case();
		void src_n_zero();
	};
	struct bit_shift_left {
		bit_shift_left();
		void ideal_case();
		void big_ideal_case();
		void src_n_less_than_by();
		void src_n_greater_than_dst_n();
		void src_n_less_than_dst_n();
		void src_n_zero();
		void dst_n_zero();
		void by_n_zero();

		void in_place_ideal_case();
		void in_place_big_ideal_case();
		void in_place_src_n_less_than_by();
		void in_place_src_n_zero();
		void in_place_by_n_zero();
	};
	struct bit_shift_right {
		bit_shift_right();
		void ideal_case();
		void big_ideal_case();
		void src_n_less_than_by();
		void src_n_greater_than_dst_n();
		void src_n_less_than_dst_n();
		void src_n_zero();
		void dst_n_zero();
		void by_n_zero();

		void in_place_ideal_case();
		void in_place_big_ideal_case();
		void in_place_src_n_less_than_by();
		void in_place_src_n_zero();
		void in_place_by_n_zero();
	};
	struct byte_shift_left {
		byte_shift_left();

		void ideal_case();
		void big_ideal_case();
		void src_n_less_than_by();
		void src_n_greater_than_dst_n();
		void src_n_less_than_dst_n();
		void src_n_zero();
		void dst_n_zero();
		void by_is_zero();

		void in_place_ideal_case();
		void in_place_big_ideal_case();
		void in_place_src_n_less_than_by();
		void in_place_src_n_zero();
		void in_place_by_is_zero();
	};
	struct byte_shift_right {
		byte_shift_right();

		void ideal_case();
		void big_ideal_case();
		void src_n_less_than_by();
		void src_n_greater_than_dst_n();
		void src_n_less_than_dst_n();
		void src_n_zero();
		void dst_n_zero();
		void by_is_zero();

		void in_place_ideal_case();
		void in_place_big_ideal_case();
		void in_place_src_n_less_than_by();
		void in_place_src_n_zero();
		void in_place_by_is_zero();
	};
}

#endif // __BASE256UMATH_UNIT_TESTS_H__