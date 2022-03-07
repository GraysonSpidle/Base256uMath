#ifndef __BASE256UMATH_CUDA_UNIT_TESTS_H__
#define __BASE256UMATH_CUDA_UNIT_TESTS_H__

namespace Base256uMathTests {
	namespace CUDA {
		// Tests all of them
		void test_unit_tests();

		namespace is_zero {
			// Given any sized number, tells you if all bytes are 0.
			// Returns true if all bytes are 0.
			// Returns false if not all bytes are 0.

			void test();

			void ideal_case();
			void big_ideal_case();
			void not_zero();
			void big_not_zero();
			void src_n_zero();
		};
		namespace compare {
			// Given any sized number, the function should return:
			//	1 : if left is greater than right.
			//	0  : if left is equal to right.
			//	-1 : if left is less than right.
			// If the numbers are of different sizes, then we treat the smallest
			// sized one as if we added bytes with zeros to the front in an attempt
			// to make it the same size.

			void test();

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
		namespace max {
			// Given any sized number, return the pointer to the larger one.

			void test();		

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
		namespace min {
			void test();

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
		namespace increment {
			void test();

			void ideal_case();
			void big_ideal_case();
			void overflow();
			void big_overflow();
		};
		namespace decrement {
			void test();

			void ideal_case();
			void big_ideal_case();
			void underflow();
			void big_underflow();
		};
		namespace add {
			void test();

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
		namespace subtract {
			void test();

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
		namespace multiply {
			void test();

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
		namespace divide {
			void test();

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
		namespace divide_no_mod {
			void test();

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
		namespace mod {
			void test();

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
		namespace log2 {
			void test();

			void ideal_case();
			void big_ideal_case();
			void src_is_zero();
			void src_n_zero();
			void dst_n_zero();
		};
		namespace log256 {
			void test();

			void ideal_case();
			void big_ideal_case();
			void src_is_zero();
			void src_n_zero();
		};
		namespace bitwise_and {
			void test();			

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
		namespace bitwise_or {
			void test();
			
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
		namespace bitwise_xor {
			void test();
			
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
		namespace bitwise_not {
			void test();
			
			void ideal_case();
			void big_ideal_case();
			void src_n_zero();
		};
		namespace bit_shift_left {
			void test();

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
		namespace bit_shift_right {
			void test();
			
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
		namespace byte_shift_left {
			void test();

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
		namespace byte_shift_right {
			void test();
			
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
	};



};
#endif // __BASE256UMATH_CUDA_UNIT_TESTS_H__