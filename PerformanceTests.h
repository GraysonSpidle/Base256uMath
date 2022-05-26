#ifndef __BASE256UMATH_PERFORMANCE_TESTS_H__
#define __BASE256UMATH_PERFORMANCE_TESTS_H__
#include <cstdlib>

namespace Base256uMathTests {
	namespace Performance {
		void test(std::size_t sample_size, std::size_t size);
		void test_obj();

		void is_zero(const std::size_t& sample_size, const std::size_t& size);
		void compare(const std::size_t& sample_size, const std::size_t& size);
		void max(const std::size_t& sample_size, const std::size_t& size);
		void min(const std::size_t& sample_size, const std::size_t& size);
		void bitwise_and(const std::size_t& sample_size, const std::size_t& size);
		void bitwise_or(const std::size_t& sample_size, const std::size_t& size);
		void bitwise_xor(const std::size_t& sample_size, const std::size_t& size);
		void bitwise_not(const std::size_t& sample_size, const std::size_t& size);
		void byte_shift_left(const std::size_t& sample_size, const std::size_t& size);
		void byte_shift_right(const std::size_t& sample_size, const std::size_t& size);
		void bit_shift_left(const std::size_t& sample_size, const std::size_t& size);
		void bit_shift_right(const std::size_t& sample_size, const std::size_t& size);
		void increment(const std::size_t& sample_size, const std::size_t& size);
		void decrement(const std::size_t& sample_size, const std::size_t& size);
		void add(const std::size_t& sample_size, const std::size_t& size);
		void subtract(const std::size_t& sample_size, const std::size_t& size);
		void log2(const std::size_t& sample_size, const std::size_t& size);
		void log256(const std::size_t& sample_size, const std::size_t& size);
		void multiply(const std::size_t& sample_size, const std::size_t& size);
		void divide(const std::size_t& sample_size, const std::size_t& size);
		void divide_no_mod(const std::size_t& sample_size, const std::size_t& size);
		void mod(const std::size_t& sample_size, const std::size_t& size);
	}
}
#endif // __BASE256UMATH_PERFORMANCE_TESTS_H__