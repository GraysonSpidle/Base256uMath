#include "PerformanceTests.h"
#include "Base256uMath.h"
#include <chrono>
#include <vector>
#include <cstring>
#include <iostream>

struct Timer {
	std::chrono::time_point<std::chrono::high_resolution_clock> _start;
	std::vector<long long>& vec;

	Timer(std::vector<long long>& vec) : vec(vec) {}

	void start() {
		_start = std::chrono::high_resolution_clock::now();
	}

	void stop() {
		auto end = std::chrono::high_resolution_clock::now();
		auto n = std::chrono::time_point_cast<std::chrono::microseconds>(end).time_since_epoch().count() -
			std::chrono::time_point_cast<std::chrono::microseconds>(_start).time_since_epoch().count();
		vec.push_back(n);
	}
};

inline double calculate_average(std::vector<long long>& vec) {
	double average = 0;
	for (auto it = vec.begin(); it != vec.end(); ++it) {
		average += *it;
	}
	if (vec.size())
		average /= vec.size();
	return average;
}

void Base256uMathTests::Performance::test(std::size_t sample_size, std::size_t size) {
	std::cout << "All times are in microseconds" << std::endl;
	std::cout << "All tests are engineered to be worst case scenario" << std::endl;
	std::cout << "number(s) size: " << size << " bytes" << std::endl;
	std::cout << "sample size: " << sample_size << " iterations" << std::endl << std::endl;
	
	Base256uMathTests::Performance::is_zero(sample_size, size);
	Base256uMathTests::Performance::compare(sample_size, size);
	Base256uMathTests::Performance::max(sample_size, size);
	Base256uMathTests::Performance::min(sample_size, size);
	Base256uMathTests::Performance::bitwise_and(sample_size, size);	
	Base256uMathTests::Performance::bitwise_or(sample_size, size);	
	Base256uMathTests::Performance::bitwise_xor(sample_size, size);	
	Base256uMathTests::Performance::bitwise_not(sample_size, size);
	Base256uMathTests::Performance::byte_shift_left(sample_size, size);	
	Base256uMathTests::Performance::byte_shift_right(sample_size, size);
	Base256uMathTests::Performance::bit_shift_left(sample_size, size);
	Base256uMathTests::Performance::bit_shift_right(sample_size, size);	
	Base256uMathTests::Performance::increment(sample_size, size);	
	Base256uMathTests::Performance::decrement(sample_size, size);
	Base256uMathTests::Performance::add(sample_size, size);
	Base256uMathTests::Performance::subtract(sample_size, size);
	Base256uMathTests::Performance::log2(sample_size, size);
	Base256uMathTests::Performance::log256(sample_size, size);	
	Base256uMathTests::Performance::multiply(sample_size, size);
	Base256uMathTests::Performance::divide(sample_size, size);
	Base256uMathTests::Performance::divide_no_mod(sample_size, size);
	Base256uMathTests::Performance::mod(sample_size, size);
}

/*
void Base256uMathTests::Performance::test_obj() {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* block = malloc(13);
	if (!block)
		abort();
	void* block2 = malloc(13);
	if (!block2) {
		free(block);
		abort();
	}

	Base256uint num0 = { block, 13 };
	Base256uint num1 = { block2, 13 };

	for (std::size_t i = 0; i < 100; i++) {
		timer.start();
		num0 + num1;
		timer.stop();
	}

	double average = calculate_average(vec);
	std::cout << "obj: " << std::to_string(average) << std::endl;

	free(block);
	free(block2);
}*/

inline void print_progress(const std::size_t& i, const std::size_t& sample_size) {
	std::cout << int(((float)i / (float)sample_size) * 100.0) << "%\r";
	std::cout.flush();
}

void Base256uMathTests::Performance::is_zero(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	if (!num)
		abort();
	memset(num, 0, size);
	reinterpret_cast<unsigned char*>(num)[size - 1] = 128;
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::is_zero(num, size);
		timer.stop();
		std::cout << "is_zero(): ";
		print_progress(i, sample_size);
	}
	free(num);
	double average = calculate_average(vec);
	std::cout << "is_zero(): " << std::to_string(average) << std::endl;
}

void Base256uMathTests::Performance::compare(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	void* num2 = malloc(size);
	if (!num || !num2)
		abort();
	memset(num, 255, size);
	memset(num2, 255, size);
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::compare(num, size, num2, size);
		timer.stop();
		std::cout << "compare(): ";
		print_progress(i, sample_size);
	}
	free(num);
	free(num2);
	double average = calculate_average(vec);
	std::cout << "compare(): " << std::to_string(average) << std::endl;
}

void Base256uMathTests::Performance::max(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	void* num2 = malloc(size);
	if (!num || !num2)
		abort();
	memset(num, 255, size);
	memset(num2, 255, size);
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::max(num, size, num2, size);
		timer.stop();
		std::cout << "max(): ";
		print_progress(i, sample_size);
	}
	free(num);
	free(num2);
	double average = calculate_average(vec);
	std::cout << "max(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::min(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	void* num2 = malloc(size);
	if (!num || !num2)
		abort();
	memset(num, 255, size);
	memset(num2, 255, size);
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::min(num, size, num2, size);
		timer.stop();
		std::cout << "min(): ";
		print_progress(i, sample_size);
	}
	free(num);
	free(num2);
	double average = calculate_average(vec);
	std::cout << "min(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::bitwise_and(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	void* num2 = malloc(size);
	if (!num || !num2)
		abort();
	memset(num, 255, size);
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::bitwise_and(num, size, num, size, num2, size);
		timer.stop();
		std::cout << "bitwise_and(): ";
		print_progress(i, sample_size);
	}
	free(num);
	free(num2);
	double average = calculate_average(vec);
	std::cout << "bitwise_and(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::bitwise_or(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	void* num2 = malloc(size);
	if (!num || !num2)
		abort();
	memset(num, 255, size);
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::bitwise_or(num, size, num, size, num2, size);
		timer.stop();
		std::cout << "bitwise_or(): ";
		print_progress(i, sample_size);
	}
	free(num);
	free(num2);
	double average = calculate_average(vec);
	std::cout << "bitwise_or(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::bitwise_xor(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	void* num2 = malloc(size);
	if (!num || !num2)
		abort();
	memset(num, 255, size);
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::bitwise_xor(num, size, num, size, num2, size);
		timer.stop();
		std::cout << "bitwise_xor(): ";
		print_progress(i, sample_size);
	}
	free(num);
	free(num2);
	double average = calculate_average(vec);
	std::cout << "bitwise_xor(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::bitwise_not(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	if (!num)
		abort();
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::bitwise_not(num, size);
		timer.stop();
		std::cout << "bitwise_not(): ";
		print_progress(i, sample_size);
	}
	free(num);
	double average = calculate_average(vec);
	std::cout << "bitwise_not(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::byte_shift_left(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
		std::size_t by = size - 1;
	void* num = malloc(size);
	if (!num)
		abort();
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::byte_shift_left(num, size, by);
		timer.stop();
		std::cout << "byte_shift_left(): ";
		print_progress(i, sample_size);
	}
	free(num);
	double average = calculate_average(vec);
	std::cout << "byte_shift_left(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::byte_shift_right(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
		std::size_t by = size - 1;
	void* num = malloc(size);
	if (!num)
		abort();
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::byte_shift_right(num, size, by);
		timer.stop();
		std::cout << "byte_shift_right(): ";
		print_progress(i, sample_size);
	}
	free(num);
	double average = calculate_average(vec);
	std::cout << "byte_shift_right(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::bit_shift_left(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
		std::size_t by = (size << 3) - 1;
	void* num = malloc(size);
	if (!num)
		abort();
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::bit_shift_left(num, size, by);
		timer.stop();
		std::cout << "bit_shift_left(): ";
		print_progress(i, sample_size);
	}
	free(num);
	double average = calculate_average(vec);
	std::cout << "bit_shift_left(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::bit_shift_right(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
		std::size_t by = (size << 3) - 1;
	void* num = malloc(size);
	if (!num)
		abort();
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::bit_shift_right(num, size, by);
		timer.stop();
		std::cout << "bit_shift_right(): ";
		print_progress(i, sample_size);
	}
	free(num);
	double average = calculate_average(vec);
	std::cout << "bit_shift_right(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::increment(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	if (!num)
		abort();
	for (std::size_t i = 0; i < sample_size; i++) {
		memset(num, 255, size);
		timer.start();
		Base256uMath::increment(num, size);
		timer.stop();
		std::cout << "increment(): ";
		print_progress(i, sample_size);
	}
	free(num);
	double average = calculate_average(vec);
	std::cout << "increment(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::decrement(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	if (!num)
		abort();
	for (std::size_t i = 0; i < sample_size; i++) {
		memset(num, 0, size);
		timer.start();
		Base256uMath::decrement(num, size);
		timer.stop();
		std::cout << "decrement(): ";
		print_progress(i, sample_size);
	}
	free(num);
	double average = calculate_average(vec);
	std::cout << "decrement(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::add(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	void* dst = malloc(size);
	if (!num || !dst)
		abort();
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::add(num, size, num, size, dst, size);
		timer.stop();
		std::cout << "add(): ";
		print_progress(i, sample_size);
	}
	free(num);
	free(dst);
	double average = calculate_average(vec);
	std::cout << "add(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::subtract(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	void* dst = malloc(size);
	if (!num || !dst)
		abort();
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::subtract(num, size, num, size, dst, size);
		timer.stop();
		std::cout << "subtract(): ";
		print_progress(i, sample_size);
	}
	free(num);
	free(dst);
	double average = calculate_average(vec);
	std::cout << "subtract(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::log2(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	if (!num)
		abort();
	Base256uMath::bit_size_t dst;
	memset(num, 0, size);
	reinterpret_cast<unsigned char*>(num)[0] = 1;
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::log2(num, size, dst, sizeof(dst));
		timer.stop();
		std::cout << "log2(): ";
		print_progress(i, sample_size);
	}
	free(num);
	double average = calculate_average(vec);
	std::cout << "log2(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::log256(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	if (!num)
		abort();
	std::size_t dst;
	memset(num, 0, size);
	reinterpret_cast<unsigned char*>(num)[0] = 1;
	for (std::size_t i = 0; i < sample_size; i++) {
		timer.start();
		Base256uMath::log256(num, size, &dst);
		timer.stop();
		std::cout << "log256(): ";
		print_progress(i, sample_size);
	}
	free(num);
	double average = calculate_average(vec);
	std::cout << "log256(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::multiply(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	std::size_t product_size = size + size;
	void* num = malloc(size);
	void* num2 = malloc(size);
	void* dst = malloc(product_size);
	if (!num || !num2 || !dst)
		abort();
	memset(num, 255, size);
	memset(num2, 255, size);
	for (std::size_t i = 0; i < sample_size; i++) {
		std::cout << "multiply(): ";
		print_progress(i, sample_size);
		timer.start();
		auto code = Base256uMath::multiply(num, size, num2, size, dst, product_size);
		timer.stop();
		if (code == Base256uMath::ErrorCodes::OOM) {
			std::cout << "OOM error encountered" << std::endl;
			abort();
		}
	}
	free(num);
	free(num2);
	free(dst);
	double average = calculate_average(vec);
	std::cout << "multiply(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::divide(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	void* divisor = malloc(size);
	void* dst = malloc(size);
	void* remainder = malloc(size);
	if (!num || !dst || !remainder || !divisor)
		abort();
	memset(divisor, 0, size);
	reinterpret_cast<unsigned char*>(divisor)[0] = 1;
	for (std::size_t i = 0; i < sample_size; i++) {
		std::cout << "divide(): ";
		print_progress(i, sample_size);
		timer.start();
		auto code = Base256uMath::divide(num, size, divisor, size, dst, size, remainder, size);
		timer.stop();
		if (code == Base256uMath::ErrorCodes::OOM) {
			std::cout << "OOM error encountered" << std::endl;
			abort();
		}
	}
	free(num);
	free(dst);
	free(remainder);
	free(divisor);
	double average = calculate_average(vec);
	std::cout << "divide(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::divide_no_mod(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	void* divisor = malloc(size);
	void* dst = malloc(size);
	if (!num || !dst || !divisor)
		abort();
	memset(num, 255, size);
	memset(divisor, 0, size);
	reinterpret_cast<unsigned char*>(divisor)[0] = 1;
	for (std::size_t i = 0; i < sample_size; i++) {
		std::cout << "divide_no_mod(): ";
		print_progress(i, sample_size);
		timer.start();
		auto code = Base256uMath::divide_no_mod(num, size, divisor, size, dst, size);
		timer.stop();
		if (code == Base256uMath::ErrorCodes::OOM) {
			std::cout << "OOM error encountered" << std::endl;
			abort();
		}
	}
	free(num);
	free(dst);
	free(divisor);
	double average = calculate_average(vec);
	std::cout << "divide_no_mod(): " << std::to_string(average) << std::endl;
}
void Base256uMathTests::Performance::mod(const std::size_t& sample_size, const std::size_t& size) {
	std::vector<long long> vec;
	Timer timer = { vec };
	void* num = malloc(size);
	void* divisor = malloc(size);
	void* dst = malloc(size);
	if (!num || !dst || !divisor)
		abort();
	memset(num, 255, size);
	memset(divisor, 0, size);
	reinterpret_cast<unsigned char*>(divisor)[0] = 1;
	for (std::size_t i = 0; i < sample_size; i++) {
		std::cout << "mod(): ";
		print_progress(i, sample_size);
		timer.start();
		auto code = Base256uMath::mod(num, size, divisor, size, dst, size);
		timer.stop();
		if (code == Base256uMath::ErrorCodes::OOM) {
			std::cout << "OOM error encountered" << std::endl;
			abort();
		}
	}
	free(num);
	free(dst);
	free(divisor);
	double average = calculate_average(vec);
	std::cout << "mod(): " << std::to_string(average) << std::endl;
}
