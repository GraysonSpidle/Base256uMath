# Base256uMath

I wrote this repo for another project.
I wanted a library that could manipulate large unsigned numbers with the capability of:
- add
- subtract
- multiply
- divide
- modulo
- log2

I also wanted the library to be trivially convertible for CUDA code, which had a laundry list of things I could not use.

I was dissatisfied with the choices I found, so I made this. I don't claim to be the greatest mathematician or computer scientist, but this gets the job done.

I've had a lot of fun writing it and I learned a lot on my journey.

## Targets
- `C++14` language standard
- Big Endian machine
- 64 bit machines (until I can confirm that it works on 32 bit).
- msvc and gcc compiler (nvcc support is coming)

## Installation
Your typical installation should be fairly simple. Drag and drop the header and source files into their appropriate directories in your project.

For the core of the library you'll want:
- `Base256uMath.h`
- `Base256uMath.cpp`

The unit tests are there just for you to test to see if the library works as intended in your environment.

### For CUDA
I'll admit, I don't know much about nvcc so don't come here for a tutorial. However, this is what I had to do to get the unit tests compiled:

- First change the `.cpp` files to `.cu` files. Otherwise it won't recognize stuff like `__global__`.
- Then do these commands (`sm_50` is the architecture of my GPU, change it to yours):
```
nvcc -arch sm_50 -m 64 --device-c Tests.cu Base256uMath.cu UnitTests.cu CUDAUnitTests.cu
nvcc -arch sm_50 -m 64 -o Tests Tests.o Base256uMath.o UnitTests.o CUDAUnitTests.o
```
- It'll have a lot of warnings, but you are probably used to that kind of stuff now :P
- Run with `./Tests`

## Tutorial
I'm gonna write a short tutorial on how to use this. In case my future self is too lazy to figure it out himself.
I have written a more in depth explanation in the `Base256uMath.h` header file.

```c++
#include "Base256uMath.h"
#include <iostream>

int main() {
  // You can initialize numbers like this (remember, big endian)
  unsigned char number[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  unsigned char another[] = { 55, 21, 23 };

  // or you can initialize numbers like regular memory blocks, just don't forget to delete them afterwards
  void* dst = malloc(10);

  // Each number is interpreted as an array of bytes and does arithmetic on them in base 256.
  // For every function, you must supply the pointer to the uchar array and the size of it.
  auto code = Base256uMath::add(number, sizeof(number), another, sizeof(another), dst, 10);

  // Most functions return an error code. None of them throw any exceptions.
  // An enumeration of error codes can be found at Base256uMath::ErrorCodes
  
  switch (code) {
  case Base256uMath::ErrorCodes::OK:
    std::cout << "Everything went okay!" << std::endl;
    break;
  case Base256uMath::ErrorCodes::FLOW:
    std::cout << "A non fatal error warning you that an overflow happened." << std::endl;
    break;
  case Base256uMath::ErrorCodes::TRUNCATED:
    std::cout << "A non fatal error warning you that the result of the operation *may* be truncated." << std::endl;
    break;
  }
  
  // You can also do arithmetic on regular numbers, but it will be much slower than normal arithmetic
  unsigned int number2 = 2;
  unsigned short another2 = 2;
  
  // When using primitive numbers, you need to get their pointers
  code = Base256uMath::subtract(&number2, sizeof(number2), &another2, sizeof(another2), dst, 10);
  
  // You can even mix up the usage of primitives and big numbers
  code = Base256uMath::multiply(number, sizeof(number), &another2, sizeof(another2), dst, 10);
  
  if (code == Base256uMath::ErrorCodes::OOM) {
    std::cout << "Oh no! We ran out of memory!" << std::endl;
  }
  else if (code < Base256uMath::ErrorCodes::OK) {
    std::cout << "Something bad happened" << std::endl;
  }
  
  free(dst);
  return 0;
}
```

## What I'm Working On Now
- in-place multiply
- in-place divide
- in-place modulo
- nvcc compatibility
- 32 bit compatibility
- polishing clarity in documentation

## Some nice things I'd like to have happen
- optimize divide
- optimize mod
- nvcc kernel variants of functions
