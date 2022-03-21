# Base256uMath
Math library for big unsigned numbers.

I wrote this repo for another project and decided to share it with the world.
I intend to keep this repo licensed with the unlicense license for the foreseeable future.
I don't claim to be the greatest mathematician or the greatest computer scientist, but this gets the job done.
I've had a lot of fun writing it and I've learned a lot on my journey.

## Capabilities

This library covers all the basic operators a number should need:
- add
- subtract
- multiply
- divide
- modulo
- increment
- decrement
- compare
- is zero (essentially the opposite of the `bool()` operator)
- bit shifting
- bitwise and
- bitwise or
- bitwise xor
- bitwise not

And also some extra functions:
- log2
- log256
- min
- max

### CUDA
This library was designed to be compatible with CUDA. What that means is the functions are designed (and tested) to work on NVIDIA CUDA enabled GPUs.
I don't have an array of GPUs to test this on (I use a `Quadro K2200` for testing), so if there is a problem on newer GPUs then I will not be aware of it.

## Targets
- `C++14` language standard
- Big Endian machine
- 64 bit machines (until I can confirm that it works on 32 bit).
- msvc and gcc compiler
- cuda support (although I would only qualify it as "barely" right now)

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
Here is a short tutorial on how to use this. In case my future self is too lazy to figure it out himself.
I urge you to read the documentation (I know, reading sucks) for more information.

```c++
#include "Base256uMath.h" // include the header file like any other
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
  // An enumeration of error codes can be found at Base256uMath::ErrorCodes.
  // Function documentation will also tell you what codes the function can return.
  
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
  
  // Error checking is what you would expect. Fatal errors are < 0 and warnings are > 0
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

This library is by no means dummy proof, and some optimizations that a compiler might do will probably need to be done by you.
For example, if you are multiplying by 2, instead of using the `multiply()` function use `bit_shift_left()`. It accomplishes the same thing but a lot faster.

## What I'm Working On Now
- optimize multiply
- optimize divide
- optimize modulo
- nvcc (CUDA) compatibility
- 32 bit compatibility (mostly just verifying it passes tests)
- polishing clarity in documentation

## Some nice things I'd like to have happen
- nvcc kernel variants of functions
