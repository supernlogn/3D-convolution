# 3D Convolution

## Introduction
Could not find a lightweight, practical implementation of this, so this project was built to fill that gap.
This library performs n-dimensional convolution in C/C++ with no heavy frameworks and no custom data structures.
It uses the Fourier transform to deliver strong performance while staying easy to follow and integrate.
The only core dependency is **FFTW3**.
Because internal buffers are allocated during execution, a standard memory allocation library (such as `stdlib.h`) is required.

## Requirements
The project uses FFTW3.

- If FFTW3 is already installed, CMake will use the system package.
- If FFTW3 is not installed, CMake can download and build FFTW3 automatically.

## How to build
Debian-oids:
1. Optional: install FFTW3 with ```sudo apt-get install libfftw3-dev``` if you want to use the system package.
2. Download this library. Just type: ```git clone https://github.com/supernlogn/3D-convolution.git```
3. Type on a terminal ```cmake -S 3D-convolution -B 3D-convolution/build```
4. Then ```cmake --build 3D-convolution/build```
5. Run tests with ```ctest --test-dir 3D-convolution/build --output-on-failure```

## Examples

The library includes example programs demonstrating its usage:

- **Convolution_example**: Basic usage of the `convnd()` function showing how to allocate tensors, perform convolution, and inspect results.
  ```bash
  ./build/Convolution_example
  ```

- **Convolution_cnn_inference**: Demonstrates a complete CNN-style inference pipeline combining convolution, bias addition, and ReLU activation with layer-by-layer statistics tracking.
  ```bash
  ./build/Convolution_cnn_inference
  ```

Both examples use automatically initialized test data and display tensor statistics (min, max, mean) to verify correctness.

## Developer Testing

The test binary supports both full-suite runs and per-case runs.

Run all registered tests through CTest:

```bash
ctest --test-dir 3D-convolution/build --output-on-failure
```

List all split test cases:

```bash
3D-convolution/build/Convolution_test --list
```

Run one specific test case directly:

```bash
3D-convolution/build/Convolution_test rank4_identity_kernel
```

Run one specific split test via CTest:

```bash
ctest --test-dir 3D-convolution/build -R convolution_rank4_identity_kernel --output-on-failure
```

Current split test cases:

- convolution_rank3_scalar_kernel
- convolution_rank3_rectangular
- convolution_rank4_mixed_values
- convolution_rank5_sparse_pattern
- convolution_rank6_high_rank
- convolution_rank3_zero_tensors
- convolution_rank4_identity_kernel
- convolution_rank5_identity_kernel
- convolution_rank3_large_deterministic
- convolution_rank4_large_deterministic

If FFTW3 is missing, the configure step will fetch and build it automatically.
You can force the bundled dependency path with ```-DCONVOLUTION_FORCE_BUNDLED_FFTW3=ON```.

Windows-oids:
To be honest: no tries in windows were done. 
But the steps would be something like this:

1. Go to this link: http://www.fftw.org/install/windows.html
2. Download this library. Just type: ```git clone https://github.com/supernlogn/3D-convolution.git```
3. Configure with CMake in your preferred IDE or terminal environment.
