/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#if !defined(CUTLASS_ENABLE_SYCL)
#include <cuda_runtime_api.h>
#endif

#include "cutlass_unit_test.h"

#if defined(CUTLASS_ENABLE_SYCL)
#include <sycl/sycl.hpp>
#endif

#if !defined(CUTLASS_ENABLE_SYCL)

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gets a CUDA device
cudaDeviceProp GetCudaDevice() {

  cudaError_t err;

  int cudaDeviceId;
  err = cudaGetDevice(&cudaDeviceId);
  if (cudaSuccess != err) {
    std::cerr << "*** Error: Could not detect active GPU device ID"
              << " [" << cudaGetErrorString(err) << "]" << std::endl;
    exit(1);
  }

  cudaDeviceProp deviceProperties;
  err = cudaGetDeviceProperties(&deviceProperties, cudaDeviceId);

  return deviceProperties;
}

/// Prints device properties
std::ostream &operator<<(std::ostream &out, cudaDeviceProp const &deviceProperties) {

  int deviceMajorMinor = deviceProperties.major * 10 + deviceProperties.minor;
  if (deviceMajorMinor) {
    int32_t clock_MHz = deviceProperties.clockRate / 1000;
    out << "GPU(compute_"
      << deviceMajorMinor << ", "
      << deviceProperties.multiProcessorCount << " SMs @ " << clock_MHz << " MHz)";
  }
  else {
    out << "No CUDA device.";
  }

  return out;
}

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Sets flags for Unit test
void FilterArchitecture() {
  // Default flags can be overwritten by --gtest_filter from commandline

  int const kMaxDevice = 999;

  // Defines text filters for each GEMM kernel based on minimum supported compute capability
  struct {

    /// Unit test filter string
    char const *filter;

    /// Minimum compute capability for the kernels in the named test
    int min_compute_capability;

    /// Maximum compute capability for which the kernels are enabled 
    int max_compute_capability;
  } 
  test_filters[] = {
    { "SM50*",                      50, kMaxDevice},
    { "SM60*",                      60, kMaxDevice},
    { "SM61*",                      61, kMaxDevice},
    { "SM70*",                      70, 75},
    { "SM75*",                      75, kMaxDevice},
    { "SM80*",                      80, kMaxDevice},
    { "SM89*",                      89, 89},
    { "SM90*",                      90, 90},
    { "IntelPVC",                   0,  0},
    { 0, 0, false }
  };


#if defined(CUTLASS_ENABLE_SYCL)
  using namespace sycl::ext::oneapi::experimental;

  // We might be adding PVC unit tests someday
  std::map<architecture, int> arch_map {
    {architecture::nvidia_gpu_sm_50, 50},
    {architecture::nvidia_gpu_sm_52, 52},
    {architecture::nvidia_gpu_sm_53, 53},
    {architecture::nvidia_gpu_sm_60, 60},
    {architecture::nvidia_gpu_sm_61, 61},
    {architecture::nvidia_gpu_sm_62, 62},
    {architecture::nvidia_gpu_sm_70, 70},
    {architecture::nvidia_gpu_sm_72, 72},
    {architecture::nvidia_gpu_sm_75, 75},
    {architecture::nvidia_gpu_sm_80, 80},
    {architecture::nvidia_gpu_sm_86, 86},
    {architecture::nvidia_gpu_sm_89, 89},
    {architecture::nvidia_gpu_sm_90, 90},
    {architecture::nvidia_gpu_sm_90a, 90},
    {architecture::intel_gpu_pvc, 0}
  };
  auto device_architecture = 
        syclcompat::get_default_queue().get_device().get_info<info::device::architecture>();
  if (device_architecture == architecture::unknown) {
    throw std::runtime_error("Encountered Unknown architecture.");
  }

  if(auto search_result = arch_map.find(device_architecture); search_result == arch_map.end()) {
    throw std::runtime_error("Detected Architecture is not supported.");
  }

  const int deviceMajorMinor = arch_map[device_architecture];

#else
  cudaError_t err;

  int cudaDeviceId;
  err = cudaGetDevice(&cudaDeviceId);
  if (cudaSuccess != err) {
    std::cerr << "*** Error: Could not detect active GPU device ID"
              << " [" << cudaGetErrorString(err) << "]" << std::endl;
    exit(1);
  }

  cudaDeviceProp deviceProperties;
  err = cudaGetDeviceProperties(&deviceProperties, cudaDeviceId);
  if (cudaSuccess != err) {
    std::cerr << "*** Error: Could not get device properties for GPU " << cudaDeviceId << " ["
              << cudaGetErrorString(err) << "]" << std::endl;
    exit(1);
  }

  int deviceMajorMinor = deviceProperties.major * 10 + deviceProperties.minor;
#endif
  
  std::stringstream ss;
  ss << "-";
  for (int i = 0, j = 0; test_filters[i].filter; ++i) {

    if (deviceMajorMinor < test_filters[i].min_compute_capability ||
        deviceMajorMinor > test_filters[i].max_compute_capability) {

      ss << (j++ ? ":" : "") << test_filters[i].filter;
    }
  }

  ::testing::GTEST_FLAG(filter) = ss.str();
  // Set negative test filters
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int CutlassUnitTestProblemCount() {
    if(const char* problem_count = std::getenv("CUTLASS_UNIT_TEST_PROBLEM_COUNT")) {

        return std::stoi(problem_count);
    } 

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
