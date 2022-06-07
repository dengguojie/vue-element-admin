/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef AI_CPU_RANDOM_DISTRIBUTIONS_H
#define AI_CPU_RANDOM_DISTRIBUTIONS_H

#include <securec.h>
#include <string.h>
#include <cmath>

#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/philox_random.h"

namespace aicpu {
namespace random {
// Helper function to convert a 32-bit interger to a float between [0..1).
float Uint32ToFloat(uint32_t x);
// Helper function to convert two 32-bit integers to a double between [0..1).
double Uint64ToDouble(uint32_t x0, uint32_t x1);
// Helper function to convert two 32-bit uniform integers to two floats
// under the unit normal distribution.
void BoxMullerFloat(uint32_t x0, uint32_t x1, float* f0, float* f1);
// Helper function to convert four 32-bit uniform integers to two doubles
// under the unit normal distribution.
void BoxMullerDouble(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
                     double* d0, double* d1);

// A class that generates unit normal distribution random numbers from the
// underlying random integer generator.
// Arguments:
//   Generator: a generator type that returns a number of uint32 upon each
//              each invocation. It needs to define kResultElementCount for the
//              sample count for each invocation, and ResultType for actual
//              returned sample type.
//   RealType: the data type of the real numbers that will be returned by the
//             distribution. This could be either half or float or double for
//             now.
// This class is meant to be implemented through specialization. The default
// is not defined by design.
template <class Generator, typename RealType>
class NormalDistribution;

// Exactly like the float version, except that we convert to half afterwards;
// There's nothing to gain from working in half internally.
template <class Generator>
class NormalDistribution<Generator, Eigen::half> {
 public:
  static constexpr int32_t kResultElementCount = Generator::kResultElementCount;
  using ResultType = Array<Eigen::half, kResultElementCount>;
  using ResultElementType = Eigen::half;

  ResultType operator()(Generator* gen) {
    ResultType result;
    typename Generator::ResultType sample = (*gen)();
    float f[2];
    for (int32_t i = 0; i < kResultElementCount; i += 2) {
      BoxMullerFloat(sample[i], sample[i + 1], &f[0], &f[1]);
      result[i] = Eigen::half(f[0]);
      result[i + 1] = Eigen::half(f[1]);
    }
    return result;
  }
};

template <class Generator>
class NormalDistribution<Generator, float> {
 public:
  // The number of elements that will be returned. default value is 4 for philox
  static constexpr int32_t kResultElementCount = Generator::kResultElementCount;
  using ResultType = Array<float, kResultElementCount>;
  using ResultElementType = float;

  ResultType operator()(Generator* gen) {
    ResultType result;
    typename Generator::ResultType sample = (*gen)();
    for (int32_t i = 0; i < kResultElementCount; i += 2) {
      BoxMullerFloat(sample[i], sample[i + 1], &result[i], &result[i + 1]);
    }
    return result;
  }
};

template <class Generator>
class NormalDistribution<Generator, double> {
 public:
  // The number of elements that will be returned.
  static constexpr int32_t kResultElementCount =
      Generator::kResultElementCount / 2;
  using ResultType = Array<double, kResultElementCount>;
  using ResultElementType = double;

  ResultType operator()(Generator* gen) {
    ResultType result;
    typename Generator::ResultType sample = (*gen)();
    for (int32_t i = 0; i < kResultElementCount; i += 2) {
      const int i2 = 2 * i;
      BoxMullerDouble(sample[i2], sample[i2 + 1], sample[i2 + 2],
                      sample[i2 + 3], &result[i], &result[i + 1]);
    }
    return result;
  }
};

// This function implements the Box-Muller transform:
void BoxMullerFloat(uint32_t x0, uint32_t x1, float* f0, float* f1) {
  const float epsilon = 1.0e-7f;
  float u1 = Uint32ToFloat(x0);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  const float v1 = 2.0f * M_PI * Uint32ToFloat(x1);
  const float u2 = Eigen::numext::sqrt(-2.0f * Eigen::numext::log(u1));
  *f0 = Eigen::numext::sin(v1);
  *f1 = Eigen::numext::cos(v1);
  *f0 *= u2;
  *f1 *= u2;
}

// This function implements the Box-Muller transform:
void BoxMullerDouble(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
                     double* d0, double* d1) {
  const double epsilon = 1.0e-7;
  double u1 = Uint64ToDouble(x0, x1);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  const double v1 = 2 * M_PI * Uint64ToDouble(x2, x3);
  const double u2 = Eigen::numext::sqrt(-2.0 * Eigen::numext::log(u1));
  *d0 = Eigen::numext::sin(v1);
  *d1 = Eigen::numext::cos(v1);
  *d0 *= u2;
  *d1 *= u2;
}

float Uint32ToFloat(uint32_t x) {
  const uint32_t man = x & 0x7fffffu;  // 23 bit mantissa
  const uint32_t exp = static_cast<uint32_t>(127);
  const uint32_t val = (exp << 23) | man;
  float result;
  memcpy_s(&result, sizeof(val), &val, sizeof(val));
  return result - 1.0f;
}

double Uint64ToDouble(uint32_t x0, uint32_t x1) {
  const uint32_t mhi = x0 & 0xfffffu;  // upper 20 bits of mantissa
  const uint32_t mlo = x1;             // lower 32 bits of mantissa
  const uint64_t man = (static_cast<uint64_t>(mhi) << 32) | mlo;  // mantissa
  const uint64_t exp = static_cast<uint64_t>(1023);
  const uint64_t val = (exp << 52) | man;
  double result;
  memcpy_s(&result, sizeof(val), &val, sizeof(val));
  return result - 1.0;
}
}  // namespace random
}  // namespace aicpu

#endif