/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_MATH_UTIL_H_
#define _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_MATH_UTIL_H_

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <sstream>

#include "cpu_context.h"
#include "log.h"
#include "status.h"

namespace aicpu {
const uint32_t kThreadNum = 32;
const uint32_t kFirstInputIndex = 0;
const uint32_t kSecondInputIndex = 1;
const uint32_t kFirstOutputIndex = 0;
const uint32_t kDynamicInput = -1;
const uint32_t kDynamicOutput = -2;

// attr name
const std::string ATTR_NAME_DTYPE = "dtype";
const std::string ATTR_NAME_RANDOM_UNIFORM_SEED = "seed";
const std::string ATTR_NAME_RANDOM_UNIFORM_SEED2 = "seed2";

/// @ingroup kernel_util
/// @brief get debug string of vector
/// @param [in] values  values in vector
/// @return string of values
template <typename T>
inline std::string VectorToString(const std::vector<T> &values) {
  std::stringstream ss;
  ss << '[';
  for (auto iter = values.begin(); iter != values.end(); ++iter) {
    ss << *iter;
    if (iter != values.end() - 1) {
      ss << ", ";
    }
  }
  ss << ']';
  return ss.str();
}

/// @ingroup kernel_util
/// @brief multiply two nonnegative int64's
/// @param
///  @li [in]  x  mul value x
///  @li [in]  y  mul value y
///  @li [out] xy product of x and y
/// @return true: normal, false: overflow
inline bool MulWithoutOverflow(const int64_t x, const int64_t y, int64_t &xy) {
  // Multiply in uint64 rather than int64 since signed overflow is undefined.
  // Negative values will wrap around to large unsigned values in the casts
  // (see section 4.7 [conv.integral] of the C++14 standard).
  const uint64_t ux = x;
  const uint64_t uy = y;
  const uint64_t uxy = ux * uy;

  // Check if we overflow uint64, using a cheap check if both inputs are small
  if ((ux | uy) >> 32 != 0) {
    // Ensure nonnegativity.  Note that negative numbers will appear "large"
    // to the unsigned comparisons above.
    if (x < 0 || y < 0) {
      KERNEL_LOG_ERROR("can't multiply negative numbers.");
      return false;
    }

    // Otherwise, detect overflow using a division
    if (ux != 0 && uxy / ux != uy) return false;
  }

  // Cast back to signed.  Any negative value will signal an error.
  xy = static_cast<int64_t>(uxy);
  return true;
}

/// @ingroup kernel_util
/// @brief add two int64's
/// @param
///  @li [in]  x    add value x
///  @li [in]  y    add value y
///  @li [out] sum  sum of x and y
/// @return true: normal, false: overflow
inline bool AddWithoutOverflow(const int64_t x, const int64_t y, int64_t &sum) {
  const uint64_t ux = x;
  const uint64_t uy = y;
  const uint64_t usum = ux + uy;
  sum = static_cast<int64_t>(usum);

  return !(x >= 0 == y >=0 && sum >= 0 != x >= 0);
}

/// @ingroup kernel_util
/// @brief normal check for calculation
/// @param [in] ctx  context
/// @return uint32_t
uint32_t NormalMathCheck(CpuKernelContext &ctx);

/// @ingroup kernel_util
/// @brief normal check for kernel
/// @param
///  @li [in] ctx           context
///  @li [in] inputs_num    num of inputs
///  @li [in] outputs_num   num of outputs
/// @return uint32_t
uint32_t NormalCheck(CpuKernelContext &ctx,
                     const uint32_t inputs_num,
                     const uint32_t outputs_num);
}  // namespace aicpu
#endif  // _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_MATH_UTIL_H_
