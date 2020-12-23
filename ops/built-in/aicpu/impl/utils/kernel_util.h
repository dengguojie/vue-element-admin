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

#ifndef AICPU_UTILS_KERNEL_UTIL_H_
#define AICPU_UTILS_KERNEL_UTIL_H_

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

/**
 * @brief get debug string of vector
 * @param values values in vector
 * @return string of values
 */
template <typename T>
inline std::string VectorToString(const std::vector<T> &values) {
  std::stringstream ss;
  for (auto iter = values.begin(); iter != values.end(); ++iter) {
    ss << *iter;
    if (iter != values.end() - 1) {
      ss << ", ";
    }
  }
  return ss.str();
}

/**
 * @brief multiply two nonnegative int64's
 * @param x mul value x
 * @param y mul value y
 * @param xy product of x and y
 * @return true: normal, false: overflow
 */
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
      KERNEL_LOG_ERROR("Can't multiply negative numbers.");
      return false;
    }

    // Otherwise, detect overflow using a division
    if (ux != 0 && uxy / ux != uy) return false;
  }

  // Cast back to signed.  Any negative value will signal an error.
  xy = static_cast<int64_t>(uxy);
  return true;
}

/**
 * @brief add two int64's
 * @param x add value x
 * @param y add value y
 * @param sum sum of x and y
 * @return true: normal, false: overflow
 */
inline bool AddWithoutOverflow(const int64_t x, const int64_t y, int64_t &sum) {
  const uint64_t ux = x;
  const uint64_t uy = y;
  const uint64_t usum = ux + uy;
  sum = static_cast<int64_t>(usum);

  return !(x >= 0 == y >=0 && sum >= 0 != x >= 0);
}

/**
 * @brief normal check for calculation
 * @param ctx context
 * @return status code
 */
uint32_t NormalMathCheck(CpuKernelContext &ctx);

/**
 * @brief normal check for kernel
 * @param ctx context
 * @param inputs_num num of inputs
 * @param outputs_num num of outputs
 * @return status code
 */
uint32_t NormalCheck(CpuKernelContext &ctx,
                     const uint32_t inputs_num,
                     const uint32_t outputs_num);

/**
 * @brief get data type from string
 * @param dtype_str string of data type
 * @return DataType
 */
DataType GetDataType(std::string dtype_str);

/**
 * @brief get string from data type
 * @param dtype data type
 * @return string of data type
 */
std::string GetDataType(DataType dtype);
}  // namespace aicpu
#endif
