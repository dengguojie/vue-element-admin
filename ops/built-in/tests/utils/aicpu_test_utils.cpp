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
#include "aicpu_test_utils.h"

uint64_t CalTotalElements(std::vector<std::vector<int64_t>> &shapes,
                          uint32_t index) {
  if(index < 0) {
    return 0;
  }
  uint64_t nums = 1;
  for(auto shape : shapes[index]) {
    nums = nums * shape;
  }
  return nums;
}

template <>
bool CompareResult(float output[], float expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    if (std::fabs(output[i] - expect_output[i]) > 1e-6) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

template <>
bool CompareResult(double output[], double expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    if (std::fabs(output[i] - expect_output[i]) > 1e-10) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

bool CompareResult(Eigen::half output[], Eigen::half expect_output[],
    uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    if ((output[i] - expect_output[i] > Eigen::half(1e-3)) ||
        (output[i] - expect_output[i] < Eigen::half(-1e-3))) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}