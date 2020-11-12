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

#ifndef _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_BCAST_H_
#define _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_BCAST_H_

#include "cpu_context.h"

namespace aicpu {
struct CalcInfo {
  CalcInfo() : input_0(nullptr), input_1(nullptr), output(nullptr) {}
  Tensor *input_0;
  Tensor *input_1;
  Tensor *output;
  std::vector<int64_t> reshape_0;
  std::vector<int64_t> reshape_1;
  std::vector<int64_t> shape_out;
  std::vector<int64_t> bcast_0;
  std::vector<int64_t> bcast_1;
};

class Bcast {
 public:
  Bcast() = default;
  ~Bcast() = default;

  uint32_t GenerateBcastInfo(const CalcInfo &calc_info);
  void GetBcastVec(CalcInfo &calc_info);

 private:
  std::vector<int64_t> x_reshape_;
  std::vector<int64_t> y_reshape_;
  std::vector<int64_t> shape_out_;
  std::vector<int64_t> x_bcast_;
  std::vector<int64_t> y_bcast_;
};
}  // namespace aicpu
#endif  // _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_BCAST_H_
