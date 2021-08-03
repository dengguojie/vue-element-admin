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
struct BCalcInfo {
  BCalcInfo() : input_0(nullptr), input_1(nullptr), output(nullptr) {}
  Tensor *input_0;
  Tensor *input_1;
  Tensor *output;
  std::vector<int64_t> reshape_0;
  std::vector<int64_t> reshape_1;
  std::vector<int64_t> shape_out;
  std::vector<int64_t> bcast_0;
  std::vector<int64_t> bcast_1;
  std::vector<int64_t> x_indexes;
  std::vector<int64_t> y_indexes;
};

class Bcast {
 public:
  Bcast() : valid_(true){};
  Bcast(std::vector<int64_t> &x_shape, std::vector<int64_t> &y_shape);
  ~Bcast() = default;

  uint32_t GenerateBcastInfo(const BCalcInfo &calc_info);
  void GetBcastVec(BCalcInfo &calc_info);
  void BCastIndexes(std::vector<int64_t> &x_indexes,
                    std::vector<int64_t> &y_indexes);
  bool IsValid() const { return valid_; }
  std::vector<int64_t> &x_reshape() { return x_reshape_; }
  std::vector<int64_t> &y_reshape() { return y_reshape_; }
  std::vector<int64_t> &result_shape() { return result_shape_; }
  std::vector<int64_t> &x_bcast() { return x_bcast_; }
  std::vector<int64_t> &y_bcast() { return y_bcast_; }

 private:
  uint32_t Init(const std::vector<int64_t> &x, const std::vector<int64_t> &y);
 private:
  bool valid_;
  std::vector<int64_t> x_reshape_;
  std::vector<int64_t> y_reshape_;
  std::vector<int64_t> shape_out_;
  std::vector<int64_t> x_bcast_;
  std::vector<int64_t> y_bcast_;
  std::vector<int64_t> result_shape_;
};
}  // namespace aicpu
#endif  // _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_BCAST_H_
