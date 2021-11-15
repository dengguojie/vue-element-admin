/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef AICPU_KERNELS_HOST_FORMAT_TRANSFER_FORMAT_TRANSFER_UTILS_H_
#define AICPU_KERNELS_HOST_FORMAT_TRANSFER_FORMAT_TRANSFER_UTILS_H_

#include <string>
#include <vector>

#include "register_format_transfer.h"

namespace aicpu {
namespace formats {
static const int kCubeSize = 16;
static const int kNiSize = 16;
static const int64_t kShapeItemNumMAX = 1024UL * 1024UL * 1024UL * 1024UL;
int64_t Lcm(int64_t a, int64_t b);
bool IsShapeValid(const std::vector<int64_t> &shape);

bool CheckShapeValid(const std::vector<int64_t> &shape,
                     const int64_t expect_dims);

int64_t GetCubeSizeByDataType(DataType data_type);

bool IsTransShapeSrcCorrect(const TransArgs &args,
                            std::vector<int64_t> &expect_shape);

bool IsTransShapeDstCorrect(const TransArgs &args,
                            std::vector<int64_t> &expect_shape);

int64_t GetItemNumByShape(const std::vector<int64_t> &shape);

template <typename T>
T Ceil(T n1, T n2) {
  if (n1 == 0) {
    return 0;
  }
  return (n2 != 0) ? (n1 - 1) / n2 + 1 : 0;
}

/**
 * Convert the data format, and put the converted format and length in the
 * result
 * @param args
 * @param result
 * @return
 */
uint32_t TransFormat(const TransArgs &args, TransResult &result);
}  // namespace formats
}  // namespace aicpu
#endif  // AICPU_KERNELS_HOST_FORMAT_TRANSFER_FORMAT_TRANSFER_UTILS_H_
