/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

#include "format_transfer_utils.h"

#include "kernel_util.h"
#include "log.h"

using namespace std;

namespace aicpu {
namespace formats {
bool IsShapeValid(const vector<int64_t> &shape) {
  if (shape.empty()) {
    return false;
  }
  int64_t num = 1;
  for (auto dim : shape) {
    if (dim < 0) {
      string error = "Invalid negative dims in the shape " +
                     FmtToStr(VectorToString(shape));
      KERNEL_LOG_ERROR("%s", error.c_str());
      return false;
    }
    if (dim != 0 && kShapeItemNumMAX / dim < num) {
      string error = "Shape overflow, the total count should be less than " +
                     FmtToStr(kShapeItemNumMAX);
      KERNEL_LOG_ERROR("%s", error.c_str());
      return false;
    }
    num *= dim;
  }
  return true;
}

bool CheckShapeValid(const vector<int64_t> &shape, const int64_t expect_dims) {
  if (expect_dims <= 0 || shape.size() != static_cast<size_t>(expect_dims)) {
    string error = "Invalid shape, dims num " + FmtToStr(shape.size()) +
                   ", expect " + FmtToStr(expect_dims);
    KERNEL_LOG_ERROR("%s", error.c_str());
    return false;
  }
  return IsShapeValid(shape);
}

int64_t GetCubeSizeByDataType(DataType data_type) {
  // Current cube does not support 4 bytes and longer data
  auto size = GetSizeByDataType(data_type);
  if (size <= 0) {
    std::string error = "Failed to get cube size, the data type " +
                        FmtToStr(DTypeStr(data_type)) + " is invalid";
    KERNEL_LOG_ERROR("%s", error.c_str());
    return -1;
  } else if (size == 1) {
    return kCubeSize * 2;  // 32 bytes cube size
  } else {
    return kCubeSize;
  }
}

bool IsTransShapeSrcCorrect(const TransArgs &args,
                            std::vector<int64_t> &expect_shape) {
  if (args.src_shape != expect_shape) {
    string error = "Failed to trans format from" +
                   FmtToStr(FormatToSerialString(args.src_format)) + " to " +
                   FmtToStr(FormatToSerialString(args.dst_format)) +
                   ", invalid relationship between src shape " +
                   FmtToStr(VectorToString(args.src_shape)) + " and dst " +
                   FmtToStr(VectorToString(args.dst_shape));
    KERNEL_LOG_ERROR("%s", error.c_str());
    return false;
  }
  return true;
}

bool IsTransShapeDstCorrect(const TransArgs &args,
                            vector<int64_t> &expect_shape) {
  if (!args.dst_shape.empty() && args.dst_shape != expect_shape) {
    string error =
        "Failed to trans format from " +
        FmtToStr(FormatToSerialString(args.src_format)) + " to " +
        FmtToStr(FormatToSerialString(args.dst_format)) + ", the dst shape" +
        FmtToStr(VectorToString(args.dst_shape)) + " is invalid, expect" +
        FmtToStr(VectorToString(expect_shape));
    KERNEL_LOG_ERROR("%s", error.c_str());
    return false;
  }
  return true;
}

int64_t GetItemNumByShape(const vector<int64_t> &shape) {
  // shape will not be greater than INT_MAX
  int64_t num = 1;
  for (auto dim : shape) {
    num *= dim;
  }
  return num;
}

uint32_t TransFormat(const TransArgs &args, TransResult &result) {
  auto transfer = BuildFormatTransfer(args);
  if (transfer == nullptr) {
    string error = "Failed to trans data from format " +
                   FmtToStr(FormatToSerialString(args.src_format)) + " to " +
                   FmtToStr(FormatToSerialString(args.dst_format));
    KERNEL_LOG_WARN("%s", error.c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto src_shape_size = GetItemNumByShape(args.src_shape);
  if (args.data == nullptr && src_shape_size != 0) {
    KERNEL_LOG_WARN("Invalid input null data");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return transfer->TransFormat(args, result);
}

int64_t Measure(int64_t x, int64_t y) {
  int64_t z = y;
  while (x % y != 0) {
    z = x % y;
    x = y;
    y = z;
  }
  return z;
}
// least common multiple
int64_t Lcm(int64_t a, int64_t b) {
  if (b == 0) {
    return -1;
  }
  int64_t temp = (a * b) / (Measure(a, b));
  return temp;
}
}  // namespace formats
}  // namespace aicpu
