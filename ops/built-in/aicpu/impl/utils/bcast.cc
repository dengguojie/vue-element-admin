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

#include "bcast.h"
#include <algorithm>
#include "log.h"
#include "status.h"

namespace {
const int64_t kNoBroadcastValue = 1;
}

namespace aicpu {
uint32_t Bcast::GenerateBcastInfo(const CalcInfo &calc_info) {
  const std::vector<int64_t> &shape_x =
      calc_info.input_0->GetTensorShape()->GetDimSizes();
  const std::vector<int64_t> &shape_y =
      calc_info.input_1->GetTensorShape()->GetDimSizes();
  const std::vector<int64_t> &shape_out =
      calc_info.output->GetTensorShape()->GetDimSizes();
  x_reshape_ = shape_x;
  y_reshape_ = shape_y;
  shape_out_ = shape_out;
  if (shape_x.empty() && shape_y.empty()) {
    // Eigen support scalar
    return KERNEL_STATUS_OK;
  }

  // resize shape_x or shape_y to make size equal
  std::reverse(x_reshape_.begin(), x_reshape_.end());
  std::reverse(y_reshape_.begin(), y_reshape_.end());
  size_t dim_num_x = x_reshape_.size();
  size_t dim_num_y = y_reshape_.size();
  size_t max_size = dim_num_x > dim_num_y ? dim_num_x : dim_num_y;
  if (dim_num_x < dim_num_y) {
    x_reshape_.resize(max_size, kNoBroadcastValue);
  } else if (dim_num_x > dim_num_y) {
    y_reshape_.resize(max_size, kNoBroadcastValue);
  }
  std::reverse(x_reshape_.begin(), x_reshape_.end());
  std::reverse(y_reshape_.begin(), y_reshape_.end());

  // Check if shape match
  if (shape_out.size() != max_size) {
    KERNEL_LOG_ERROR("shape mismatch, max_dim_in=%zu, dim_out=%zu.", max_size,
                     shape_out.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t i = 0; i < max_size; i++) {
    if (shape_out_[i] != std::max(x_reshape_[i], y_reshape_[i])) {
      KERNEL_LOG_ERROR(
          "shape mismatch, dim_x[%zu]=%lld, dim_y[%zu]=%lld, "
          "dim_out[%zu]=%lld.",
          i, x_reshape_[i], i, y_reshape_[i], i, shape_out_[i]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  // genarate broarcast info
  x_bcast_.resize(max_size, kNoBroadcastValue);
  y_bcast_.resize(max_size, kNoBroadcastValue);
  for (size_t i = 0; i < max_size; i++) {
    if (x_reshape_[i] == y_reshape_[i]) {
      continue;
    }
    if (x_reshape_[i] == kNoBroadcastValue) {
      x_bcast_[i] = y_reshape_[i];
    } else if (y_reshape_[i] == kNoBroadcastValue) {
      y_bcast_[i] = x_reshape_[i];
    } else {
      KERNEL_LOG_ERROR(
          "Broadcast not support, dim_x[%zu]=%lld, dim_y[%zu]=%lld.", i,
          x_reshape_[i], i, y_reshape_[i]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  return KERNEL_STATUS_OK;
}

void Bcast::GetBcastVec(CalcInfo &calc_info) {
  calc_info.reshape_0 = std::move(x_reshape_);
  calc_info.reshape_1 = std::move(y_reshape_);
  calc_info.shape_out = std::move(shape_out_);
  calc_info.bcast_0 = std::move(x_bcast_);
  calc_info.bcast_1 = std::move(y_bcast_);
}

void Bcast::BCastIndexes(std::vector<int64_t> &x_indexes,
                         std::vector<int64_t> &y_indexes) {
  std::reverse(x_reshape_.begin(), x_reshape_.end());
  std::reverse(y_reshape_.begin(), y_reshape_.end());
  std::reverse(shape_out_.begin(), shape_out_.end());

  // Process 0-th dimension
  int64_t x_dim = 1;
  int64_t y_dim = 1;
  int64_t out_dim = 1;

  // If x and y are both scalar, then shape_out_ is empty
  if (!shape_out_.empty()) {
    x_dim = x_reshape_.at(0);
    y_dim = y_reshape_.at(0);
    out_dim = shape_out_.at(0);
  }

  int64_t x_bias = x_dim;
  int64_t y_bias = y_dim;

  for (int64_t i = 0; i < out_dim; i++) {
    x_indexes.push_back(x_dim == 1 ? 0 : i);
    y_indexes.push_back(y_dim == 1 ? 0 : i);
  }

  // Process the remaining dimensions
  for (size_t i = 1; i < shape_out_.size(); i++) {
    x_dim = x_reshape_.at(i);    // i-th dimension of x.
    y_dim = y_reshape_.at(i);    // i-th dimension of y.
    out_dim = shape_out_.at(i);  // i-th dimension of shape_out_.

    int64_t stride = x_indexes.size();
    for (int64_t j = 1; j < out_dim; j++) {
      for (int64_t k = 0; k < stride; k++) {
        x_indexes.push_back(x_indexes.at(k) + (x_dim == 1 ? 0 : (j * x_bias)));
        y_indexes.push_back(y_indexes.at(k) + (y_dim == 1 ? 0 : (j * y_bias)));
      }
    }
    x_bias *= x_dim;
    y_bias *= y_dim;
  }

  std::reverse(x_reshape_.begin(), x_reshape_.end());
  std::reverse(y_reshape_.begin(), y_reshape_.end());
  std::reverse(shape_out_.begin(), shape_out_.end());
}
}  // namespace aicpu
