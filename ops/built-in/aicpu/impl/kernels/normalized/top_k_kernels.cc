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

#include "top_k_kernels.h"

#include <securec.h>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"

namespace {
const char *TOPK = "TopK";

template <typename T>
struct ValueIndex {
  T value;
  int32_t index;
  bool operator<(const ValueIndex<T> &other) const {
    if (value == other.value) {
      return index < other.index;
    }
    return value > other.value;
  }
};

class TopK {
 public:
  template <typename T>
  static void GetValueAndSelect(T *input, T *value, int32_t *indices,
                                int32_t k) {
    for (int i = 0; i < k; i++) {
      value[i] = input[i];
      indices[i] = i;
      int node_son = i;
      while (node_son != 0) {
        if (value[node_son] < value[(node_son - 1) / 2]) {
          TopK::Exchange(value, indices, node_son, (node_son - 1) / 2);
          node_son = (node_son - 1) / 2;
        } else if ((value[node_son] == value[(node_son - 1) / 2]) && ((indices[node_son] > indices[(node_son - 1) / 2]))) {
          TopK::Exchange(value, indices, node_son, (node_son - 1) / 2);
          node_son = (node_son - 1) / 2;
        } else {
          break;
        }
      }
    }
  }

  template <typename T>
  static void Select(T *input, T *value, int32_t *indices, int32_t k,
                     int32_t n) {
    for (int i = k; i < n; i++) {
      if (input[i] > value[0]) {
        value[0] = input[i];
        indices[0] = i;
        int32_t node_father = 0;
        while (node_father * 2 + 1 < k) {
          if (node_father * 2 + 2 >= k) {
            if (value[node_father] > value[node_father * 2 + 1]) {
              TopK::Exchange(value, indices, node_father, node_father * 2 + 1);
            }
            else if ((value[node_father] == value[node_father * 2 + 1]) && (indices[node_father] < indices[node_father * 2 + 1])) {
              TopK::Exchange(value, indices, node_father, node_father * 2 + 1);
            }
            break;
          } else if (value[node_father] < value[node_father * 2 + 1]) {
            if (value[node_father] < value[node_father * 2 + 2]) {
              break;
            } else if (value[node_father] > value[node_father * 2 + 2]) {
              TopK::Exchange(value, indices, node_father, node_father * 2 + 2);
              node_father = node_father * 2 + 2;
            } else if (indices[node_father] < indices[node_father * 2 + 2]) {
              TopK::Exchange(value, indices, node_father, node_father * 2 + 2);
              node_father = node_father * 2 + 2;
            } else {
              break;
            }
          } else if (value[node_father] > value[node_father * 2 + 1]) {
            if (value[node_father] <= value[node_father * 2 + 2]) {
              TopK::Exchange(value, indices, node_father, node_father * 2 + 1);
              node_father = node_father * 2 + 1;
            } else {
              if (value[node_father * 2 + 1] < value[node_father * 2 + 2]) {
                TopK::Exchange(value, indices, node_father,
                               node_father * 2 + 1);
                node_father = node_father * 2 + 1;
              } else if (value[node_father * 2 + 1] > value[node_father * 2 + 2]) {
                TopK::Exchange(value, indices, node_father,
                               node_father * 2 + 2);
                node_father = node_father * 2 + 2;
              } else if (indices[node_father * 2 + 1] < indices[node_father * 2 + 2]) {
                TopK::Exchange(value, indices, node_father,
                               node_father * 2 + 2);
                node_father = node_father * 2 + 2;
              } else {
                TopK::Exchange(value, indices, node_father,
                               node_father * 2 + 1);
                node_father = node_father * 2 + 1;
              }
            }
          } else {
            if (value[node_father] > value[node_father * 2 + 2]) {
              TopK::Exchange(value, indices, node_father,
                               node_father * 2 + 2);
              node_father = node_father * 2 + 2;
            }
            else if (value[node_father] < value[node_father * 2 + 2]) {
              if (indices[node_father] < indices[node_father * 2 + 1]) {
                TopK::Exchange(value, indices, node_father,
                               node_father * 2 + 1);
                node_father = node_father * 2 + 1;
              } else {
                break;
              }
            } else if ((indices[node_father] < indices[node_father * 2 + 1]) && (indices[node_father * 2 + 1] > indices[node_father * 2 + 2])) {
              TopK::Exchange(value, indices, node_father,
                               node_father * 2 + 1);
              node_father = node_father * 2 + 1;
            } else if ((indices[node_father] < indices[node_father * 2 + 2]) && (indices[node_father * 2 + 1] < indices[node_father * 2 + 2])) {
              TopK::Exchange(value, indices, node_father,
                               node_father * 2 + 2);
              node_father = node_father * 2 + 2;
            } else {
              break;
            }
          }
        }
      }
    }
  }

 private:
  template <typename T>
  static void Exchange(T *value, int32_t *indices, int32_t index1,
                       int32_t index2) {
    T tmp1 = value[index1];
    value[index1] = value[index2];
    value[index2] = tmp1;
    int32_t tmp2 = indices[index1];
    indices[index1] = indices[index2];
    indices[index2] = tmp2;
  }
};

}  // namespace

namespace aicpu {
uint32_t TopKCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("TopKCpuKernel::Compute start! ");
  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("TopKCpuKernel::GetInputAndCheck failed! ");
    return res;
  }
  switch (matrix_info_.matrix_type) {
    case DT_FLOAT16:
      DoCompute<Eigen::half>(ctx);
      break;
    case DT_FLOAT:
      DoCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      DoCompute<double>(ctx);
      break;
    case DT_UINT8:
      DoCompute<uint8_t>(ctx);
      break;
    case DT_INT8:
      DoCompute<int8_t>(ctx);
      break;
    case DT_UINT16:
      DoCompute<uint16_t>(ctx);
      break;
    case DT_INT16:
      DoCompute<int16_t>(ctx);
      break;
    case DT_UINT32:
      DoCompute<uint32_t>(ctx);
      break;
    case DT_INT32:
      DoCompute<int32_t>(ctx);
      break;
    case DT_UINT64:
      DoCompute<uint64_t>(ctx);
      break;
    case DT_INT64:
      DoCompute<int64_t>(ctx);
      break;
    default: {
      KERNEL_LOG_ERROR("TopK op don't support input tensor types: %d",
                       matrix_info_.matrix_type);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  KERNEL_LOG_INFO("TopKCpuKernel::Compute end!! ");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TopKCpuKernel::DoCompute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("TopKCpuKernel::DoCompute start!! ");
  auto input_data = reinterpret_cast<T *>(input_tensor_->GetData());
  auto values_data = reinterpret_cast<T *>(output_values_->GetData());
  auto indices_data = reinterpret_cast<int32_t *>(output_indices_->GetData());
  auto shard_top_k = [&](size_t start, size_t end) {
    TopKCpuKernel::TopKForNVector(
        input_data + start * col_, values_data + start * k_,
        indices_data + start * k_, col_, k_, end - start, sorted_);
  };
  CpuKernelUtils::ParallelFor(ctx, row_, 1, shard_top_k);
  KERNEL_LOG_INFO("TopKCpuKernel::DoCompute end! ");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TopKCpuKernel::TopKForNVector(T *input, T *value, int32_t *indices,
                                       int col, int k, int n, bool sorted) {
  for (int i = 0; i < n; i++) {
    TopK::GetValueAndSelect(input + i * col, value + i * k, indices + i * k, k);
    TopK::Select(input + i * col, value + i * k, indices + i * k, k, col);
    if (sorted) {
      std::vector<ValueIndex<T>> data(k);
      for (int j = 0; j < k; j++) {
        data[j].value = *(value + i * k + j);
        data[j].index = *(indices + i * k + j);
      }
      std::sort(data.begin(), data.end());
      for (int j = 0; j < k; j++) {
        *(value + i * k + j) = data[j].value;
        *(indices + i * k + j) = data[j].index;
      }
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t TopKCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  // get x
  input_tensor_ = ctx.Input(0);
  if (input_tensor_ == nullptr) {
    KERNEL_LOG_ERROR("get input: x failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // get col_
  std::shared_ptr<TensorShape> input_shape = input_tensor_->GetTensorShape();
  int32_t input_rank = input_shape->GetDims();
  if (input_rank < 1) {
    KERNEL_LOG_ERROR("input must be >= 1-D");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  col_ = input_shape->GetDimSize(input_rank - 1);
  if (col_ <= 0) {
    KERNEL_LOG_ERROR("col_:%lld must be > 0", col_);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // cal row_
  size_t input_size = 1;
  matrix_info_.matrix_type =
      static_cast<DataType>(input_tensor_->GetDataType());
  for (int32_t i = 0; i < input_rank; ++i) {
    matrix_info_.matrix_shape.push_back(input_shape->GetDimSize(i));
    input_size *= input_shape->GetDimSize(i);
  }
  row_ = input_size / col_;

  // get k
  Tensor *k_tensor = ctx.Input(1);
  if (k_tensor == nullptr) {
    KERNEL_LOG_ERROR("get input: k failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  k_ = *static_cast<int32_t *>(k_tensor->GetData());
  if (k_ <= 0) {
    KERNEL_LOG_ERROR("k must be greater than 0, but got %d", k_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (col_ < k_) {
    KERNEL_LOG_ERROR("input must have at least %d(k) columns, but got %d", k_,
                     col_);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // get attr: sorted
  AttrValue *sorted = ctx.GetAttr("sorted");
  KERNEL_CHECK_NULLPTR(sorted, KERNEL_STATUS_PARAM_INVALID,
                       "get attr:sorted failed.");
  sorted_ = sorted->GetBool();

  // get values
  output_values_ = ctx.Output(0);
  if (output_values_ == nullptr) {
    KERNEL_LOG_ERROR("get output: values failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // get indices
  output_indices_ = ctx.Output(1);
  if (output_indices_ == nullptr) {
    KERNEL_LOG_ERROR("get output: indices failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(TOPK, TopKCpuKernel);
}  // namespace aicpu
