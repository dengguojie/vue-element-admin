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

#include "top_k.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

namespace {
const char *TOPK = "TopK";

template <typename T>
struct ValueIndex {
  T value;
  int32_t index;
};

template <typename T>
bool CompareDescending(const ValueIndex<T> &one, const ValueIndex<T> &another) {
  if (one.value == another.value) {
    return one.index < another.index;
  }
  return one.value > another.value;
}

template <typename T>
bool CompareAscending(const ValueIndex<T> &one, const ValueIndex<T> &another) {
  if (one.value == another.value) {
    return one.index < another.index;
  }
  return one.value < another.value;
}

class TopKMax {
 public:
  /**
   * @brief put the first k input data into value
   * @param input address of input data
   * @param value address of value data
   * @param indices address of indices data
   * @param k number of top elements
   */
  template <typename T>
  static void GetValueAndSelect(T *input, T *value, int32_t *indices,
                                int32_t k) {
    for (int i = 0; i < k; i++) {
      value[i] = input[i];
      indices[i] = i;
      int node_son = i;
      while (node_son != 0) {
        if ((value[node_son] < value[(node_son - 1) / 2]) ||
            ((value[node_son] == value[(node_son - 1) / 2]) &&
             (indices[node_son] > indices[(node_son - 1) / 2]))) {
          TopKMax::Exchange(value, indices, node_son, (node_son - 1) / 2);
          node_son = (node_son - 1) / 2;
        } else {
          break;
        }
      }
    }
  }

  /**
   * @brief put the k-th input data to the n-th input data into value
   * @param input address of input data
   * @param value address of value data
   * @param indices address of indices data
   * @param k number of top elements
   * @param n the length of one vector
   */
  template <typename T>
  static void Select(T *input, T *value, int32_t *indices, int32_t k,
                     int32_t n) {
    for (int i = k; i < n; i++) {
      if (input[i] > value[0]) {
        // when input is greater than the minimum heap
        value[0] = input[i];
        indices[0] = i;
        int32_t node_father = 0;
        while (node_father * 2 + 1 < k) {
          int32_t node_left = node_father * 2 + 1;
          int32_t node_right = node_father * 2 + 2;
          if (node_right >= k) {
            // when the right son node doesn't exit
            TopKMax::ExchangeForTypeOne(value, indices, node_father, node_left);
            break;
          } else if (value[node_father] < value[node_left]) {
            // when the father node is less than the left son node
            if (TopKMax::ExchangeForTypeTwo(value, indices, node_father,
                                         node_right)) {
              break;
            }
          } else if (value[node_father] > value[node_left]) {
            // when the father node is greater than the left son node
            TopKMax::ExchangeForTypeThree(value, indices, node_father, node_left,
                                       node_right);
          } else {
            // when the father node is equal to the left son node
            if (TopKMax::ExchangeForTypeFour(value, indices, node_father,
                                          node_left, node_right)) {
              break;
            }
          }
        }
      }
    }
  }

 private:
  /**
   * @brief exchange the father node and the child node when the right son node
   * doesn't exit
   * @param value address of value data
   * @param indices address of indices data
   * @param node_father index of father node
   * @param node_left index of left son node
   */
  template <typename T>
  static void ExchangeForTypeOne(T *value, int32_t *indices,
                                 int32_t node_father, int32_t node_left) {
    if ((value[node_father] > value[node_left]) ||
        ((value[node_father] == value[node_left]) &&
         (indices[node_father] < indices[node_left]))) {
      TopKMax::Exchange(value, indices, node_father, node_left);
    }
  }

  /**
   * @brief exchange the father node and the child node when the father node is
   * less than the left son node
   * @param value address of value data
   * @param indices address of indices data
   * @param node_father index of father node
   * @param node_right index of right son node
   * @return whether to exit loop
   */
  template <typename T>
  static bool ExchangeForTypeTwo(T *value, int32_t *indices,
                                 int32_t &node_father, int32_t node_right) {
    if (value[node_father] < value[node_right]) {
      return true;
    } else if ((value[node_father] > value[node_right]) ||
               (indices[node_father] < indices[node_right])) {
      TopKMax::Exchange(value, indices, node_father, node_right);
      node_father = node_right;
      return false;
    } else {
      return true;
    }
  }

  /**
   * @brief exchange the father node and the child node when the father node is
   * greater than the left son node
   * @param value address of value data
   * @param indices address of indices data
   * @param node_father index of father node
   * @param node_left index of left son node
   * @param node_right index of right son node
   */
  template <typename T>
  static void ExchangeForTypeThree(T *value, int32_t *indices,
                                   int32_t &node_father, int32_t node_left,
                                   int32_t node_right) {
    if (value[node_father] <= value[node_right]) {
      TopKMax::Exchange(value, indices, node_father, node_left);
      node_father = node_left;
    } else {
      if (value[node_left] < value[node_right]) {
        TopKMax::Exchange(value, indices, node_father, node_left);
        node_father = node_left;
      } else if (value[node_left] > value[node_right]) {
        TopKMax::Exchange(value, indices, node_father, node_right);
        node_father = node_right;
      } else if (indices[node_left] < indices[node_right]) {
        TopKMax::Exchange(value, indices, node_father, node_right);
        node_father = node_right;
      } else {
        TopKMax::Exchange(value, indices, node_father, node_left);
        node_father = node_left;
      }
    }
  }

  /**
   * @brief exchange the father node and the child node when the father node is
   * equal to the left son node
   * @param value address of value data
   * @param indices address of indices data
   * @param node_father index of father node
   * @param node_left index of left son node
   * @param node_right index of right son node
   * @return whether to exit loop
   */
  template <typename T>
  static bool ExchangeForTypeFour(T *value, int32_t *indices,
                                  int32_t &node_father, int32_t node_left,
                                  int32_t node_right) {
    if (value[node_father] > value[node_right]) {
      TopKMax::Exchange(value, indices, node_father, node_right);
      node_father = node_right;
    } else if (value[node_father] < value[node_right]) {
      if (indices[node_father] < indices[node_left]) {
        TopKMax::Exchange(value, indices, node_father, node_left);
        node_father = node_left;
      } else {
        return true;
      }
    } else if ((indices[node_father] < indices[node_left]) &&
               (indices[node_left] > indices[node_right])) {
      TopKMax::Exchange(value, indices, node_father, node_left);
      node_father = node_left;
    } else if ((indices[node_father] < indices[node_right]) &&
               (indices[node_left] < indices[node_right])) {
      TopKMax::Exchange(value, indices, node_father, node_right);
      node_father = node_right;
    } else {
      return true;
    }
    return false;
  }

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

class TopKMin {
 public:
  /**
   * @brief put the first k input data into value
   * @param input address of input data
   * @param value address of value data
   * @param indices address of indices data
   * @param k number of top elements
   */
  template <typename T>
  static void GetValueAndSelect(T *input, T *value, int32_t *indices,
                                int32_t k) {
    for (int i = 0; i < k; i++) {
      value[i] = input[i];
      indices[i] = i;
      int node_son = i;
      while (node_son != 0) {
        if ((value[node_son] > value[(node_son - 1) / 2]) ||
            ((value[node_son] == value[(node_son - 1) / 2]) &&
             (indices[node_son] > indices[(node_son - 1) / 2]))) {
          TopKMin::Exchange(value, indices, node_son, (node_son - 1) / 2);
          node_son = (node_son - 1) / 2;
        } else {
          break;
        }
      }
    }
  }

  /**
   * @brief put the k-th input data to the n-th input data into value
   * @param input address of input data
   * @param value address of value data
   * @param indices address of indices data
   * @param k number of top elements
   * @param n the length of one vector
   */
  template <typename T>
  static void Select(T *input, T *value, int32_t *indices, int32_t k,
                     int32_t n) {
    for (int i = k; i < n; i++) {
      if (input[i] < value[0]) {
        // when input is greater than the minimum heap
        value[0] = input[i];
        indices[0] = i;
        int32_t node_father = 0;
        while (node_father * 2 + 1 < k) {
          int32_t node_left = node_father * 2 + 1;
          int32_t node_right = node_father * 2 + 2;
          if (node_right >= k) {
            // when the right son node doesn't exit
            TopKMin::ExchangeForTypeOne(value, indices, node_father, node_left);
            break;
          } else if (value[node_father] > value[node_left]) {
            // when the father node is less than the left son node
            if (TopKMin::ExchangeForTypeTwo(value, indices, node_father,
                                         node_right)) {
              break;
            }
          } else if (value[node_father] < value[node_left]) {
            // when the father node is greater than the left son node
            TopKMin::ExchangeForTypeThree(value, indices, node_father, node_left,
                                       node_right);
          } else {
            // when the father node is equal to the left son node
            if (TopKMin::ExchangeForTypeFour(value, indices, node_father,
                                          node_left, node_right)) {
              break;
            }
          }
        }
      }
    }
  }

 private:
  /**
   * @brief exchange the father node and the child node when the right son node
   * doesn't exit
   * @param value address of value data
   * @param indices address of indices data
   * @param node_father index of father node
   * @param node_left index of left son node
   */
  template <typename T>
  static void ExchangeForTypeOne(T *value, int32_t *indices,
                                 int32_t node_father, int32_t node_left) {
    if ((value[node_father] < value[node_left]) ||
        ((value[node_father] == value[node_left]) &&
         (indices[node_father] < indices[node_left]))) {
      TopKMin::Exchange(value, indices, node_father, node_left);
    }
  }

  /**
   * @brief exchange the father node and the child node when the father node is
   * greater than the left son node
   * @param value address of value data
   * @param indices address of indices data
   * @param node_father index of father node
   * @param node_right index of right son node
   * @return whether to exit loop
   */
  template <typename T>
  static bool ExchangeForTypeTwo(T *value, int32_t *indices,
                                 int32_t &node_father, int32_t node_right) {
    if (value[node_father] > value[node_right]) {
      return true;
    } else if ((value[node_father] < value[node_right]) ||
               (indices[node_father] < indices[node_right])) {
      TopKMin::Exchange(value, indices, node_father, node_right);
      node_father = node_right;
      return false;
    } else {
      return true;
    }
  }

  /**
   * @brief exchange the father node and the child node when the father node is
   * less than the left son node
   * @param value address of value data
   * @param indices address of indices data
   * @param node_father index of father node
   * @param node_left index of left son node
   * @param node_right index of right son node
   */
  template <typename T>
  static void ExchangeForTypeThree(T *value, int32_t *indices,
                                   int32_t &node_father, int32_t node_left,
                                   int32_t node_right) {
    if (value[node_father] >= value[node_right]) {
      TopKMin::Exchange(value, indices, node_father, node_left);
      node_father = node_left;
    } else {
      if (value[node_left] > value[node_right]) {
        TopKMin::Exchange(value, indices, node_father, node_left);
        node_father = node_left;
      } else if (value[node_left] < value[node_right]) {
        TopKMin::Exchange(value, indices, node_father, node_right);
        node_father = node_right;
      } else if (indices[node_left] < indices[node_right]) {
        TopKMin::Exchange(value, indices, node_father, node_right);
        node_father = node_right;
      } else {
        TopKMin::Exchange(value, indices, node_father, node_left);
        node_father = node_left;
      }
    }
  }

  /**
   * @brief exchange the father node and the child node when the father node is
   * equal to the left son node
   * @param value address of value data
   * @param indices address of indices data
   * @param node_father index of father node
   * @param node_left index of left son node
   * @param node_right index of right son node
   * @return whether to exit loop
   */
  template <typename T>
  static bool ExchangeForTypeFour(T *value, int32_t *indices,
                                  int32_t &node_father, int32_t node_left,
                                  int32_t node_right) {
    if (value[node_father] < value[node_right]) {
      TopKMin::Exchange(value, indices, node_father, node_right);
      node_father = node_right;
    } else if (value[node_father] > value[node_right]) {
      if (indices[node_father] < indices[node_left]) {
        TopKMin::Exchange(value, indices, node_father, node_left);
        node_father = node_left;
      } else {
        return true;
      }
    } else if ((indices[node_father] < indices[node_left]) &&
               (indices[node_left] > indices[node_right])) {
      TopKMin::Exchange(value, indices, node_father, node_left);
      node_father = node_left;
    } else if ((indices[node_father] < indices[node_right]) &&
               (indices[node_left] < indices[node_right])) {
      TopKMin::Exchange(value, indices, node_father, node_right);
      node_father = node_right;
    } else {
      return true;
    }
    return false;
  }

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
  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }
  switch (data_type_) {
    case DT_FLOAT16:
      res = DoCompute<Eigen::half>(ctx);
      break;
    case DT_FLOAT:
      res = DoCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      res = DoCompute<double>(ctx);
      break;
    case DT_UINT8:
      res = DoCompute<uint8_t>(ctx);
      break;
    case DT_INT8:
      res = DoCompute<int8_t>(ctx);
      break;
    case DT_UINT16:
      res = DoCompute<uint16_t>(ctx);
      break;
    case DT_INT16:
      res = DoCompute<int16_t>(ctx);
      break;
    case DT_UINT32:
      res = DoCompute<uint32_t>(ctx);
      break;
    case DT_INT32:
      res = DoCompute<int32_t>(ctx);
      break;
    case DT_UINT64:
      res = DoCompute<uint64_t>(ctx);
      break;
    case DT_INT64:
      res = DoCompute<int64_t>(ctx);
      break;
    default: {
      KERNEL_LOG_ERROR("TopK op don't support input tensor type [%s]",
                       DTypeStr(data_type_).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  if (res != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TopKCpuKernel::DoCompute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("TopKCpuKernel::DoCompute start");
  auto input_data = reinterpret_cast<T *>(input_tensor_->GetData());
  auto values_data = reinterpret_cast<T *>(output_values_->GetData());
  auto indices_data = reinterpret_cast<int32_t *>(output_indices_->GetData());

/*
Procedure for multi-core concurrent computing:
1. Call the CpuKernelUtils::GetCPUNum function to obtain the number of AI CPUs (max_core_num).
2. Calculate the computing data size on each AI CPU (per_unit_size) by dividing the total data size by the number of AI CPUs.
3. Implement the working process function shard of each compute unit, and compile the computing logic that needs to be
   concurrently executed in the function.
4. Call the CpuKernelUtils::ParallelFor function and input parameters such as the CpuKernelContext object (ctx), total
   data size (data_num), computing data size on each AI CPU (per_unit_size), and working process function shard of each
   compute unit. Then execute multi-core concurrent computing.
For example:
uint32_t min_core_num = 1;
int64_t max_core_num =
      std::max(min_core_num, CpuKernelUtils::GetCPUNum(ctx));
per_unit_size = data_num / max_core_num;
auto shard = [&](size_t start, size_t end) {
	for (size_t i = start; i < end; i++) {
	// Execution process      
	 ... ...
	}
};
CpuKernelUtils::ParallelFor(ctx, data_num, per_unit_size, shard);
*/

  auto shard_top_k = [&](size_t start, size_t end) {
    TopKForNVector(
        input_data + start * col_, values_data + start * k_,
        indices_data + start * k_ * (input_rank_ - dim_), end - start);
  };
  uint32_t ret = CpuKernelUtils::ParallelFor(
      ctx, row_, 1, shard_top_k);  // the minimum unit of segmentation is 1
  if (ret != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor failed");
    return KERNEL_STATUS_INNER_ERROR;
  }
  KERNEL_LOG_INFO("TopKCpuKernel::DoCompute end! ");
  return KERNEL_STATUS_OK;
}

template <typename T>
void TopKCpuKernel::TopKForNVector(T *input, T *value, int32_t *indices,
                                   int n) {
  std::shared_ptr<TensorShape> input_shape = input_tensor_->GetTensorShape();
  std::vector<int32_t> shape_end(input_rank_ - dim_);
  shape_end[input_rank_ - dim_ - 1] = 1;
  for (int32_t i = input_rank_ - dim_ - 2; i >= 0; i--) {
    shape_end[i] = shape_end[i + 1] * input_shape->GetDimSize(i + dim_ + 1);
  }
  for (int i = 0; i < n; i++) {
    if (largest_) {
      TopKMax::GetValueAndSelect(input + i * col_, value + i * k_, indices + i * k_ * (input_rank_ - dim_), k_);
      TopKMax::Select(input + i * col_, value + i * k_, indices + i * k_ * (input_rank_ - dim_), k_, col_);
    } else {
      TopKMin::GetValueAndSelect(input + i * col_, value + i * k_, indices + i * k_ * (input_rank_ - dim_), k_);
      TopKMin::Select(input + i * col_, value + i * k_, indices + i * k_ * (input_rank_ - dim_), k_, col_);
    }
    if (sorted_) {
      std::vector<ValueIndex<T>> data(k_);
      for (int j = 0; j < k_; j++) {
        data[j].value = *(value + i * k_ + j);
        data[j].index = *(indices + i * k_ * (input_rank_ - dim_) + j);
      }
      if (largest_) {
        std::sort(data.begin(), data.end(), CompareDescending<T>);
      } else {
        std::sort(data.begin(), data.end(), CompareAscending<T>);
      }
      for (int j = 0; j < k_; j++) {
        *(value + i * k_ + j) = data[j].value;
        int32_t index_old = data[j].index;
        for (size_t m = 0; m < shape_end.size(); m++) {
          *(indices + (i * k_ + j) * (input_rank_ - dim_) + m) = index_old / shape_end[m];
          index_old %= shape_end[m];
        }
      }
    } else {
      for (int j = 0; j < k_; j++) {
        int32_t index_old = *(indices + i * k_ * (input_rank_ - dim_) + j);
        for (size_t m = 0; m < shape_end.size(); m++) {
          *(indices + (i * k_ + j) * (input_rank_ - dim_) + m) = index_old / shape_end[m];
          index_old %= shape_end[m];
        }
      }
    }
  }
}

uint32_t TopKCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  // get x
  input_tensor_ = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(input_tensor_, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[0], name[x] failed");

  // get attr: dim
  std::shared_ptr<TensorShape> input_shape = input_tensor_->GetTensorShape();
  KERNEL_CHECK_NULLPTR(input_shape, KERNEL_STATUS_PARAM_INVALID,
                       "Get shape of input[0], name[x] failed");
  int32_t input_rank = input_shape->GetDims();
  if (input_rank < 1) {
    KERNEL_LOG_ERROR("Rank[%d] must be >= 1-D", input_rank);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  input_rank_ = input_rank;
  AttrValue *dim = ctx.GetAttr("dim");
  dim_ = (dim == nullptr) ? -1 : (dim->GetInt());
  dim_ = dim_ < 0 ? (input_rank + dim_) : dim_;
  KERNEL_CHECK_FALSE(((dim_ >= 0) && (dim_ < input_rank)),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Get Invalid attr[dim] value[%d]", dim_);

  // get col_ and row_
  col_ = 1;
  row_ = 1;
  for (int i = 0; i < input_rank; i++) {
    if (i < dim_) {
      row_ *= input_shape->GetDimSize(i);
    } else {
      col_ *= input_shape->GetDimSize(i);
    }
  }
  KERNEL_CHECK_FALSE((col_ > 0),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Col[%d] must be > 0", col_);
  KERNEL_CHECK_FALSE((row_ > 0),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Row[%d] must be > 0", row_);

  // get data_type_
  data_type_ = static_cast<DataType>(input_tensor_->GetDataType());

  // get k
  Tensor *k_tensor = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(k_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[1], name[k] failed");
  KERNEL_CHECK_NULLPTR(k_tensor->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input[1], name[k] failed");
  k_ = *static_cast<int32_t *>(k_tensor->GetData());
  if (k_ <= 0) {
    KERNEL_LOG_ERROR("K[%d] must be greater than 0", k_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (col_ < k_) {
    KERNEL_LOG_ERROR("Input must have at least %d columns, but got %d", k_,
                     col_);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // get attr: sorted
  AttrValue *sorted = ctx.GetAttr("sorted");
  sorted_ = (sorted == nullptr) ? true : (sorted->GetBool());

  // get attr: largest
  AttrValue *largest = ctx.GetAttr("largest");
  largest_ = (largest == nullptr) ? true : (largest->GetBool());

  // get values
  output_values_ = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_values_, KERNEL_STATUS_PARAM_INVALID,
                       "Get output[0], name[values] failed");

  // get indices
  output_indices_ = ctx.Output(1);
  KERNEL_CHECK_NULLPTR(output_indices_, KERNEL_STATUS_PARAM_INVALID,
                       "Get output[1], name[indices] failed");

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(TOPK, TopKCpuKernel);
}  // namespace aicpu
