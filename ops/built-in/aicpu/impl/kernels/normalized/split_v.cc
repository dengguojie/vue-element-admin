/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "split_v.h"
#include "utils/kernel_util.h"

namespace {
const char *kSplitV = "SplitV";
}

namespace aicpu {
uint32_t SplitVCpuKernel::CheckAndInitParams(CpuKernelContext &ctx) {
  // get Attr num_split
  AttrValue *num_split_ptr = ctx.GetAttr("num_split");
  KERNEL_CHECK_NULLPTR(num_split_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr num_split failed.");
  num_split_ = num_split_ptr->GetInt();
  KERNEL_CHECK_FALSE((num_split_ >= 1), KERNEL_STATUS_PARAM_INVALID,
                     "Attr num_split must >= 1, but got attr num_split[%lld]", num_split_);
  // get input split_dim
  Tensor *split_dim_ptr = ctx.Input(2);
  KERNEL_CHECK_NULLPTR(split_dim_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input split_dim failed.");
  auto split_dim_shape_ptr = split_dim_ptr->GetTensorShape();
  KERNEL_CHECK_NULLPTR(split_dim_shape_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input split_dim shape failed.");
  KERNEL_CHECK_FALSE((split_dim_shape_ptr->GetDims() == 0), KERNEL_STATUS_PARAM_INVALID,
                     "Input split_dim should be a scalar integer, but got rank[%lld]", split_dim_shape_ptr->GetDims());
  KERNEL_CHECK_FALSE((split_dim_ptr->GetDataType() == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                     "Input split_dim data type must be DT_INT32, but got data type[%s]",
                     DTypeStr(split_dim_ptr->GetDataType()).c_str());
  auto split_dim_data_ptr = split_dim_ptr->GetData();
  KERNEL_CHECK_NULLPTR(split_dim_data_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input split_dim data failed.");
  split_dim_ = *(reinterpret_cast<int32_t *>(split_dim_data_ptr));
  KERNEL_CHECK_FALSE((split_dim_ >= 0), KERNEL_STATUS_PARAM_INVALID,
                     "Input split_dim must >= 0, but got input split_dim[%lld]", split_dim_);
  // get input value
  Tensor *value_ptr = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(value_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input value failed.");
  value_data_ptr_ = value_ptr->GetData();
  KERNEL_CHECK_NULLPTR(value_data_ptr_, KERNEL_STATUS_PARAM_INVALID,
                       "Get input value data failed.");
  auto value_shape_ptr = value_ptr->GetTensorShape();
  KERNEL_CHECK_NULLPTR(value_shape_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input value shape failed.");
  int64_t value_dim = value_shape_ptr->GetDims();
  KERNEL_CHECK_FALSE(value_dim > split_dim_, KERNEL_STATUS_PARAM_INVALID,
                     "Dim of Input value must greater than split_dim, value dim is [%d], split_dim is [%d].",
                     value_dim, num_split_);
  int64_t real_dim = value_shape_ptr->GetDimSize(split_dim_);
  value_shape_vec_ = value_shape_ptr->GetDimSizes();
  data_type_ = value_ptr->GetDataType();
  value_num_ = value_ptr->NumElements();
  // get input size_splits
  Tensor *size_splits_ptr = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(size_splits_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input size_splits failed.");
  auto size_splits_shape_ptr = size_splits_ptr->GetTensorShape();
  KERNEL_CHECK_NULLPTR(size_splits_shape_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input size_splits shape failed.");
  int64_t size_splits_dim = size_splits_shape_ptr->GetDims();
  KERNEL_CHECK_FALSE((size_splits_dim == 1), KERNEL_STATUS_PARAM_INVALID,
                     "Input size_splits should be a 1-D Tensor, but got rank[%d].",
                     size_splits_dim);
  int64_t size_split_num = size_splits_shape_ptr->GetDimSize(0);
  KERNEL_CHECK_FALSE((size_split_num == num_split_), KERNEL_STATUS_PARAM_INVALID,
                     "Size of Input size_splits should be equal to Attr num_split, but got [%d], num_split is [%d].",
                     size_split_num, num_split_);
  DataType size_splits_type = size_splits_ptr->GetDataType();
  KERNEL_CHECK_FALSE(((size_splits_type == DT_INT32) || (size_splits_type == DT_INT64)),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Input size_splits data type must be DT_INT32 or DT_INT64, but got data type[%s]",
                     DTypeStr(size_splits_type).c_str());
  auto size_splits_data_ptr = size_splits_ptr->GetData();
  KERNEL_CHECK_NULLPTR(size_splits_data_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input size_splits data failed.");
  if (size_splits_type == DT_INT32) {
    KERNEL_CHECK_FALSE((GetSizeSplits<int32_t>(size_splits_data_ptr, real_dim) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "GetSizeSplits failed.");
  } else {
    KERNEL_CHECK_FALSE((GetSizeSplits<int64_t>(size_splits_data_ptr, real_dim) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "GetSizeSplits failed.");
  }
  // get output data
  output_ptr_vec_.resize(num_split_);
  for (int64_t i = 0; i < num_split_; i++) {
    Tensor *output_ptr = ctx.Output(i);
    KERNEL_CHECK_NULLPTR(output_ptr, KERNEL_STATUS_PARAM_INVALID,
                         "Get output [%d] failed.", i);
    auto output_data_ptr = output_ptr->GetData();
    KERNEL_CHECK_NULLPTR(output_data_ptr, KERNEL_STATUS_PARAM_INVALID,
                         "Get output data [%d] failed.", i);
    output_ptr_vec_[i] = output_data_ptr;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SplitVCpuKernel::GetSizeSplits(void *size_splits_data_ptr, int64_t real_dim) {
  size_splits_.resize(num_split_);
  T *size_splits_data = reinterpret_cast<T *>(size_splits_data_ptr);
  int64_t unique_one_dim = -1;
  T total_dim = 0;
  for (int64_t i = 0; i < num_split_; i++) {
    T cur_dim = size_splits_data[i];
    if (cur_dim == -1) {
      KERNEL_CHECK_FALSE(unique_one_dim == -1,
                         KERNEL_STATUS_PARAM_INVALID,
                         "There should only one element in size_splits can be -1.");
      unique_one_dim = i;
    } else {
      total_dim += cur_dim;
    }
    size_splits_[i] = static_cast<int64_t>(cur_dim);
  }
  KERNEL_CHECK_FALSE(((unique_one_dim == -1) && (total_dim == real_dim)) ||
                     ((unique_one_dim >= 0) && (total_dim <= real_dim)),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Determined shape must either match input shape along split_dim exactly if"
                     "fully specified, or be less than the size of input along split_dim if not"
                     "fully specified. Got [%d], real_dim is [%d].", total_dim, real_dim);
  if (unique_one_dim >= 0) {
    size_splits_[unique_one_dim] = static_cast<int64_t>(real_dim - total_dim);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SplitVCpuKernel::SplitVWithOneOutput(T *input_data_ptr,
                                              std::vector<T *> output_data_vec) {
  int64_t copy_size = value_num_ * sizeof(T);
  auto mem_ret = memcpy_s(output_data_vec[0], copy_size, input_data_ptr, copy_size);
  KERNEL_CHECK_FALSE((mem_ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                     "Memcpy size[%zu] from input value to output[0] failed.",
                     copy_size);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SplitVCpuKernel::SplitVWithDimZero(T *input_data_ptr,
                                            std::vector<T *> output_data_vec) {
  int64_t copy_num = value_num_ / value_shape_vec_[0];
  T *input_copy_ptr = input_data_ptr;
  for (int32_t i = 0; i < num_split_; i++) {
    int64_t copy_size_per = size_splits_[i] * copy_num;
    int64_t copy_size = copy_size_per * sizeof(T);
    auto mem_ret = memcpy_s(output_data_vec[i], copy_size, input_copy_ptr, copy_size);
    KERNEL_CHECK_FALSE((mem_ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                       "Memcpy size[%zu] from input value to output[%d] failed.",
                       copy_size, i);
    input_copy_ptr += copy_size_per;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SplitVCpuKernel::SplitVCompute(T *input_data_ptr,
                                        std::vector<T *> output_data_vec) {
  int64_t prefix = 1;
  for (int32_t i = 0; i < split_dim_; i++) {
    prefix *= value_shape_vec_[i];
  }
  int64_t midfix = value_shape_vec_[split_dim_];
  int64_t subfix = 1;
  for (size_t i = split_dim_ + 1; i < value_shape_vec_.size(); i++) {
    subfix *= value_shape_vec_[i];
  }
  int64_t offset = 0;
  for (int64_t i = 0; i < num_split_; i++) {
    T *output_data_ptr = output_data_vec[i];
    T *input_copy_ptr = input_data_ptr + offset;
    int64_t copy_num = subfix * size_splits_[i];
    int64_t copy_size = copy_num * sizeof(T);
    for (int64_t j = 0; j < prefix; j++) {
      auto mem_ret = memcpy_s(output_data_ptr, copy_size, input_copy_ptr, copy_size);
      KERNEL_CHECK_FALSE((mem_ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                         "Memcpy size[%zu] from input value to output[%d] failed.",
                         copy_size, i);
      input_copy_ptr += (subfix * midfix);
      output_data_ptr += copy_num;
    }
    offset += copy_num;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SplitVCpuKernel::DoCompute(CpuKernelContext &ctx) {
  T *input_data_ptr = reinterpret_cast<T *>(value_data_ptr_);
  std::vector<T *> output_data_vec;
  output_data_vec.resize(num_split_);
  for (int64_t i = 0; i < num_split_; i++) {
    output_data_vec[i] = reinterpret_cast<T *>(output_ptr_vec_[i]);
  }

  if (num_split_ == 1) {
    KERNEL_CHECK_FALSE((SplitVWithOneOutput<T>(input_data_ptr, output_data_vec) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "SplitVWithOneOutput failed.");
    return KERNEL_STATUS_OK;
  }
  if (split_dim_ == 0) {
    KERNEL_CHECK_FALSE((SplitVWithDimZero<T>(input_data_ptr, output_data_vec) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "SplitVWithDimZero failed.");
    return KERNEL_STATUS_OK;
  }
  KERNEL_CHECK_FALSE((SplitVCompute<T>(input_data_ptr, output_data_vec) == KERNEL_STATUS_OK),
                     KERNEL_STATUS_PARAM_INVALID,
                     "SplitV Compute failed.");
  return KERNEL_STATUS_OK;
}

uint32_t SplitVCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_CHECK_FALSE((CheckAndInitParams(ctx) == KERNEL_STATUS_OK),
                     KERNEL_STATUS_PARAM_INVALID, "CheckAndInitParams failed.");
  switch (data_type_) {
    case DT_FLOAT16:
      return DoCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return DoCompute<float>(ctx);
    case DT_DOUBLE:
      return DoCompute<double>(ctx);
    case DT_BOOL:
      return DoCompute<bool>(ctx);
    case DT_INT8:
      return DoCompute<int8_t>(ctx);
    case DT_INT16:
      return DoCompute<int16_t>(ctx);
    case DT_INT32:
      return DoCompute<int32_t>(ctx);
    case DT_INT64:
      return DoCompute<int64_t>(ctx);
    case DT_UINT8:
      return DoCompute<uint8_t>(ctx);
    case DT_UINT16:
      return DoCompute<uint16_t>(ctx);
    case DT_UINT32:
      return DoCompute<uint32_t>(ctx);
    case DT_UINT64:
      return DoCompute<uint64_t>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupport datatype[%s]", DTypeStr(data_type_).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_CPU_KERNEL(kSplitV, SplitVCpuKernel);
}  // namespace aicpu