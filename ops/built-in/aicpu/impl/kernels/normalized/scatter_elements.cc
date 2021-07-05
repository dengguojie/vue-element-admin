/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "scatter_elements.h"
#include <atomic>
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const int32_t kInputNum = 3;
const int32_t kOutputNum = 1;
const int32_t KSplitSize = 64 * 1024;
const char *kScatterElements = "ScatterElements";
#define DO_COMPUTE_CASE(DTYPE, TYPE, ITYPE, CTX)   \
  case (DTYPE): {                                  \
    if(ITYPE == DT_INT32) {                        \
      return DoCompute<TYPE, int32_t>(ctx);        \
    } else {                                       \
      return DoCompute<TYPE, int64_t>(ctx);        \
    }                                              \
  }
}

namespace aicpu {
uint32_t ScatterElementsCpuKernel::Compute(CpuKernelContext &ctx) {
  // check param
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "ScatterElements check input or output is failed");
  Tensor *input_data = ctx.Input(0);
  Tensor *input_indices = ctx.Input(1);
  auto data_type = input_data->GetDataType();
  auto data_type_indices = input_indices->GetDataType();
  KERNEL_CHECK_FALSE(
      (data_type_indices == DT_INT32 || data_type_indices == DT_INT64),
      KERNEL_STATUS_PARAM_INVALID, "Input[1] data type[%s] is unsupported",
      DTypeStr(data_type_indices).c_str());
  switch (data_type) {
    DO_COMPUTE_CASE(DT_FLOAT16, Eigen::half, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_FLOAT, float, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_DOUBLE, double, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_BOOL, bool, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_INT8, int8_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_INT16, int16_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_INT32, int32_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_INT64, int64_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_UINT8, uint8_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_UINT16, uint16_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_UINT32, uint32_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_UINT64, uint64_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, data_type_indices,
                    ctx);
  default:
    std::string err_msg = ConcatString(
                  "Input[0] data type[",DTypeStr(data_type),"] is unsupported.",
                  "It should be float16|float|double|bool|int8|int16|int32|int64|uint8|",
                  "unint32|unit64|complex16|complex32");
    KERNEL_LOG_ERROR("%s", err_msg.c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T, typename TI>
uint32_t ScatterElementsCpuKernel::DoCompute(CpuKernelContext &ctx) {
  // check parameters basic attribution are valid
  auto input_x1 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  KERNEL_CHECK_NULLPTR(input_x1, KERNEL_STATUS_PARAM_INVALID,
                       "Get input original value failed");
  auto input_x2 = reinterpret_cast<TI *>(ctx.Input(1)->GetData());
  KERNEL_CHECK_NULLPTR(input_x2, KERNEL_STATUS_PARAM_INVALID,
                       "Get input indices value failed");
  auto input_x3 = reinterpret_cast<T *>(ctx.Input(2)->GetData());
  KERNEL_CHECK_NULLPTR(input_x3, KERNEL_STATUS_PARAM_INVALID,
                       "Get input updates value failed");
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  KERNEL_CHECK_NULLPTR(output_y, KERNEL_STATUS_PARAM_INVALID,
                       "Get output failed");
  auto shape_x1 = ctx.Input(0)->GetTensorShape();
  KERNEL_CHECK_NULLPTR(shape_x1, KERNEL_STATUS_PARAM_INVALID,
                       "Get input original shape failed");
  auto shape_x2 = ctx.Input(1)->GetTensorShape();
  KERNEL_CHECK_NULLPTR(shape_x2, KERNEL_STATUS_PARAM_INVALID,
                       "Get input indices shape failed");
  auto shape_x3 = ctx.Input(2)->GetTensorShape();
  KERNEL_CHECK_NULLPTR(shape_x3, KERNEL_STATUS_PARAM_INVALID,
                       "Get input updates shape failed");
  auto shape_y = ctx.Output(0)->GetTensorShape();
  KERNEL_CHECK_NULLPTR(shape_y, KERNEL_STATUS_PARAM_INVALID,
                       "Get output shape failed");
  AttrValue *axis = ctx.GetAttr("axis");
  int64_t axis_value = (axis == nullptr) ? 0 : axis->GetInt();
  // Get and check 3 input dim info
  int64_t value_dim_num_x1 = shape_x1->GetDims();
  int64_t value_dim_num_x2 = shape_x2->GetDims();
  int64_t value_dim_num_x3 = shape_x3->GetDims();
  std::vector<int64_t> value_dim_x1 = shape_x1->GetDimSizes();
  std::vector<int64_t> value_dim_x2 = shape_x2->GetDimSizes();
  std::vector<int64_t> value_dim_x3 = shape_x3->GetDimSizes();
  KERNEL_CHECK_FALSE(
      (value_dim_num_x1 == value_dim_num_x2 &&
       value_dim_num_x2 == value_dim_num_x3),
      KERNEL_STATUS_PARAM_INVALID,
      "3 inputs dim values are different; data:%lld,indices:%lld,update:%lld",
      value_dim_num_x1, value_dim_num_x2, value_dim_num_x3);
  KERNEL_CHECK_FALSE(
      (axis_value >= value_dim_num_x1 * -1 && axis_value < value_dim_num_x1),
      KERNEL_STATUS_PARAM_INVALID, "Axis_value %lld is out of range %lld",
      axis_value, value_dim_num_x1);
  std::vector<int64_t> data_dim_vec;
  std::vector<int64_t> index_dim_vec;
  int64_t sub_data_fix = 1;
  int64_t sub_index_fix = 1;
  for (int64_t i = value_dim_num_x2 - 1; i >= 0; --i) {
    KERNEL_CHECK_FALSE((value_dim_x1[i] >= value_dim_x2[i] &&
                        value_dim_x2[i] == value_dim_x3[i] &&
                        value_dim_x3[i] > 0),
                       KERNEL_STATUS_PARAM_INVALID,
                       "The %d dimension verfication failed:input0[%d],input1[%d],input2[%d]",
                       i, value_dim_x1, value_dim_x2, value_dim_x3);
    if (i > 0) {
      sub_data_fix *= value_dim_x1[i];
      data_dim_vec.push_back(sub_data_fix);
      sub_index_fix *= value_dim_x2[i];
      index_dim_vec.push_back(sub_index_fix);
    }
  }
  axis_value = axis_value < 0 ? axis_value + value_dim_num_x1 : axis_value;
  int64_t axis_dim_value = shape_x1->GetDimSize(axis_value);
  int64_t update_value_num = ctx.Input(1)->NumElements();
  int64_t total_value_num = ctx.Input(0)->NumElements();
  // using input to initial output
  std::atomic<int32_t> work_ret(KERNEL_STATUS_OK);
  int64_t initial_size = total_value_num * sizeof(T);
  int64_t max_thread_num = initial_size / KSplitSize;
  if (max_thread_num > total_value_num || max_thread_num == 0) {
    max_thread_num = total_value_num;
  }
  int64_t per_core_size = total_value_num / max_thread_num * sizeof(T);
  int64_t last_core_size = total_value_num % max_thread_num * sizeof(T) +
                           per_core_size;
  auto shard_copy = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      int64_t core_size = (i == max_thread_num - 1) ?
                          last_core_size : per_core_size;
      int64_t ptr_offset = i * (total_value_num / max_thread_num);
      if (ptr_offset >= total_value_num) {
        work_ret = KERNEL_STATUS_PARAM_INVALID;
        KERNEL_LOG_ERROR("Pointer offset %lld more than %lld which is overflow",
                         ptr_offset, total_value_num);
        return;
      }
      auto out = output_y + ptr_offset;
      auto in = input_x1 + ptr_offset;
      auto mem_ret = memcpy_s(out, core_size, in, core_size);
      if (mem_ret != EOK) {
        KERNEL_LOG_ERROR("Initial memory copy failed[%d].Offerst is %d, copy size is %d",
                         mem_ret, ptr_offset, core_size);
        work_ret = KERNEL_STATUS_INNER_ERROR;
        return;
      }
    }
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, max_thread_num, 1, shard_copy),
                      "Initial output data failed!")
  if (work_ret != KERNEL_STATUS_OK) {
    return work_ret;
  }
  // update data according to indices
  auto shard_update = [&](size_t start, size_t end) {
    for(int64_t i = start; i < end; ++i) {
      int64_t remain_index = i;
      int64_t index_value = 0;
      int64_t counter = 0;
      int64_t input_x2_value = input_x2[i] < 0 ?
                               input_x2[i] + axis_dim_value : input_x2[i];
      // check update indices are in bounds
      if (input_x2_value >= axis_dim_value || input_x2_value < 0) {
        work_ret = KERNEL_STATUS_PARAM_INVALID;
        KERNEL_LOG_ERROR("Indices value %lld is out of bounds %lld",
                        input_x2[i], axis_dim_value);
        return;
      }
      for (int64_t j = index_dim_vec.size() - 1; j >= 0; --j) {
        index_value += ((counter == axis_value ? input_x2_value:
                        remain_index / index_dim_vec[j]) * data_dim_vec[j]);
        remain_index %= index_dim_vec[j];
        ++counter;
      }
      index_value += (counter == axis_value ? input_x2_value : remain_index);
      if (index_value>=total_value_num) {
        work_ret = KERNEL_STATUS_PARAM_INVALID;
        KERNEL_LOG_ERROR("Update index %lld more than %lld which is overflow",
                         index_value, total_value_num);
        return;
      }
      output_y[index_value] = input_x3[i];
    }
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, update_value_num, 1, shard_update),
                      "Update process error!")
  if(work_ret != KERNEL_STATUS_OK) {
    return work_ret;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kScatterElements, ScatterElementsCpuKernel);
} // namespace aicpu
