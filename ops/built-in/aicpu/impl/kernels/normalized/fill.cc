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

#include "fill.h"
#include <set>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kFill = "Fill";

#define FILL_CALCULATE_DIMS_CASE(DTYPE, TYPE)                                 \
  case (DTYPE): {                                                             \
    if (CalcDims<TYPE>(dims_tensor, dims) != KERNEL_STATUS_OK) {              \
      KERNEL_LOG_ERROR("Fill kernel calculate dims failed");                  \
      return KERNEL_STATUS_PARAM_INVALID;                                     \
    }                                                                         \
    break;                                                                    \
  }

#define FILL_GENERATE_CASE(DTYPE, TYPE)                                       \
  case (DTYPE): {                                                             \
    auto value = *(reinterpret_cast<const TYPE *>(value_tensor->GetData()));  \
    if (AddrAlignedCheck(output->GetData())) {                                \
      FILL_EIGEN_TENSOR_ASSIGN_CASE(TYPE, Eigen::Aligned);                    \
    } else {                                                                  \
      FILL_EIGEN_TENSOR_ASSIGN_CASE(TYPE, Eigen::Unaligned);                  \
    }                                                                         \
    break;                                                                    \
  }

#define FILL_EIGEN_TENSOR_ASSIGN_CASE(TYPE, ALIGNMENT_TYPE)  do {             \
    Eigen::TensorMap<Eigen::Tensor<TYPE, 1>, ALIGNMENT_TYPE> eigen_output(    \
      static_cast<TYPE *>(output->GetData()),                                 \
      output->GetTensorShape()->NumElements());                               \
    eigen_output.setConstant(value);                                          \
  } while(0)
}

namespace aicpu {
uint32_t FillCpuKernel::Compute(CpuKernelContext &ctx) {
  std::vector<int64_t> dims;
  Tensor *dims_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(dims_tensor, KERNEL_STATUS_PARAM_INVALID, "Get dims input failed")
  auto dims_dtype = dims_tensor->GetDataType();
  switch (dims_dtype) {
    FILL_CALCULATE_DIMS_CASE(DT_INT32, int32_t)
    FILL_CALCULATE_DIMS_CASE(DT_INT64, int64_t)
    default:
      KERNEL_LOG_ERROR("Fill kernel dims data_type [%u] not support, support data_types: DT_INT32, DT_INT64",
                       dims_dtype);
      return KERNEL_STATUS_PARAM_INVALID;
  }

  Tensor *value_tensor = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(value_tensor, KERNEL_STATUS_PARAM_INVALID, "Get value input failed")
  KERNEL_CHECK_NULLPTR(value_tensor->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get value input data failed")
  KERNEL_CHECK_NULLPTR(value_tensor->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get value input shape failed")
  if (!value_tensor->GetTensorShape()->GetDimSizes().empty()) {
    KERNEL_LOG_ERROR("Fill kernel value input is not a scalar.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *output = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID, "Get output failed")
  KERNEL_CHECK_NULLPTR(output->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
  KERNEL_CHECK_NULLPTR(output->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get output shape failed")
  if (output->GetTensorShape()->GetDimSizes() != dims) {
    KERNEL_LOG_ERROR("Fill kernel output shape not matched.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto input_dtype = value_tensor->GetDataType();
  auto output_dtype = output->GetDataType();
  if (input_dtype != output_dtype) {
    KERNEL_LOG_ERROR("Fill kernel data type not matched, value input dtype [%u], output dtype [%u].",
                     input_dtype, output_dtype);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  switch (output_dtype) {
    FILL_GENERATE_CASE(DT_INT8, int8_t)
    FILL_GENERATE_CASE(DT_UINT8, uint8_t)
    FILL_GENERATE_CASE(DT_INT16, int16_t)
    FILL_GENERATE_CASE(DT_UINT16, uint16_t)
    FILL_GENERATE_CASE(DT_INT32, int32_t)
    FILL_GENERATE_CASE(DT_UINT32, uint32_t)
    FILL_GENERATE_CASE(DT_INT64, int64_t)
    FILL_GENERATE_CASE(DT_UINT64, uint64_t)
    FILL_GENERATE_CASE(DT_BOOL, bool)
    FILL_GENERATE_CASE(DT_FLOAT16, Eigen::half)
    FILL_GENERATE_CASE(DT_FLOAT, float)
    FILL_GENERATE_CASE(DT_DOUBLE, double)
    default:
      KERNEL_LOG_ERROR("Fill kernel data type [%u] not support", output_dtype);
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t FillCpuKernel::CalcDims(const Tensor *dims_tensor, std::vector<int64_t> &dims) {
  uint64_t data_num = dims_tensor->GetDataSize() / sizeof(T);
  if (data_num == 0) {
    KERNEL_LOG_INFO("Fill kernel: dims is empty, no need to fill");
    return KERNEL_STATUS_OK;
  }

  KERNEL_CHECK_NULLPTR(dims_tensor->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get dims data failed")
  for (uint64_t i = 0; i < data_num; i++) {
    auto dim = *(reinterpret_cast<const T *>(dims_tensor->GetData()) + i);
    if (dim < 0) {
      KERNEL_LOG_ERROR("Fill kernel: input dim [%llu] is negative, value=[%lld]", i, static_cast<int64_t>(dim));
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (dim == 0) {
      KERNEL_LOG_INFO("Fill kernel: input dim [%llu] is zero", i);
      dims.clear();
      break;
    }
    dims.emplace_back(dim);
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kFill, FillCpuKernel);
}  // namespace aicpu
