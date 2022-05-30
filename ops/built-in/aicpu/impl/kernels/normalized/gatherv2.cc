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

#include "gatherv2.h"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "securec.h"
#include "utils/kernel_util.h"

namespace {
const char *const kGatherV2 = "GatherV2";
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
const uint32_t kIndexTwo = 2;
} // namespace

namespace aicpu {
template <typename T, typename Index>
uint32_t DoGatherV2Compute(const CpuKernelContext &ctx) {
  Tensor* params = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(params, KERNEL_STATUS_PARAM_INVALID, "Get params failed.");
  Tensor* indices = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(indices, KERNEL_STATUS_PARAM_INVALID, "Get indices failed.");
  DataType indicesTwo_type = ctx.Input(kIndexTwo)->GetDataType();
  int64_t axis = 0;
  if (indicesTwo_type == DT_INT32) {
    auto axis_addr = static_cast<int32_t *>(ctx.Input(kIndexTwo)->GetData());
    if (axis_addr != NULL) {
      axis = static_cast<int64_t>(*axis_addr);
    }
  } else {
    auto axis_addr = static_cast<int64_t *>(ctx.Input(kIndexTwo)->GetData());
    if (axis_addr != NULL) {
      axis = static_cast<int64_t>(*axis_addr);
    }
  }
  if (axis < 0) {
    axis = params->GetTensorShape()->GetDims() + axis;
  }

  int64_t gather_dim_size = params->GetTensorShape()->GetDimSize(static_cast<int32_t>(axis));
  int64_t indices_num = indices->NumElements();

  int64_t outer_size = 1;
  int64_t inner_size = 1;

  for (int64_t i = 0; i < axis; ++i) {
    outer_size *= params->GetTensorShape()->GetDimSize(static_cast<int32_t>(i));
  }
  for (int64_t i = axis + 1; i < params->GetTensorShape()->GetDims(); ++i) {
    inner_size *= params->GetTensorShape()->GetDimSize(static_cast<int32_t>(i));
  }

  int64_t slice_size = inner_size * static_cast<int64_t>(sizeof(T));
  auto params_base = static_cast<T *>(params->GetData());
  KERNEL_CHECK_NULLPTR(params_base, KERNEL_STATUS_PARAM_INVALID, "Get params_base failed.");
  auto indices_data = static_cast<Index *>(indices->GetData());
  KERNEL_CHECK_NULLPTR(indices_data, KERNEL_STATUS_PARAM_INVALID, "Get indices_data failed.");
  auto out_base = static_cast<T *>(ctx.Output(0)->GetData());
  KERNEL_CHECK_NULLPTR(out_base, KERNEL_STATUS_PARAM_INVALID, "Get output failed.");
  for (int64_t i = 0; i < outer_size; ++i) {
    for (int64_t j = 0; j < indices_num; ++j) {
      auto params_idx = (i * gather_dim_size + indices_data[j]) * inner_size;
      auto out_idx = (i * indices_num + j) * inner_size;
      auto cpret = memcpy_s(out_base + out_idx, slice_size, params_base + params_idx, slice_size);
      KERNEL_CHECK_FALSE((cpret == EOK), KERNEL_STATUS_INNER_ERROR, "memcpy_s to output failed.");
    }
  }

  return KERNEL_STATUS_OK;
}

template <typename IndicesType>
uint32_t IndicesCompute(CpuKernelContext &ctx) {
  DataType params_type = ctx.Input(0)->GetDataType();
  std::map<int, std::function<uint32_t(CpuKernelContext &)>> calls;
  calls[DT_FLOAT16] = DoGatherV2Compute<Eigen::half, IndicesType>;
  calls[DT_FLOAT] = DoGatherV2Compute<float, IndicesType>;
  calls[DT_DOUBLE] = DoGatherV2Compute<double, IndicesType>;
  calls[DT_INT8] = DoGatherV2Compute<int8_t, IndicesType>;
  calls[DT_INT16] = DoGatherV2Compute<int16_t, IndicesType>;
  calls[DT_INT32] = DoGatherV2Compute<int32_t, IndicesType>;
  calls[DT_INT64] = DoGatherV2Compute<int64_t, IndicesType>;
  calls[DT_UINT8] = DoGatherV2Compute<uint8_t, IndicesType>;
  calls[DT_UINT16] = DoGatherV2Compute<uint16_t, IndicesType>;
  calls[DT_UINT32] = DoGatherV2Compute<uint32_t, IndicesType>;
  calls[DT_UINT64] = DoGatherV2Compute<uint64_t, IndicesType>;
  calls[DT_COMPLEX64] = DoGatherV2Compute<std::complex<float>, IndicesType>;
  calls[DT_COMPLEX128] = DoGatherV2Compute<std::complex<double>, IndicesType>;
  return calls[params_type](ctx);
}

uint32_t GatherV2CpuKernel::GetInputAndCheck(const CpuKernelContext &ctx) {
  Tensor* params = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(params, KERNEL_STATUS_PARAM_INVALID,
                       "Get params failed.");
  DataType indicesTwo_type = ctx.Input(kIndexTwo)->GetDataType();
  int64_t axis = 0;
  if (indicesTwo_type == DT_INT32) {
    auto axis_addr = static_cast<int32_t *>(ctx.Input(kIndexTwo)->GetData());
    if (axis_addr != NULL) {
      axis = static_cast<int64_t>(*axis_addr);
    }
  } else {
    auto axis_addr = static_cast<int64_t *>(ctx.Input(kIndexTwo)->GetData());
    if (axis_addr != NULL) {
      axis = static_cast<int64_t>(*axis_addr);
    }
  }
  auto params_dims = params->GetTensorShape()->GetDims();
  int64_t min_params_dim = axis < 0 ? -axis : axis + 1;
  KERNEL_CHECK_FALSE((params_dims >= min_params_dim), KERNEL_STATUS_PARAM_INVALID,
                     "Shape must be at least rank [%d] but is rank [%d]",
                     min_params_dim, params_dims);

  auto batch_dims_ptr = ctx.GetAttr("batch_dims");
  KERNEL_CHECK_NULLPTR(batch_dims_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get batch dims failed.");
  int64_t batch_dims = batch_dims_ptr->GetInt();
  KERNEL_CHECK_FALSE((batch_dims >= 0), KERNEL_STATUS_PARAM_INVALID,
                     "Batch_dims must be at least 0 but is [%d]", batch_dims);
  return KERNEL_STATUS_OK;
}

uint32_t GatherV2CpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Check GatherV2 params failed.");

  auto ret = GetInputAndCheck(ctx);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret, "GetInputAndCheck failed");
  Tensor *indices = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(indices, KERNEL_STATUS_PARAM_INVALID, "Get indices failed.");
  DataType indices_type = indices->GetDataType();
  if (indices_type == DT_INT32) {
    ret = IndicesCompute<int32_t>(ctx);
  } else {
    ret = IndicesCompute<int64_t>(ctx);
  }
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret,
                     "Compute failed");
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kGatherV2, GatherV2CpuKernel);
} // namespace aicpu