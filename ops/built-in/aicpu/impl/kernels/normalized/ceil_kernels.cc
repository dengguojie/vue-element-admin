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

#include "ceil_kernels.h"

#include <securec.h>
#include <stdint.h>

#include "Eigen/Dense"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "cpu_kernel_utils.h"

namespace {
const char *CEIL = "Ceil";                // op name
const size_t K_CEIL_OUTPUT_DESC_NUM = 1;  // output dims
const size_t K_CEIL_INPUT_NUM = 1;        // input dims
}  // namespace

namespace aicpu {
uint32_t CeilCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("CeilCpuKernel::Compute begin.");

  if ((ctx.GetInputsSize() != K_CEIL_INPUT_NUM) ||
      (ctx.GetOutputsSize() != K_CEIL_OUTPUT_DESC_NUM)) {
    KERNEL_LOG_ERROR(
        "Unexpected Ceil node, node input size: %zu, node output size: %zu, "
        "node name: %s",
        ctx.GetInputsSize(), ctx.GetOutputsSize(), ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *x = ctx.Input(0);
  if (x == nullptr) {
    KERNEL_LOG_ERROR("Ceil input x is null");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *y = ctx.Output(0);
  if (y == nullptr) {
    KERNEL_LOG_ERROR("Ceil output y is null");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  uint64_t dataSize = x->GetDataSize();
  KERNEL_LOG_INFO("Ceil input size:%llu.", dataSize);
  DataType dataType = DataType(x->GetDataType());
  KERNEL_LOG_INFO("Ceil input type:%d", dataType);

  switch (dataType) {
    case DT_FLOAT16:
      ComputeCeil<Eigen::half>(x, y, dataSize, ctx);
      break;
    case DT_FLOAT:
      ComputeCeil<float>(x, y, dataSize, ctx);
      break;
    case DT_DOUBLE:
      ComputeCeil<double>(x, y, dataSize, ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Ceil invalid input type:%d", dataType);
      break;
  }

  KERNEL_LOG_INFO("CeilCpuKernel::Compute end.");
  return KERNEL_STATUS_OK;
}

template <typename T>
void CeilCpuKernel::ComputeCeil(Tensor *x, Tensor *y, uint64_t dataSize, CpuKernelContext &ctx) {
  auto xAddr = x->GetData();
  auto yAddr = y->GetData();
  // 1 represents row vector
  auto shard_ceil = [&](size_t start, size_t end) {
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> mapX((T *)xAddr + start,
                                                         end - start);
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> mapY((T *)yAddr + start,
                                                         end - start);
    mapY = mapX.array().ceil().matrix();
  };
  CpuKernelUtils::ParallelFor(ctx, dataSize / sizeof(T), 1, shard_ceil);
  KERNEL_LOG_INFO("CeilCpuKernel::Compute success.");
}

REGISTER_CPU_KERNEL(CEIL, CeilCpuKernel);
}  // namespace aicpu
