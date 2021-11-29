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

#include "ceil.h"

#include <stdint.h>

#include "Eigen/Dense"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

namespace {
const char *CEIL = "Ceil";                // op name
const size_t K_CEIL_OUTPUT_DESC_NUM = 1;  // output size
const size_t K_CEIL_INPUT_NUM = 1;        // input size
}  // namespace

namespace aicpu {
uint32_t CeilCpuKernel::Compute(CpuKernelContext &ctx) {
  if (ctx.GetInputsSize() != K_CEIL_INPUT_NUM) {
    KERNEL_LOG_ERROR("Ceil node input size should be %zu, but get %zu",
                     K_CEIL_INPUT_NUM, ctx.GetInputsSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.GetOutputsSize() != K_CEIL_OUTPUT_DESC_NUM) {
    KERNEL_LOG_ERROR("Ceil node output size should be %zu, but get %zu",
                     K_CEIL_OUTPUT_DESC_NUM, ctx.GetOutputsSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *x = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(x, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[0], name[x] failed");
  Tensor *y = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(y, KERNEL_STATUS_PARAM_INVALID,
                       "Get output[0], name[y] failed");
  uint64_t data_size = x->GetDataSize();
  DataType data_type = DataType(x->GetDataType());
  uint32_t res = KERNEL_STATUS_OK;
  switch (data_type) {
    case DT_FLOAT16:
      res = ComputeCeil<Eigen::half>(x, y, data_size, ctx);
      break;
    case DT_FLOAT:
      res = ComputeCeil<float>(x, y, data_size, ctx);
      break;
    case DT_DOUBLE:
      res = ComputeCeil<double>(x, y, data_size, ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Ceil invalid input type [%s]", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (res != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CeilCpuKernel::ComputeCeil(Tensor *x, Tensor *y, uint64_t data_size,
                                    CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("CeilCpuKernel::ComputeCeil start");
  auto x_addr = x->GetData();
  auto y_addr = y->GetData();
  auto shard_ceil = [&](size_t start, size_t end) {
    // 1 represents row vector
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> map_x((T *)x_addr + start,
                                                         end - start);
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> map_y((T *)y_addr + start,
                                                         end - start);
    map_y = map_x.array().ceil().matrix();
  };
  // the minimum unit of segmentation is 1
  uint32_t ret =
      CpuKernelUtils::ParallelFor(ctx, data_size / sizeof(T), 1, shard_ceil);
  if (ret != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor failed");
    return KERNEL_STATUS_INNER_ERROR;
  }
  KERNEL_LOG_INFO("CeilCpuKernel::ComputeCeil end");
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(CEIL, CeilCpuKernel);
}  // namespace aicpu
