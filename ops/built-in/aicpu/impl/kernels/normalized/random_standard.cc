/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

#include "random_standard.h"

#include <algorithm>
#include <random>

#include "random/guarded_philox_random.h"
#include "random/philox_random_dist.h"
#include "random/random_distributions.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "utils/philox_random.h"

namespace {
const char *const kRandomStandardNormal = "RandomStandardNormal";
}

namespace aicpu {
uint32_t RandomStandardCpuKernel::Compute(CpuKernelContext &ctx) {
  auto attr_value = ctx.GetAttr("dtype");
  KERNEL_CHECK_NULLPTR(attr_value, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr[dtype] failed")
  auto data_type = static_cast<DataType>(attr_value->GetDataType());
  Tensor *output = ctx.Output(kFirstOutputIndex);
  KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID, "Get output failed")
  if (data_type != output->GetDataType()) {
    KERNEL_LOG_ERROR(
        "RandomStandard kernel data type not matched, dtype is [%u], "
        "out_data_type is [%u].",
        data_type, output->GetDataType());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // choose random data generate function depend on dataType
  switch (data_type) {
    case DT_FLOAT16:
      Generate<Eigen::half>(ctx, output);
      break;
    case DT_FLOAT:
      Generate<float>(ctx, output);
      break;
    case DT_DOUBLE:
      Generate<double>(ctx, output);
      break;
    default:
      KERNEL_LOG_ERROR("RandomStandard kernel data type [%u] not support.",
                       data_type);
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void RandomStandardCpuKernel::Generate(const CpuKernelContext &ctx,
                                       Tensor *output) {
  random::PhiloxRandomDist<random::NormalDistribution<PhiloxRandom, T>>
      philoxRandomDist(ctx);
  philoxRandomDist.generate(ctx, output);
}

REGISTER_CPU_KERNEL(kRandomStandardNormal, RandomStandardCpuKernel);
}  // namespace aicpu
