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

#ifndef AI_CPU_PHILOX_RANDOM_DIS_H
#define AI_CPU_PHILOX_RANDOM_DIS_H

#include <algorithm>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "guarded_philox_random.h"
#include "kernel_util.h"
#include "utils/philox_random.h"
namespace aicpu {
namespace random {
template <class Distribution, typename T>
void fillTask(Distribution dist, PhiloxRandom gen, T* outputData,
              int64_t output_size, int32_t group_size, int64_t start_group,
              int64_t limit_group) {
  gen.Skip(start_group);
  int64_t offset = start_group * group_size;
  int64_t full_group = std::min(limit_group, output_size / group_size);
  for (int64_t index = start_group; index < full_group; index++) {
    auto samples = dist(&gen);
    std::copy(&samples[0], &samples[0] + group_size, outputData + offset);
    offset += group_size;
  }
  if (full_group < limit_group) {
    int64_t remaining_size = output_size - full_group * group_size;
    auto samples = dist(&gen);
    std::copy(&samples[0], &samples[0] + remaining_size, outputData + offset);
  }
}

template <typename Distribution>
class PhiloxRandomDist {
 public:
  using T = typename Distribution::ResultElementType;
  explicit PhiloxRandomDist(const CpuKernelContext& ctx) : generator_() {
    generator_.Init(ctx);
  }

  uint32_t generate(const CpuKernelContext& ctx, Tensor* output) {
    T* outputData = reinterpret_cast<T*>(output->GetData());
    auto output_size = output->NumElements();
    auto group_size = Distribution::kResultElementCount;
    if (group_size <= 0) {
      KERNEL_LOG_ERROR("group_size must greater 0,and group_size are [%ld] ",
                       group_size);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    auto group_count = (output_size + group_size - 1) / group_size;

    auto gen = generator_.ReserveRandomOutputs(output_size, 256);

    if (output_size >= kParallelDataNumSameShape) {
      uint32_t minCoreNum = 1;
      int64_t maxCoreNum =
          std::max(minCoreNum, CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
      auto shard = [&gen, output_size, group_size, outputData](
                       int64_t start_group, int64_t limit_group) {
        fillTask(Distribution(), gen, outputData, output_size, group_size,
                 start_group, limit_group);
      };

      KERNEL_HANDLE_ERROR(
          CpuKernelUtils::ParallelFor(ctx, group_count,
                                      group_count / maxCoreNum, shard),
          "PhiloxRandomDist parallelFor failed.");
    } else {
      fillTask(Distribution(), gen, outputData, output_size, group_size, 0,
               group_count);
    }
    return KERNEL_STATUS_OK;
  }

 private:
  const int64_t kParallelDataNumSameShape = 7 * 1024;
  GuardedPhiloxRandom generator_;
};
}  // namespace random
}  // namespace aicpu
#endif