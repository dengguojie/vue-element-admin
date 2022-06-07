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

#ifndef AI_CPU_GUARDED_PHILOX_RANDOM_H_
#define AI_CPU_GUARDED_PHILOX_RANDOM_H_

#include "cpu_kernel.h"
#include "log.h"
#include "utils/philox_random.h"

namespace aicpu {
namespace random {
// Wrapper around the Philox generator
class GuardedPhiloxRandom {
 public:
  GuardedPhiloxRandom() = default;
  // Initialize the generator from attributes "seed" and "seed2".
  // If both seeds are unspecified, use random seeds.
  void Init(const CpuKernelContext& context);

  // Initialize with given seeds.
  void Init(int64_t seed, int64_t seed2);
  void Init(PhiloxRandom::ResultType counter, PhiloxRandom::Key key);

  // Reserve a certain number of 128-bit samples.
  PhiloxRandom ReserveSamples128(int64_t samples);

  // Reserve enough random samples in the generator for the given output count.
  PhiloxRandom ReserveRandomOutputs(int64_t output_count, int64_t multiplier) {
    int64_t conservative_sample_count = output_count * multiplier;
    return ReserveSamples128(conservative_sample_count);
  }

 private:
  PhiloxRandom generator_;
};
}  // namespace random
}  // namespace aicpu

#endif
