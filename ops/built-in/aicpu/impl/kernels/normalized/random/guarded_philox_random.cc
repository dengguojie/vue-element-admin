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

#include "guarded_philox_random.h"

#include "utils.h"

namespace aicpu {
namespace random {
void GuardedPhiloxRandom::Init(const CpuKernelContext& ctx) {
  int64_t seed = 0;
  int64_t seed2 = 0;

  auto attr_seed = ctx.GetAttr("seed");
  if (attr_seed != nullptr) {
    seed = attr_seed->GetInt();
  }
  auto attr_seed2 = ctx.GetAttr("seed2");
  if (attr_seed2 != nullptr) {
    seed2 = attr_seed2->GetInt();
  }
  Init(seed, seed2);
}

void GuardedPhiloxRandom::Init(int64_t seed, int64_t seed2) {
  if (seed == 0 && seed2 == 0) {
    seed = New64();
    seed2 = New64();
  }
  generator_ = PhiloxRandom(seed, seed2);
}

void GuardedPhiloxRandom::Init(PhiloxRandom::ResultType counter,
                               PhiloxRandom::Key key) {
  generator_ = PhiloxRandom(counter, key);
}

PhiloxRandom GuardedPhiloxRandom::ReserveSamples128(int64_t samples) {
  auto local = generator_;
  generator_.Skip(samples);
  return local;
}
}  // namespace random
}  // namespace aicpu
