/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "utils.h"

#include <random>

namespace aicpu {
namespace random {
std::mt19937_64* InitRngWithRandomSeed() {
  std::random_device device("/dev/urandom");
  return new std::mt19937_64(device());
}

uint64_t New64() {
  static std::mt19937_64* rng = InitRngWithRandomSeed();
  return (*rng)();
}
}  // namespace random
}  // namespace aicpu
