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
#include "aicpu_sharder.h"

#include "aicpu_context.h"

namespace aicpu {
SharderNonBlock::SharderNonBlock()
    : schedule_(NULL), doTask_(NULL), cpuCoreNum_(0) {}

SharderNonBlock &SharderNonBlock::GetInstance() {
  static SharderNonBlock sharderNonBlock;
  return sharderNonBlock;
}

void SharderNonBlock::ParallelFor(int64_t total, int64_t perUnitSize,
                                  const SharderWork &work) {
  work(0, total);
}

uint32_t SharderNonBlock::GetCPUNum() { return 1; }

status_t GetThreadLocalCtx(const std::string &key, std::string &value) {
  return AICPU_ERROR_NONE;
}
}  // namespace aicpu
