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
#ifndef AICPU_CONTEXT_COMMON_STATUS_H_
#define AICPU_CONTEXT_COMMON_STATUS_H_

#include <cstdint>

namespace aicpu {
/*
 * status code
 */
enum KernelStatus : uint32_t {
  KERNEL_STATUS_OK = 0,
  KERNEL_STATUS_PARAM_INVALID,
  KERNEL_STATUS_INNER_ERROR,
  KERNEL_STATUS_PROTOBUF_ERROR,
  KERNEL_STATUS_SHARDER_ERROR
};
}  // namespace aicpu
#endif  // AICPU_CONTEXT_COMMON_STATUS_H_
