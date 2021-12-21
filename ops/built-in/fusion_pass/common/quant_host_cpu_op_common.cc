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

/*!
 * \file quant_host_cpu_op_common.cpp
 * \brief
 */
#include "quant_host_cpu_op_common.h"

namespace fe {
inline Status CheckInt64MulOverflowForPass(int64_t a, int64_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > ((int64_t)INT64_MAX / b)) {
        return FAILED;
      }
    } else {
      if (b < ((int64_t)INT64_MIN / a)) {
        return FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < ((int64_t)INT64_MIN / b)) {
        return FAILED;
      }
    } else {
      if ((a != 0) && (b < ((int64_t)INT64_MAX / a))) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status GetkernelDataCountForPass(const std::vector<int64_t>& filterDIms, int64_t& kernelDataCount) {
  for (size_t i = 0; i < filterDIms.size(); i++) {
    if (CheckInt64MulOverflowForPass(kernelDataCount, filterDIms.at(i)) != SUCCESS) {
      return FAILED;
    }
    kernelDataCount *= filterDIms.at(i);
  }
  return SUCCESS;
}
}  // namespace fe
