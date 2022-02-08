/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef RESOURCE_DEF_H__
#define RESOURCE_DEF_H__

#include "runtime/rt.h"

namespace fe {
/**
 * @ingroup fe
 * @brief save context of fe library
 */
using ccContext_t = struct tagCcContext {
  rtStream_t streamId;
  uint32_t opIndex;
};

using ccHandle_t = struct tagCcContext*;

/**
 * @ingroup fe
 * @brief function parameter type
 */
enum class FuncParamType {
  FUSION_L2,
  GLOBAL_MEMORY_CLEAR,
  MAX_NUM,
};

/**
 * @ingroup fe
 * @brief set function point state
 */
bool setFuncState(FuncParamType type, bool isOpen);

/**
 * @ingroup cce
 * @brief cce get function point state
 */
bool getFuncState(FuncParamType type);

}  // namespace fe
#endif  // RESOURCE_DEF_H__
