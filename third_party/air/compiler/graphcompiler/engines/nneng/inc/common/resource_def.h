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
 * @ingroup cce
 * @brief return value
 */
typedef enum tagCcStatus {
  CC_STATUS_SUCCESS = 0,         /**< succ */
  CC_STATUS_NOT_INITIALIZED = 1, /**< not init */
  CC_STATUS_ALLOC_FAILED = 2,    /**< alloc mem failed */
  CC_STATUS_BAD_PARAM = 3,       /**< para check failed */
  CC_STATUS_INTERNAL_ERROR = 4,  /**< internal error */
  CC_STATUS_KERNEL_ERROR = 5,    /**< kernel error */
  CC_STATUS_RUNTIME_ERROR = 6,   /**< runtime error */
  CC_STATUS_NOT_SUPPORTED = 7,   /**< unsupport error */
  CC_STATUS_INVALID_VALUE = 7,   /**< invalid value error for blas*/
  CC_STATUS_RESERVED             /**< just for check */
} ccStatus_t;

/**
 * @ingroup fe
 * @brief save context of fe library
 */
typedef struct tagCcContext {
  rtStream_t streamId;
  uint32_t opIndex;
} ccContext_t;

typedef struct tagCcContext *ccHandle_t;

typedef struct tagContextInfo {
  ccHandle_t handle;
  rtStream_t stream;
  uint8_t *memBase;
  uint64_t totalMemSize;
  uint8_t *weightsMemBase;
  uint64_t weightsMemSize;
  uint8_t *weightsMemBaseHost;
} ContextInfo;

/**
 * @ingroup fe
 * @brief function parameter type
 */
typedef enum tagFuncType {
  FUSION_L2,
  GLOBAL_MEMORY_CLEAR,
  MAX_NUM,
} funcParamType_t;

/**
 * @ingroup fe
 * @brief set function point state
 */
ccStatus_t setFuncState(funcParamType_t type, bool isOpen);

/**
 * @ingroup cce
 * @brief cce get function point state
 */
bool getFuncState(funcParamType_t type);

}  // namespace fe
#endif  // RESOURCE_DEF_H__
