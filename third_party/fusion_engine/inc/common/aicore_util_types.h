/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#ifndef FUSION_ENGINE_INC_COMMON_AICORE_UTIL_TYPES_H_
#define FUSION_ENGINE_INC_COMMON_AICORE_UTIL_TYPES_H_

#include <map>
#include <string>
#include <vector>
#include "graph/anchor.h"
#include "graph/types.h"

namespace fe {
    enum OpReduceType { REDUCE_MEAN = 0, REDUCE_ADD, REDUCE_MAX, REDUCE_MIN};

    enum OpL1FusionType { L1FUSION_DISABLE = 0, L1FUSION_BASIC, L1FUSION_INPUT_CTR};
    static const std::string OP_SLICE_INFO = "_op_slice_info";
    static const std::string FUSION_OP_SLICE_INFO = "_fusion_op_slice_info";
}
#endif // FUSION_ENGINE_INC_COMMON_AICORE_UTIL_TYPES_H_