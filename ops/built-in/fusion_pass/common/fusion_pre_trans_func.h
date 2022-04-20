/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 * \file fusion_pre_trans_func.h
 * \brief fusion pre-transdata and cube node
 */
#ifndef BUILTIN_FUSIONPASS_COMMON_FUSION_PRE_TRANS_FUNC_H
#define BUILTIN_FUSIONPASS_COMMON_FUSION_PRE_TRANS_FUNC_H

#include <vector>
#include "graph/utils/type_utils.h"
#include "pattern_fusion_util.h"


namespace fe {
void FusePreTransdata(std::vector<ge::NodePtr>& cube_nodes, std::vector<ge::NodePtr>& fusion_nodes);
}  // namespace fe
#endif  // BUILTIN_FUSIONPASS_COMMON_FUSION_PRE_TRANS_FUNC_H