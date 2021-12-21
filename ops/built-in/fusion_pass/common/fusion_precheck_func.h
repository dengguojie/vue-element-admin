/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file fusion_precheck_func.h
 * \brief precheck functions.
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_COMMON_FUSION_PRECHECK_FUNC_H_
#define OPS_BUILT_IN_FUSION_PASS_COMMON_FUSION_PRECHECK_FUNC_H_

#include "graph/utils/node_utils.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
namespace fe {

Status ApplyRmsPropPreCheck(ge::NodePtr node);
Status FusedMulApplyMomentumPreCheck(ge::NodePtr node);
Status FusedMulApplyMomentumExternPreCheck(ge::NodePtr node);
Status FusedMulApplyKerasMomentumPreCheck(ge::NodePtr node);
Status ApplyAdagradV2PreCheck(ge::NodePtr node);
Status ApplyKerasMomentumPreCheck(ge::NodePtr node);
Status SparseApplyFtrlPreCheck(ge::NodePtr node);
Status SparseApplyFtrlV2PreCheck(ge::NodePtr node);
Status SparseApplyAdagradV2PreCheck(ge::NodePtr node);
Status SparseApplyRmsPropPreCheck(ge::NodePtr node);
Status SparseApplyAdadeltaPreCheck(ge::NodePtr node);
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_COMMON_FUSION_PRECHECK_FUNC_H_
