/**
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief precheck functions.
 *
 * @version 1.0
 *
 */

#ifndef BUILT_IN_FUSION_PASS_PRECHECK_FUNC_H_
#define BUILT_IN_FUSION_PASS_PRECHECK_FUNC_H_

#include "graph/utils/node_utils.h"
namespace fe {

bool ApplyRmsPropPreCheck(ge::NodePtr node);
bool FusedMulApplyMomentumPreCheck(ge::NodePtr node);
bool FusedMulApplyMomentumExternPreCheck(ge::NodePtr node);
bool ApplyAdagradV2PreCheck(ge::NodePtr node);
bool ApplyKerasMomentumPreCheck(ge::NodePtr node);
bool SparseApplyFtrlPreCheck(ge::NodePtr node);
bool SparseApplyFtrlV2PreCheck(ge::NodePtr node);
bool SparseApplyAdagradV2PreCheck(ge::NodePtr node);
bool SparseApplyRmsPropPreCheck(ge::NodePtr node);
bool SparseApplyAdadeltaPreCheck(ge::NodePtr node);
}  // namespace fe
#endif
