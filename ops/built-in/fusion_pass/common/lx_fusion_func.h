/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file lx_fusion_func.h
 * \brief fe_util
 */
#ifndef BUILTIN_FUSIONPASS_LX_FUISON_FUNC_H
#define BUILTIN_FUSIONPASS_LX_FUISON_FUNC_H
#include <string>
#include <vector>
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
#include "common/lxfusion_json_util.h"
#include "common/op_slice_info.h"
#include "graph/utils/attr_utils.h"


namespace fe {

void DelSplitInfoByOutputAxis(std::vector<AxisSplitMap>& split_maps, int axis);
void DelSplitInfoByInputAxis(std::vector<AxisSplitMap>& split_maps, int axis);
bool GetSplitMap(std::vector<AxisSplitMap>& split_maps, ge::NodePtr& cube_node, const string& fused_op_type);
void SetSplitMap(std::vector<AxisSplitMap>& split_maps,
                 std::vector<ge::NodePtr>& fusionNodes, const string& fused_op_type);
void AddElemwiseSplitMap(std::vector<AxisSplitMap>& split_maps, ge::NodePtr& elemWiseNode, int& index);
}  // namespace fe
#endif  // BUILTIN_FUSIONPASS_LX_FUISON_FUNC_H