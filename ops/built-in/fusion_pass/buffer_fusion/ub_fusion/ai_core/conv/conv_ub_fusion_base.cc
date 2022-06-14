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
 * \file conv_ub_fusion_base.cpp
 * \brief
 */
#include "conv_ub_fusion_base.h"
#include <math.h>
#include <vector>
#include <nlohmann/json.hpp>
#include "common/util/platform_info.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph/utils/attr_utils.h"
#include "common/lxfusion_json_util.h"
#include "lx_fusion_func.h"

namespace fe {
/***************************************************************
The other input can be a scalar or broadcast with a value on the C1 and C0
***************************************************************/
bool BroadcastPatternRule::CheckTensorData(const ge::GeTensorDesc tensor_0, const ge::GeTensorDesc tensor_1)
{
    if (tensor_1.GetFormat() == ge::FORMAT_NC1HWC0 &&
        tensor_0.GetShape().GetDimNum() == 5) {  // The shape size needs to be same
        int64_t tensor1_c1_shape = tensor_1.GetShape().GetDim(1);  // get C1 shape size
        int64_t tensor1_c0_shape = tensor_1.GetShape().GetDim(4);  // get C0 shape size
        if (tensor_0.GetShape().GetDim(1) == tensor1_c1_shape &&   // C1 shape needs to be same
            tensor_0.GetShape().GetDim(4) == tensor1_c0_shape &&   // C0 shape needs to be same
            tensor_0.GetShape().GetShapeSize() == tensor1_c1_shape * tensor1_c0_shape) {
            return true;
        }
    }
    return false;
}

bool BroadcastPatternRule::CheckBroadcastFusionScenario(const BufferFusionMapping &mapping,
                                                        vector<ge::NodePtr> &fusion_nodes)
{
    for (auto &item : mapping) {
        const BufferFusionOpDesc *op_desc = item.first;
        if (op_desc != nullptr && op_desc->types[0] == OP_PATTERN_BROAD_CAST &&
            op_desc->inputs.size() == 2) {      // The operator must have two inputs
            ge::NodePtr node = item.second[0];
            if (node == nullptr) {
                return false;
            }
            const ge::GeTensorDesc input_tensor_0 = node->GetOpDesc()->GetInputDesc(0);
            const ge::GeTensorDesc input_tensor_1 = node->GetOpDesc()->GetInputDesc(1);
            if (input_tensor_0.GetShape().GetDimNum() == 1 ||
                input_tensor_1.GetShape().GetDimNum() == 1) {
                continue;
            }
            if (CheckTensorData(input_tensor_0, input_tensor_1) || CheckTensorData(input_tensor_1, input_tensor_0)) {
                continue;
            }
            fusion_nodes.clear();
            return false;
        }
    }
    return true;
}
}  // namespace fe