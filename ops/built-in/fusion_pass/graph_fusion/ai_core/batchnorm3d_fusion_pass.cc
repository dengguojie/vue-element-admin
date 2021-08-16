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

/*!
 * \file batchnorm3d_fusion_pass.h
 * \brief batchnorm3d_fusion_pass(BatchNorm - > BatchNorm3d)
 */

#include "batchnorm3d_fusion_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "error_util.h"

namespace fe {
static const string PATTERN_BATCHNORM = "BatchNorm";
static const string OP_TYPE_BATCHNORM = "BatchNorm";
static const int DIM_4 = 4;

vector<FusionPattern *> BatchNorm3DFusionPass::DefinePatterns() {
    vector<FusionPattern*> patterns;
    FusionPattern *pattern = new (std::nothrow) FusionPattern("BatchNorm3DFusionPass");
    if (pattern == nullptr) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "pattern is nullptr,Create pattern failed.");
        return patterns;
    }
    pattern->AddOpDesc(PATTERN_BATCHNORM, {OP_TYPE_BATCHNORM})
            .SetOutput(PATTERN_BATCHNORM);
    patterns.push_back(pattern);
    return patterns;
}

Status BatchNorm3DFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusionNodes) {
    ge::NodePtr batchnorm_node = GetNodeFromMapping(PATTERN_BATCHNORM, mapping);
    FUSION_PASS_CHECK(batchnorm_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GetNodeFromMapping failed."), return FAILED);
    auto op_desc = batchnorm_node->GetOpDesc();
    FUSION_PASS_CHECK(op_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GetOpDesc failed."), return FAILED);
    
    auto input_desc = op_desc->GetInputDesc(0);
    auto dim_num  = input_desc.GetShape().GetDimNum();
    if (dim_num <= DIM_4) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Input dim is 4d, not need change");
      return NOT_CHANGED;
    }

    std::string onnx = "";
    ge::AttrUtils::GetStr(op_desc, "onnx", onnx);
    if (onnx == "") {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "this op is not onnx, not need change");
      return NOT_CHANGED;
    }
    input_desc.SetFormat(ge::FORMAT_NCDHW);
    input_desc.SetOriginFormat(ge::FORMAT_NCDHW);
    op_desc->UpdateInputDesc("x", input_desc);

    auto output_desc = op_desc->GetOutputDesc(0);
    output_desc.SetFormat(ge::FORMAT_NCDHW);
    output_desc.SetOriginFormat(ge::FORMAT_NCDHW);
    op_desc->UpdateOutputDesc("y", output_desc);
    ge::AttrUtils::SetStr(op_desc, "data_format", "NCDHW");
    op_desc->SetType(FUSED_OP_TYPE);
    return SUCCESS;
}
REGISTER_PASS("BatchNorm3DFusionPass", BUILT_IN_GRAPH_PASS, BatchNorm3DFusionPass);
}