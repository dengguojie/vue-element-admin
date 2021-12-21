/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

#include "dynamic_gru_v2_transdatarnn_fusion_pass.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph_optimizer/fusion_common/graph_pass_util.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "external/graph/operator_factory.h"
#include "common/util/platform_info.h"

using namespace ge;
namespace fe {
static const char *FUSED_NODE = "DynamicGRUV2";
static const std::string PATTERN_FUSEDNODE = "DynamicGRUV2";

vector<FusionPattern *> DynamicGRUV2TransFusionPass::DefinePatterns()
{
  vector<FusionPattern *> patterns;

  FusionPattern *pattern = new (std::nothrow) FusionPattern("DynamicGRUV2TransFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                "dynamicGRUV2 transdatarnn pattern object failed."), return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, { FUSED_NODE }).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);
  return patterns;
}

Status DynamicGRUV2TransFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes)
{
  // get the NodePtr of dynamic_gru_v2
  OP_LOGI(FUSED_OP_TYPE.c_str(), "dynamic_gru_v2 transdatarnn start fusion");

  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "fusedNode is null, fusion failed."),
  return PARAM_INVALID);

  // get the OpDescPtr of dynamic_gru_v2
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "fusedNode OpDesc is null, fusion failed."),
  return PARAM_INVALID);

  int64_t input_x = fusedDesc->GetInputDesc(0).GetOriginShape().GetDim(2);
  int64_t hidden_size = fusedDesc->GetInputDesc(2).GetOriginShape().GetDim(0);

  if (input_x % 16 == 0 && hidden_size % 16 == 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "dynamic_gru_v2 do not need transdatarnn.");
    return NOT_CHANGED;
  }

  int64_t inputAlign = (input_x + 15) / 16;
  int64_t hiddenAlign = (hidden_size + 15) / 16;

  vector<int64_t> weightInputDim{inputAlign, 3 * hiddenAlign, 16, 16};
  vector<int64_t> weightHiddenDim{hiddenAlign, 3 * hiddenAlign, 16, 16};
  vector<int64_t> biasDim{3 * hiddenAlign * 16};

  ge::GeTensorDesc weightInputDesc = fusedDesc->GetInputDesc(1);
  weightInputDesc.SetShape(ge::GeShape(weightInputDim));
  weightInputDesc.SetFormat(ge::FORMAT_FRACTAL_ZN_RNN);
  fusedDesc->UpdateInputDesc("weight_input", weightInputDesc);

  ge::GeTensorDesc weightHiddenDesc = fusedDesc->GetInputDesc(2);
  weightHiddenDesc.SetShape(ge::GeShape(weightHiddenDim));
  weightHiddenDesc.SetFormat(ge::FORMAT_FRACTAL_ZN_RNN);
  fusedDesc->UpdateInputDesc("weight_hidden", weightHiddenDesc);

  bool hasInputBias = fusedDesc->MutableInputDesc("bias_input") != nullptr;
  if (hasInputBias) {
    ge::GeTensorDesc biasInputDesc = fusedDesc->GetInputDesc(3);
    biasInputDesc.SetShape(ge::GeShape(biasDim));
    biasInputDesc.SetFormat(ge::FORMAT_ND_RNN_BIAS);
    fusedDesc->UpdateInputDesc("bias_input", biasInputDesc);
  }

  bool hasHiddenBias = fusedDesc->MutableInputDesc("bias_hidden") != nullptr;
  if (hasHiddenBias) {
    ge::GeTensorDesc biasHiddenDesc = fusedDesc->GetInputDesc(4);
    biasHiddenDesc.SetShape(ge::GeShape(biasDim));
    biasHiddenDesc.SetFormat(ge::FORMAT_ND_RNN_BIAS);
    fusedDesc->UpdateInputDesc("bias_hidden", biasHiddenDesc);
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "dynamic_gru_v2 transdatarnn end fusion");
  return SUCCESS;
}

REGISTER_PASS("DynamicGRUV2TransFusionPass", BUILT_IN_GRAPH_PASS, DynamicGRUV2TransFusionPass);
} // namespace fe
