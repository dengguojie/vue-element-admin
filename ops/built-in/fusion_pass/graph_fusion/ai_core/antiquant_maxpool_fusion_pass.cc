/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @brief antiquant maxpool fusion pass
 *
 */

#include "antiquant_maxpool_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const char *ANTIQUANT = "AscendAntiQuant";
static const char *MAXPOOL = "MaxPool";
static const char *QUANT = "AscendQuant";
static const std::string PATTERN_ANTIQUANT = "FusedNodeAntiQuant";
static const std::string PATTERN_MAXPOOL = "FusedNodeMaxPool";
static const std::string PATTERN_QUANT = "FusedNodeQuant";

vector<FusionPattern*> AntiQuantMaxPoolFusionPass::DefinePatterns() {
  vector < FusionPattern * > patterns;
  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("AntiQuantMaxPoolFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

  pattern->AddOpDesc(PATTERN_ANTIQUANT, {ANTIQUANT})
      .AddOpDesc(PATTERN_MAXPOOL, {MAXPOOL})
      .AddOpDesc(PATTERN_QUANT, {QUANT})
      .SetInputs(PATTERN_MAXPOOL, {PATTERN_ANTIQUANT})
      .SetInputs(PATTERN_QUANT, {PATTERN_MAXPOOL})
      .SetOutput(PATTERN_QUANT);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AntiQuantMaxPoolFusionPass pattern end");
  return patterns;
}

Status AntiQuantMaxPoolFusionPass::Fusion(ge::ComputeGraph& graph,
                                        Mapping& mapping,
                                        vector<ge::NodePtr> &fusionNodes) {
  // get all nodes
  ge::NodePtr anti_node = GetNodeFromMapping(PATTERN_ANTIQUANT, mapping);
  ge::NodePtr maxpool_node = GetNodeFromMapping(PATTERN_MAXPOOL, mapping);
  FUSION_PASS_CHECK(anti_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "anti_node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(maxpool_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "maxpool_node is null, fusion failed."), return PARAM_INVALID);

  // if not satisfied, then back
  int flag_back = 0;
  ge::NodePtr quant_node = anti_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  if (quant_node->GetOpDesc()->GetType() != "AscendQuant") {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the input node of anti_node is not AscendQuant, not change");
    return NOT_CHANGED;
  }

  ge::NodePtr peer_node = quant_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  auto peer_type = peer_node->GetOpDesc()->GetType();
  if ((peer_type == "MaxPool") || (peer_type == "ConcatV2")) {
    flag_back = 1;
  }
  if (peer_type == "ConcatV2") {
    size_t inputs_num = peer_node->GetOpDesc()->GetInputsSize();
    for (size_t i = 0; i < inputs_num - 1; ++i) {
      ge::GeTensorDesc input_desc = peer_node->GetOpDesc()->GetInputDesc(i);
      ge::GeShape input_shape = input_desc.GetOriginShape();
      int dimNum = input_shape.GetDimNum();
      if (dimNum > 0) {
        int dimC = input_shape.GetDim(dimNum - 1);
        if (dimC % 32 != 0) {
          flag_back = 1;
          break;
        }
      }
    }
  }

  if (flag_back == 1) {
    // connect output edge
    for (auto inDataAnchor :
        anti_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
        FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(anti_node->GetOutDataAnchor(0),
                                            inDataAnchor) != SUCCESS,
                OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed for back."), return FAILED);
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(peer_node->GetOutDataAnchor(0),
                                        inDataAnchor) != SUCCESS,
                OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed for back."), return FAILED);
    }
    // set AscendQuant attr
    float scale;
    Operator quant_op = ge::OpDescUtils::CreateOperatorFromNode(quant_node);
    if (GRAPH_SUCCESS == quant_op.GetAttr("origin_scale", scale)) {
      quant_op.SetAttr("scale", scale);
    }
    // set RequantHostCpuOp attr
    float quant_scale = 1;
    if (peer_type == "ConcatV2") {
      size_t inputs_num = peer_node->GetOpDesc()->GetInputsSize();
      for (size_t i = 0; i < inputs_num - 1; ++i) {
        ge::NodePtr dequant_node = peer_node->GetInDataAnchor(i)->GetPeerOutAnchor()->GetOwnerNode();
        if (dequant_node->GetOpDesc()->GetType() == "AscendDequant") {
          ge::NodePtr requant_node = dequant_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
          if (requant_node->GetOpDesc()->GetType() == "RequantHostCpuOp") {
            Operator requant_op = ge::OpDescUtils::CreateOperatorFromNode(requant_node);
            requant_op.SetAttr("quant_scale", quant_scale);
          } else {
            OP_LOGI(FUSED_OP_TYPE.c_str(), "peer node is not RequantHostCpuOp, not changed");
            return NOT_CHANGED;
          }
        } else {
          OP_LOGI(FUSED_OP_TYPE.c_str(), "peer node is not AscendDequant, not changed");
          return NOT_CHANGED;
        }
      }
    } else {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "peer node is not ConcatV2, not changed");
      return NOT_CHANGED;
    }
    // delete fused nodes
    FUSION_PASS_CHECK(graph.RemoveNode(anti_node) != SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove anti_node failed for back."), return FAILED);
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "AntiQuantMaxPoolFusionPass graph fusion success!");
  return SUCCESS;
}
REGISTER_PASS("AntiQuantMaxPoolFusionPass", BUILT_IN_GRAPH_PASS, AntiQuantMaxPoolFusionPass);
}
