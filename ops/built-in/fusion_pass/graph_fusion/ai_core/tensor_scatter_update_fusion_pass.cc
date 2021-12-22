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
 * \file tensor_scatter_update_fusion_pass.cpp
 * \brief TensorScatterUpdate fusion pass
 *   (TensorScatterUpdate --> TensorMove & ScatterNdUpdate)
 */
#include "tensor_scatter_update_fusion_pass.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>
#include "op_log.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

namespace fe {
static const string PATTERN_FUSEDNODE = "FusedNodeTensorScatterUpdate";
static const string FUSED_NODE = "TensorScatterUpdate";
static const string TENSORMOVE = "TensorMove";
static const string SCATTERNDUPDATE = "ScatterNdUpdate";

/*  TensorScatterUpdate --> TensorMove & ScatterNdUpdate

   x        indices    updates           x    indices updates
    \         |          /               |        /   /
     \        |         /                |       /   /
      tensor_scatter_update      -->tensor_move /   /
              |                          |     /   /
              |                          |    /   /
              y                     scatter_nd_update
                                         |
                                         |
                                         y
*/

vector<FusionPattern*> TensorScatterUpdateFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define TensorScatterUpdateFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("TensorScatterUpdateFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define TensorScatterUpdateFusionPass pattern end");
  return patterns;
}

Status TensorScatterUpdateFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                             vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define TensorScatterUpdateFusionPass fusion begin.");
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // create tensor_move OpDesc and scatter_nd_update OpDesc
  std::shared_ptr<ge::OpDesc> tensorMoveOpdesc = nullptr;
  tensorMoveOpdesc = std::make_shared<ge::OpDesc>(fusedNode->GetName(), TENSORMOVE);
  FUSION_PASS_CHECK(tensorMoveOpdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "tensorMoveOpdesc is null, fusion failed."),
                                                   return PARAM_INVALID);
  std::shared_ptr<ge::OpDesc> scatterNdUpdatedesc = nullptr;
  scatterNdUpdatedesc = std::make_shared<ge::OpDesc>(fusedNode->GetName(), SCATTERNDUPDATE);
  FUSION_PASS_CHECK(scatterNdUpdatedesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "scatterNdUpdatedesc is null, fusion failed."),
                    return PARAM_INVALID);

  // add input and output
  ge::GeTensorDesc input_x = fusedNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(tensorMoveOpdesc->AddInputDesc(input_x) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input x failed."), return FAILED);
  ge::GeTensorDesc output_y = fusedNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(tensorMoveOpdesc->AddOutputDesc(output_y) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output y failed."), return FAILED);
  ge::NodePtr tensorMoveNode = graph.AddNode(tensorMoveOpdesc);

  ge::GeTensorDesc input_var = tensorMoveNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(scatterNdUpdatedesc->AddInputDesc("var", input_var) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input var failed."), return FAILED);
  ge::GeTensorDesc input_indices = fusedNode->GetOpDesc()->GetInputDesc(1);
  FUSION_PASS_CHECK(scatterNdUpdatedesc->AddInputDesc("indices", input_indices) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input indices failed."), return FAILED);
  ge::GeTensorDesc input_updates = fusedNode->GetOpDesc()->GetInputDesc(2);
  FUSION_PASS_CHECK(scatterNdUpdatedesc->AddInputDesc("updates", input_updates) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input updates failed."), return FAILED);
  ge::GeTensorDesc output_var = fusedNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(scatterNdUpdatedesc->AddOutputDesc("var", output_var) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output var failed."), return FAILED);

  ge::NodePtr scatterNdUpdateNode = graph.AddNode(scatterNdUpdatedesc);

  // set attr
  Operator scatterNdUpdate = ge::OpDescUtils::CreateOperatorFromNode(scatterNdUpdateNode);
  scatterNdUpdate.SetAttr("use_locking", false);

  FUSION_PASS_CHECK(tensorMoveNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "tensorMoveNode fusionNode:%s is null, fusion failed.",
                            tensorMoveNode->GetName().c_str()),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(scatterNdUpdateNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "scatterNdUpdateNode fusionNode:%s is null, fusion failed.",
                            scatterNdUpdateNode->GetName().c_str()),
                    return PARAM_INVALID);

  tensorMoveOpdesc->SetName(fusedDesc->GetName() + "/TensorMove");
  scatterNdUpdatedesc->SetName(fusedDesc->GetName() + "/ScatterNdUpdate");
  tensorMoveOpdesc->SetType("TensorMove");
  scatterNdUpdatedesc->SetType("ScatterNdUpdate");
  fusionNodes.push_back(tensorMoveNode);
  fusionNodes.push_back(scatterNdUpdateNode);

  // connect x with tensor_move
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            tensorMoveNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge0 failed."), return FAILED);
  // connect tensor_move's output with scatter_nd_update
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(tensorMoveNode->GetOutDataAnchor(0), scatterNdUpdateNode->GetInDataAnchor(0)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge1 failed."), return FAILED);
  // connect indices with scatter_nd_update
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            scatterNdUpdateNode->GetInDataAnchor(1)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge2 failed."), return FAILED);
  // connect updates with scatter_nd_update
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(2)->GetPeerOutAnchor(),
                                            scatterNdUpdateNode->GetInDataAnchor(2)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge3 failed."), return FAILED);
  // connect scatter_nd_update'output with next node
  for (auto inDataAnchor : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."),
                                                     return FAILED);

    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(scatterNdUpdateNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge4 failed."), return FAILED);
  }

  FUSION_PASS_CHECK(graph.RemoveNode(fusedNode) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Remove fusedNode failed."),
                    return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define TensorScatterUpdateFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("TensorScatterUpdateFusionPass", BUILT_IN_GRAPH_PASS, TensorScatterUpdateFusionPass);
}  // namespace fe
