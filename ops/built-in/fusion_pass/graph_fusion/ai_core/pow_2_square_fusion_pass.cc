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
 * \file pow_2_square_fusion_pass.cpp
 * \brief pow fusion pass( --> square)
 */
#include "pow_2_square_fusion_pass.h"
#include <math.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace std;
using namespace ge;

namespace fe {
static const char* POW = "Pow";
static const char* SQUARE = "Square";
static const string PATTERN_POW = "Pow";
static const string CONSTANT = "Const";
static const string CONSTANTOP = "Constant";
static const string DATAOP = "Data";

vector<FusionPattern*> Pow2SquareFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("Pow2SquareFusionPass");
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter Pow2SquareFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_POW, {POW}).SetOutput(PATTERN_POW);
  patterns.push_back(pattern);
  return patterns;
}

Status Pow2SquareFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  float constData = 0.0;
  size_t constSize2 = 0;
  ge::DataType constType2 = DT_UNDEFINED;
  float* constDataPtr = 0;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define Pow2SquareFusionPass fusion begin");
  ge::NodePtr pow_node = GetNodeFromMapping(PATTERN_POW, mapping);
  FUSION_PASS_CHECK(pow_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "pow node is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr pow_desc = pow_node->GetOpDesc();
  FUSION_PASS_CHECK(pow_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "pow_node's Op_desc is null, fusion failed."),
                    return PARAM_INVALID);

  ge::ConstGeTensorPtr constTensor2 = nullptr;
  ge::InDataAnchorPtr PowAnchorPtr2 = pow_node->GetInDataAnchor(1);
  ge::OutDataAnchorPtr constAnchorPtr2 = PowAnchorPtr2->GetPeerOutAnchor();
  ge::NodePtr constNode2 = constAnchorPtr2->GetOwnerNode();
  ge::OpDescPtr constNode2_desc = constNode2->GetOpDesc();
  std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(constNode2);
  if (type != CONSTANT && type != CONSTANTOP && type != DATAOP) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The type of y input is not constant.");
    return NOT_CHANGED;
  }
  vector<ge::GeTensorPtr> pow_y = ge::OpDescUtils::MutableWeights(constNode2);
  FUSION_PASS_CHECK(pow_y.empty(), OP_LOGI(FUSED_OP_TYPE.c_str(), "Pow input y is tensor!"),
                    return NOT_CHANGED);
  constTensor2 = pow_y[0];
  constSize2 = constTensor2->GetData().GetSize();
  constType2 = constTensor2->GetTensorDesc().GetDataType();
  if (constTensor2->GetData().GetData() != nullptr) {
    constDataPtr = (float*)constTensor2->GetData().GetData();
    constData = (float)(*constDataPtr);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Pow index is %f", constData);
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Pow input y is tensor");
    return NOT_CHANGED;
  }
  if (fabs(constData - 2.0) <= 1e-6 && constType2 == ge::DT_FLOAT && constSize2 == 4) {
    ge::GeTensorDesc input_desc0 = pow_node->GetOpDesc()->GetInputDesc(0);
    ge::GeTensorDesc input_desc1 = pow_node->GetOpDesc()->GetInputDesc(1);
    ge::GeTensorDesc output_desc0 = pow_node->GetOpDesc()->GetOutputDesc(0);

    ge::OpDescPtr square_op;
    FUSION_PASS_MAKE_SHARED((square_op = std::make_shared<ge::OpDesc>(pow_node->GetName() + "/" + SQUARE, SQUARE)),
                            return INTERNAL_ERROR);
    square_op->AddInputDesc("x", input_desc0);
    square_op->AddOutputDesc("y", output_desc0);
    ge::NodePtr square_node = graph.AddNode(square_op);
    newNodes.push_back(square_node);

    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(pow_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                         square_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add square node in data edge failed."), return FAILED);

    if (pow_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of layerpownode is [%d].",
              pow_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
      for (InDataAnchorPtr inAnchorPtr : pow_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
        inAnchorPtr->UnlinkAll();
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(square_node->GetOutDataAnchor(0), inAnchorPtr),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                  "Add edge from fused node:%s's 2nd index to fusion "
                                  "node:%s's 1st index failed.",
                                  pow_node->GetName().c_str(), square_node->GetName().c_str()),
                          return FAILED);
        OP_LOGD(FUSED_OP_TYPE.c_str(),
                "Add edge from fused node:%s's 2nd index to fusion node:%s's 1st "
                "index.",
                pow_node->GetName().c_str(), square_node->GetName().c_str());
      }
    }
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(pow_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                            pow_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove pow node in data0 edge failed."), return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(pow_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                                            pow_node->GetInDataAnchor(1)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove pow node in data1 edge failed."), return FAILED);

    FUSION_PASS_CHECK(graph.RemoveNode(pow_node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove pow node failed."),
                      return FAILED);
    //    remove constNode2 node if output is 1
    if (constNode2->GetAllOutDataAnchors().size() == 1) {
      if (constNode2->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() == 0 &&
          constNode2->GetAllInDataAnchors().size() == 0) {
        FUSION_PASS_CHECK(graph.RemoveNode(constNode2) != SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove constant 2 node failed."), return FAILED);
      }
    }
    return SUCCESS;
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Pow2SquareFusionPass fusion NOT_CHANGED");
    return NOT_CHANGED;
  }
}
REGISTER_PASS("Pow2SquareFusionPass", BUILT_IN_GRAPH_PASS, Pow2SquareFusionPass);
}  // namespace fe
