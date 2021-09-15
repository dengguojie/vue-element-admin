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
 * \file mul_square_fusion_pass.cpp
 * \brief mul fusion pass( --> square)
 *   input0     input1            input0
 *      \       /                    |
 *       \    /        ------->      |
 *        \ /                        |
 *        mul                     square
 *
 */

#include "mul_square_fusion_pass.h"
#include <math.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace std;
using namespace ge;

namespace fe {
static const char* MUL = "Mul";
static const char* SQUARE = "Square";
static const string PATTERN_MUL = "Mul";

vector<FusionPattern*> MulSquareFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("MulSquareFusionPass");
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter MulSquareFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_MUL, {MUL}).SetOutput(PATTERN_MUL);
  patterns.push_back(pattern);
  return patterns;
}

Status MulSquareFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define MulSquareFusionPass fusion begin");
  ge::NodePtr mul_node = GetNodeFromMapping(PATTERN_MUL, mapping);
  FUSION_PASS_CHECK(mul_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mul node is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr mul_desc = mul_node->GetOpDesc();
  FUSION_PASS_CHECK(mul_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mul_node's Op_desc is null, fusion failed."),
                    return PARAM_INVALID);
  ge::ConstGeTensorPtr constTensor = nullptr;
  ge::InDataAnchorPtr mul_anchorptr_1 = mul_node->GetInDataAnchor(0);
  ge::OutDataAnchorPtr anchorptr_1 = mul_anchorptr_1->GetPeerOutAnchor();
  ge::NodePtr input_node_1 = anchorptr_1->GetOwnerNode();
  ge::InDataAnchorPtr mul_anchorptr_2 = mul_node->GetInDataAnchor(1);
  ge::OutDataAnchorPtr anchorptr_2 = mul_anchorptr_2->GetPeerOutAnchor();
  ge::NodePtr input_node_2 = anchorptr_2->GetOwnerNode();
  // get mul input name
  string input1_node_name = input_node_1->GetOpDesc()->GetName();
  string input2_node_name = input_node_2->GetOpDesc()->GetName();
  // get mul input type
  string input1_node_type = input_node_1->GetType();
  string input2_node_type = input_node_2->GetType();
  ge::GeTensorDesc input_desc0 = mul_node->GetOpDesc()->GetInputDesc(0);
  // get mul input0 format and shape
  ge::Format input0_format = input_desc0.GetFormat();
  ge::GeShape input0_shape = input_desc0.GetShape();
  DataType datatype_mul_0 = input_desc0.GetDataType();

  ge::GeTensorDesc input_desc1 = mul_node->GetOpDesc()->GetInputDesc(1);
  // get mul input1 format and shape
  ge::Format input1_format = input_desc1.GetFormat();
  ge::GeShape input1_shape = input_desc1.GetShape();
  DataType datatype_mul_1 = input_desc1.GetDataType();
  std::set<DataType> supported_perm_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32};
  ge::GeTensorDesc output_desc0 = mul_node->GetOpDesc()->GetOutputDesc(0);

  // shape,dtype,format,node must be same
  if (input_node_2 == input_node_1 && datatype_mul_0 == datatype_mul_1 && anchorptr_1 == anchorptr_2 &&
      input0_format == input1_format && input0_shape == input1_shape &&
      supported_perm_dtypes.count(datatype_mul_0) != 0 && supported_perm_dtypes.count(datatype_mul_1) != 0) {
    ge::OpDescPtr square_op;
    FUSION_PASS_MAKE_SHARED((square_op = std::make_shared<ge::OpDesc>(mul_node->GetName() + "/" + SQUARE, SQUARE)),
                            return INTERNAL_ERROR);
    square_op->AddInputDesc("x", input_desc0);
    square_op->AddOutputDesc("y", output_desc0);
    ge::NodePtr square_node = graph.AddNode(square_op);
    newNodes.push_back(square_node);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(mul_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                         square_node->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add square node in data edge failed."), return FAILED);
    if (mul_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of layermulnode is [%d].",
              mul_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
      for (InDataAnchorPtr inAnchorPtr : mul_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
        inAnchorPtr->UnlinkAll();
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(square_node->GetOutDataAnchor(0), inAnchorPtr),
                          OP_LOGE(FUSED_OP_TYPE.c_str(),
                                  "Add edge from fused node:%s's 2nd index to fusion "
                                  "node:%s's 1st index failed.",
                                  mul_node->GetName().c_str(), square_node->GetName().c_str()),
                          return FAILED);
        OP_LOGD(FUSED_OP_TYPE.c_str(),
                "Add edge from fused node:%s's 2nd index to fusion node:%s's 1st "
                "index.",
                mul_node->GetName().c_str(), square_node->GetName().c_str());
      }
    }
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(mul_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                            mul_node->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mul node in data0 edge failed."), return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(mul_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                                            mul_node->GetInDataAnchor(1)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mul node in data1 edge failed."), return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(mul_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mul node failed."),
                      return FAILED);
    return SUCCESS;
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "MulSquareFusionPass fusion NOT_CHANGED");
    return NOT_CHANGED;
  }
}

REGISTER_PASS("MulSquareFusionPass", BUILT_IN_GRAPH_PASS, MulSquareFusionPass);
}  // namespace fe
