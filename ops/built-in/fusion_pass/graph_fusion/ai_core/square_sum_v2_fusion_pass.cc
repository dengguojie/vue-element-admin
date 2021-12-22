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

/*!
 * \file square_sum_v2_fusion_pass.cpp
 * \brief square_sum_v2 fusion pass( --> square_sum_v2)
 */
#include "square_sum_v2_fusion_pass.h"

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {
static const char* REDUCESUMD = "ReduceSumD";
static const char* SQUARE = "Square";

static const string PATTERN_SQUARE = "Square";
static const string PATTERN_REDUCESUMD = "ReduceSumD";
static const string AXES = "axes";
static const string AXIS = "axis";
static const string KEEPDIMS = "keep_dims";

static const string SQUARESUMV2 = "SquareSumV2";

/*
          input0
            |
            |
          square0:0
          /    \
         /      \
      output1  reduceSumD
                 |
                 |
              output0
*/

vector<FusionPattern*> SquareSumV2FusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SquareSumV2 pattern begin");
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("SquareSumV2FusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_REDUCESUMD, {REDUCESUMD})
      .AddOpDesc(PATTERN_SQUARE, {SQUARE})
      .SetInputs(PATTERN_REDUCESUMD, {PATTERN_SQUARE})
      .SetOutput(PATTERN_REDUCESUMD);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SquareSumV2FusionPass pattern end");
  return patterns;
}

Status SquareSumV2FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SquareSumV2FusionPass fusion begin");

  // 1.get original nodes
  ge::NodePtr square_node = GetNodeFromMapping(PATTERN_SQUARE, mapping);
  ge::NodePtr sum_node = GetNodeFromMapping(PATTERN_REDUCESUMD, mapping);

  FUSION_PASS_CHECK(sum_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "sum node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(square_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "square_node node is null, fusion failed."),
                    return PARAM_INVALID);

  // get attrs num_classes and dtype of new node
  std::vector<int64_t> axis;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(sum_node->GetOpDesc(), AXES, axis),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get axis attr failed"), return FAILED);
  bool keep_dims;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(sum_node->GetOpDesc(), KEEPDIMS, keep_dims),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get keep_dims attr failed."), return FAILED);

  vector<int64_t> dims = square_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  FUSION_PASS_CHECK(dims.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "input dims is empty, fusion failed."),
                    return PARAM_INVALID);

  std::vector<int64_t> axis_temp;  // change negative axis to positive
  for (std::vector<int64_t>::iterator axis_it = axis.begin(); axis_it < axis.end(); ++axis_it) {
    if (*axis_it < 0) {
      axis_temp.push_back(*axis_it + dims.size());
    } else {
      axis_temp.push_back(*axis_it);
    }
  }

  int32_t index;
  int32_t multi = dims[dims.size() - 1];

  for (std::vector<int64_t>::iterator it = dims.end() - 2; it >= dims.begin(); --it) {
    index = it - dims.begin();
    if ((std::find(axis_temp.begin(), axis_temp.end(), index) != axis_temp.end()) ==
        (std::find(axis_temp.begin(), axis_temp.end(), index + 1) != axis_temp.end())) {
      multi *= *it;
    } else {
      break;
    }
  }

  DataType input_dtype = square_node->GetOpDesc()->GetInputDesc(0).GetDataType();
  int size_of_dtype = GetSizeByDataType(input_dtype);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "reduce num %d.", (int)multi);
  if (((multi * size_of_dtype) % 32 != 0) || (dims[dims.size() - 1] * size_of_dtype < 32)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "reduce num does not align 32B, square_sum_v2 not support");
    return NOT_CHANGED;
  }

  FUSION_PASS_CHECK(
      square_node->GetOutAllNodes().size() != 2,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "square node[%s] don't have output1 out node.", square_node->GetName().c_str()),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      square_node->GetOutDataAnchor(0)->GetFirstPeerAnchor()->GetOwnerNode()->GetName() == sum_node->GetName(),
      OP_LOGI(FUSED_OP_TYPE.c_str(), "sum node is not second output node of square"), return NOT_CHANGED);

  // 2.get input_node and output_node
  ge::NodePtr input_node = square_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  FUSION_PASS_CHECK(input_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "input node is null."), return NOT_CHANGED);
  // 3.define attrs of input edge based on original info
  ge::OpDescPtr square_sum_v2_op;
  FUSION_PASS_MAKE_SHARED(
      (square_sum_v2_op = std::make_shared<ge::OpDesc>(square_node->GetName() + "/" + SQUARESUMV2, SQUARESUMV2)),
      return INTERNAL_ERROR);
  ge::GeTensorDesc input_desc0 = square_node->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc output_desc0 = sum_node->GetOpDesc()->GetOutputDesc(0);
  ge::GeTensorDesc output_desc1 = square_node->GetOpDesc()->GetOutputDesc(0);
  square_sum_v2_op->AddInputDesc("input0", input_desc0);
  square_sum_v2_op->AddOutputDesc("output1", output_desc0);
  square_sum_v2_op->AddOutputDesc("output2", output_desc1);

  // 4.push back new node to graph
  ge::NodePtr square_sum_v2_node = graph.AddNode(square_sum_v2_op);
  newNodes.push_back(square_sum_v2_node);

  // 5. add edge for new node
  ge::OutDataAnchorPtr new_in_anchor_ptr0 = square_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  ge::GraphUtils::AddEdge(new_in_anchor_ptr0, square_sum_v2_node->GetInDataAnchor(0));

  for (auto in_data_anchor : sum_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(sum_node->GetOutDataAnchor(0), in_data_anchor),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove sum node out data edge failed."),
                                                     return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(square_sum_v2_node->GetOutDataAnchor(0), in_data_anchor),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Add square_sum_v2 node out data edge failed."),
                                                     return FAILED);
  }

  for (auto in_data_anchor : square_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    if (in_data_anchor->GetOwnerNode()->GetName() != sum_node->GetName()) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(square_node->GetOutDataAnchor(0), in_data_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                       "Remove square node out data edge failed."),
                                                       return FAILED);
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(square_sum_v2_node->GetOutDataAnchor(1), in_data_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                       "Add square_sum_v2 node out data edge failed."),
                                                       return FAILED);
    }
  }
  ge::OpDescPtr square_sum_v2_desc = square_sum_v2_node->GetOpDesc();
  FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(square_sum_v2_desc, KEEPDIMS, keep_dims),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set keep_dims attr failed"), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(square_sum_v2_desc, AXIS, axis),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set axis attr failed"), return FAILED);

  // 7.add control edge to new node
  if (square_node->GetInControlAnchor()) {
    for (auto out_control_anchor : square_node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(out_control_anchor, square_sum_v2_node->GetInControlAnchor()) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add square_sum_v2 node input control edge failed."),
                                         return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_control_anchor, square_node->GetInControlAnchor()) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                       "Remove square node input control edge failed."),
                                                       return FAILED);
    }
  }
  if (sum_node->GetOutControlAnchor()) {
    for (auto in_control_anchor : sum_node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(square_sum_v2_node->GetOutControlAnchor(), in_control_anchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add sum node out control edge failed."),
                                         return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(sum_node->GetOutControlAnchor(), in_control_anchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                       "Remove sum node out control edge failed."),
                                                       return FAILED);
    }
  }

  // 8.remove input node and const node in subgraph
  FUSION_PASS_CHECK(graph.RemoveNode(square_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove square node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(sum_node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Remove sum node failed."),
                    return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SquareSumV2FusionPass fusion end");

  return SUCCESS;
}

REGISTER_PASS("SquareSumV2", BUILT_IN_GRAPH_PASS, SquareSumV2FusionPass);
}  // namespace fe