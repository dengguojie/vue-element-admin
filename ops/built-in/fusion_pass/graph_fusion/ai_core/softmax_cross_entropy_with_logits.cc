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
 * \file softmax_cross_entropy_with_logits.cpp
 * \brief
 */
#include "softmax_cross_entropy_with_logits.h"

#include <iostream>
#include <vector>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_RESHAPE = "Reshape";
static const string PATTERN_RESHAPE_1 = "Reshape_1";
static const string PATTERN_RESHAPE_2 = "Reshape_2";
static const string PATTERN_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS = "SoftmaxCrossEntropyWithLogits";
static const string PATTERN_INPUTS = "input";

/*
               |
            Reshape_2
           /        \
          /      xentropy
         Const    /     \
                 /       \
              Reshape  Reshape_1
                /          \
               /            \
*/
Status SoftmaxCrossEntropyWithLogitsPass::RemoveNode(ge::NodePtr node, ge::ComputeGraph& graph) {
  // remove input data edge
  for (size_t i = 0; i < node->GetAllInDataAnchors().size(); ++i) {
    auto inDataAnchor = node->GetInDataAnchor(i);
    FUSION_PASS_CHECK(inDataAnchor == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "inDataAnchor is null, remove node failed."),
                                                     return FAILED);
    auto preOutDataAnchor = inDataAnchor->GetPeerOutAnchor();
    FUSION_PASS_CHECK(preOutDataAnchor == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "preOutDataAnchor is null, remove node failed."),
                                                     return FAILED);

    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(preOutDataAnchor, inDataAnchor) != ge::GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove node failed."), return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge %u of node %s", i, node->GetName().c_str());
  }
  // delete the node
  FUSION_PASS_CHECK(graph.RemoveNode(node) != ge::GRAPH_SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "remove node failed"),
                    return FAILED);
  return SUCCESS;
}

vector<FusionPattern*> SoftmaxCrossEntropyWithLogitsPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxCrossEntropyWithLogitsPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SoftmaxCrossEntropyWithLogitsFusion");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object failed"),
                    return patterns);

  pattern->AddOpDesc(PATTERN_RESHAPE, {"Reshape"})
      .AddOpDesc(PATTERN_RESHAPE_1, {"Reshape"})
      .AddOpDesc(PATTERN_RESHAPE_2, {"Reshape"})
      .AddOpDesc(PATTERN_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS, {"SoftmaxCrossEntropyWithLogits"})
      .AddOpDesc(PATTERN_INPUTS)
      .SetInputs(PATTERN_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS, {PATTERN_RESHAPE, PATTERN_RESHAPE_1})
      .SetInputs(PATTERN_RESHAPE_2, {PATTERN_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS, PATTERN_INPUTS})
      .SetOutput(PATTERN_RESHAPE_2);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxCrossXentropyWithLogitsPass pattern end");
  return patterns;
}

Status SoftmaxCrossEntropyWithLogitsPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                 vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxCrossEntropyWithLogitsPass fusion begin");
  ge::NodePtr reshape = GetNodeFromMapping(PATTERN_RESHAPE, mapping);
  ge::NodePtr reshape_1 = GetNodeFromMapping(PATTERN_RESHAPE_1, mapping);
  ge::NodePtr reshape_2 = GetNodeFromMapping(PATTERN_RESHAPE_2, mapping);
  ge::NodePtr softmax_cross_entropy_with_logits =
      GetNodeFromMapping(PATTERN_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS, mapping);

  FUSION_PASS_CHECK(reshape == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reshape is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(reshape_1 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reshape_1 is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(reshape_2 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reshape_2 is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(softmax_cross_entropy_with_logits == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "softmax_cross_entropy_with_logits is null"),
                                                   return PARAM_INVALID);

  // get xentropy input Tensor
  ge::GeTensorDesc first_input_tensor = reshape->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc second_input_tensor = reshape_1->GetOpDesc()->GetInputDesc(0);
  // if input data format is not 4D, will return
  if (first_input_tensor.GetShape().GetDims().size() != 4 or second_input_tensor.GetShape().GetDims().size() != 4) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "Input shape is not 4D, needn't to fusion, SoftmaxCrossEntropyWithLogitsPass fusion end");
    return SUCCESS;
  }

  // check xentropy first output Tensor shape
  ge::GeTensorDesc first_output_tensor = reshape_2->GetOpDesc()->GetOutputDesc(0);
  if (first_output_tensor.GetShape().GetDims().size() != 4) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "Output shape is not 4D, cannot do fusion, SoftmaxCrossEntropyWithLogitsPass fussion end");
    return SUCCESS;
  }

  // 4D xentropy only support NCHW format input, first output is NHWC, second output is NCHW
  // create the Shape for second output of xentropy according the first input tensor, format: NCHW
  ge::GeShape first_input_tensor_shape = first_input_tensor.GetShape();
  ge::GeShape second_input_tensor_shape = second_input_tensor.GetShape();
  ge::GeShape first_output_tensor_shape = first_output_tensor.GetShape();
  ge::GeShape second_output_tensor_shape = first_input_tensor.GetShape();

  if (first_input_tensor.GetFormat() == ge::FORMAT_NHWC) {
    // change NHWC to NCHW
    second_output_tensor_shape.SetDim(1, first_input_tensor_shape.GetDim(3));
    second_output_tensor_shape.SetDim(2, first_input_tensor_shape.GetDim(1));
    second_output_tensor_shape.SetDim(3, first_input_tensor_shape.GetDim(2));
  }

  // get xentropy input and output Tensor
  ge::GeTensorDesc first_input = softmax_cross_entropy_with_logits->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc second_input = softmax_cross_entropy_with_logits->GetOpDesc()->GetInputDesc(1);
  ge::GeTensorDesc first_output = softmax_cross_entropy_with_logits->GetOpDesc()->GetOutputDesc(0);
  ge::GeTensorDesc second_output = softmax_cross_entropy_with_logits->GetOpDesc()->GetOutputDesc(1);

  first_input.SetShape(second_output_tensor_shape);
  second_input.SetShape(second_output_tensor_shape);
  first_output.SetShape(first_output_tensor_shape);
  second_output.SetShape(second_output_tensor_shape);

  // only support NCHW
  first_input.SetFormat(ge::FORMAT_NCHW);
  second_input.SetFormat(ge::FORMAT_NCHW);
  second_output.SetFormat(ge::FORMAT_NCHW);

  first_input.SetOriginShape(first_input_tensor_shape);
  second_input.SetOriginShape(second_input_tensor_shape);
  first_output.SetOriginShape(first_output_tensor_shape);
  second_output.SetOriginShape(first_input_tensor_shape);

  softmax_cross_entropy_with_logits->GetOpDesc()->UpdateInputDesc(0, first_input);
  softmax_cross_entropy_with_logits->GetOpDesc()->UpdateInputDesc(1, second_input);
  softmax_cross_entropy_with_logits->GetOpDesc()->UpdateOutputDesc(0, first_output);
  softmax_cross_entropy_with_logits->GetOpDesc()->UpdateOutputDesc(1, second_output);

  auto reshapeInDataAnchor = reshape->GetInDataAnchor(0);
  FUSION_PASS_CHECK(reshapeInDataAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "reshapeInDataAnchor is null"),
                    return FAILED);
  auto reshapePeerOutDataAnchor = reshapeInDataAnchor->GetPeerOutAnchor();

  auto reshape1InDataAnchor = reshape_1->GetInDataAnchor(0);
  FUSION_PASS_CHECK(reshapeInDataAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "reshapeInDataAnchor is null"),
                    return FAILED);
  auto reshape1PeerOutDataAnchor = reshape1InDataAnchor->GetPeerOutAnchor();

  auto reshape2OutDataAnchor = reshape_2->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(reshape2OutDataAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "reshape2OutDataAnchor is null"),
                    return FAILED);
  auto reshapePeerInDataAnchors = reshape2OutDataAnchor->GetPeerInDataAnchors();

  auto inputDataAnchor1 = softmax_cross_entropy_with_logits->GetInDataAnchor(0);
  FUSION_PASS_CHECK(inputDataAnchor1 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "inputDataAnchor1 is null"),
                    return FAILED);

  auto inputDataAnchor2 = softmax_cross_entropy_with_logits->GetInDataAnchor(1);
  FUSION_PASS_CHECK(inputDataAnchor2 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "inputDataAnchor2 is null"),
                    return FAILED);

  auto outDataAnchor = softmax_cross_entropy_with_logits->GetOutDataAnchor(0);

  // delete reshape node
  FUSION_PASS_CHECK(RemoveNode(reshape, graph) == FAILED, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "remove reshape node failed"),
                    return FAILED);
  FUSION_PASS_CHECK(RemoveNode(reshape_1, graph) == FAILED,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove reshape1 node failed"),
                                                   return FAILED);
  FUSION_PASS_CHECK(RemoveNode(reshape_2, graph) == FAILED,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove reshape2 node failed"),
                                                   return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(reshapePeerOutDataAnchor, inputDataAnchor1) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input edge 1 error"), return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(reshape1PeerOutDataAnchor, inputDataAnchor2) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input edge 2 error"), return FAILED);

  for (unsigned int i = 0; i < reshapePeerInDataAnchors.size(); ++i) {
    ge::InDataAnchorPtr dstAnchor = reshapePeerInDataAnchors.at(i);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(outDataAnchor, dstAnchor) != ge::GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output anchor Failed."),
                                                     return FAILED);
  }
  fusionNodes.push_back(softmax_cross_entropy_with_logits);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxCrossEntropyWithLogitsPass fusion end");
  return SUCCESS;
}

REGISTER_PASS("SoftmaxCrossEntropyWithLogitsPass", BUILT_IN_GRAPH_PASS, SoftmaxCrossEntropyWithLogitsPass);
}  // namespace fe
