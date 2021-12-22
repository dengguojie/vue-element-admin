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
 * \file softmax_cross_entropy_with_logits_grad.cpp
 * \brief
 */
#include "softmax_cross_entropy_with_logits_grad.h"

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
static const string PATTERN_RESHAPE_GRAD_2 = "Reshape_2_grad";
static const string PATTERN_EXPANDDIMS = "ExpandDims";
static const string PATTERN_INPUTS = "input";
static const string PATTERN_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS = "SoftmaxCrossEntropyWithLogits";
static const string PATTERN_MUL = "mul";
static const string PATTERN_RESHAPE_GRAD = "Reshape_grad";

Status SoftmaxCrossEntropyWithLogitsGradPass::RemoveNode(ge::NodePtr node, ge::ComputeGraph& graph) {
  // remove input data edge
  for (size_t i = 0; i < node->GetAllInDataAnchors().size(); ++i) {
    auto inDataAnchor = node->GetInDataAnchor(i);
    FUSION_PASS_CHECK(inDataAnchor == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "inDataAnchor is null, remove node failed."),
                                                     return FAILED);
    auto preOutDataAnchor = inDataAnchor->GetPeerOutAnchor();
    if (preOutDataAnchor == nullptr) {
      continue;
    }
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(preOutDataAnchor, inDataAnchor) != ge::GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge failed."), return FAILED);
  }

  // delete node
  FUSION_PASS_CHECK(graph.RemoveNode(node) != ge::GRAPH_SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "remove node failed"),
                    return FAILED);
  return SUCCESS;
}

vector<FusionPattern*> SoftmaxCrossEntropyWithLogitsGradPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxCrossEntropyWithLogitsGradPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SoftmaxCrossEntropyWithLogitsGradFussion");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object failed"),
                    return patterns);

  pattern->AddOpDesc(PATTERN_RESHAPE_GRAD_2, {"Reshape"})
      .AddOpDesc(PATTERN_EXPANDDIMS, {"ExpandDims"})
      .AddOpDesc(PATTERN_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS, {"SoftmaxCrossEntropyWithLogits"})
      .AddOpDesc(PATTERN_MUL, {"Mul"})
      .AddOpDesc(PATTERN_RESHAPE_GRAD, {"Reshape"})
      .AddOpDesc(PATTERN_INPUTS)
      .SetInputs(PATTERN_EXPANDDIMS, {PATTERN_RESHAPE_GRAD_2, PATTERN_INPUTS})
      .SetInputs(PATTERN_MUL, {PATTERN_EXPANDDIMS, PATTERN_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS})
      .SetInputs(PATTERN_RESHAPE_GRAD, {PATTERN_MUL, PATTERN_INPUTS})
      .SetOutput(PATTERN_RESHAPE_GRAD);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxCrossEntropyWithLogitsGradPass pattern end");
  return patterns;
}

Status SoftmaxCrossEntropyWithLogitsGradPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                     vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxCrossEntropyWithLogitsGradPass fusion begin");
  ge::NodePtr reshape_grad_2 = GetNodeFromMapping(PATTERN_RESHAPE_GRAD_2, mapping);
  ge::NodePtr expandDim = GetNodeFromMapping(PATTERN_EXPANDDIMS, mapping);
  ge::NodePtr xentropy = GetNodeFromMapping(PATTERN_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS, mapping);
  ge::NodePtr mul = GetNodeFromMapping(PATTERN_MUL, mapping);
  ge::NodePtr reshape_grad = GetNodeFromMapping(PATTERN_RESHAPE_GRAD, mapping);

  FUSION_PASS_CHECK(reshape_grad_2 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "reshape_grad_2 is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(expandDim == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "expandDim is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(xentropy == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "xentropy is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mul is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(reshape_grad == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "reshape_grad is null"),
                    return PARAM_INVALID);

  // get the first input of Reshape_2_grad, if is 4D do fusion; else return
  ge::GeTensorDesc grad_2_input_tensor = reshape_grad_2->GetOpDesc()->GetInputDesc(0);
  ge::GeShape first_input_tensor_shape = grad_2_input_tensor.GetShape();
  ge::GeTensorDesc grad_output_tensor = reshape_grad->GetOpDesc()->GetOutputDesc(0);
  ge::GeShape grad_output_shape = grad_output_tensor.GetShape();
  if (first_input_tensor_shape.GetDims().size() != 4) {  // if not 4D, will return
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "Input shape is not 4D, needn't to fusion, SoftmaxCrossEntropyWithLogitsGradPass fusion end");
    return SUCCESS;
  }
  if (grad_output_shape.GetDims().size() != 4) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "output shape is not 4D, needn't to fusion");
    return SUCCESS;
  }

  ge::GeTensorDesc first_input = mul->GetOpDesc()->GetInputDesc(0);    // get xentropy_grad first input Tensor
  ge::GeTensorDesc second_input = mul->GetOpDesc()->GetInputDesc(1);   // get xentropy_grad second input Tensor
  ge::GeTensorDesc first_output = mul->GetOpDesc()->GetOutputDesc(0);  // get xentropy_grad first output Tensor

  // create the Shape according to the Reshape_grad_2
  first_input.SetShape(first_input_tensor_shape);
  first_input.SetOriginShape(first_input_tensor_shape);

  ge::GeShape xentropy_grad_tensor_shape = grad_2_input_tensor.GetShape();
  size_t org_sec_input_shape_size = second_input.GetShape().GetDimNum();

  // update the last dimension of input and output shape according original input shape
  xentropy_grad_tensor_shape.SetDim(3, second_input.GetShape().GetDim(org_sec_input_shape_size - 1));
  second_input.SetShape(xentropy_grad_tensor_shape);
  second_input.SetOriginShape(xentropy_grad_tensor_shape);
  first_output.SetShape(xentropy_grad_tensor_shape);
  first_output.SetOriginShape(xentropy_grad_tensor_shape);

  mul->GetOpDesc()->UpdateInputDesc(0, first_input);
  mul->GetOpDesc()->UpdateInputDesc(1, second_input);
  mul->GetOpDesc()->UpdateOutputDesc(0, first_output);

  auto reshapeGrad2InDataAnchor = reshape_grad_2->GetInDataAnchor(0);
  FUSION_PASS_CHECK(reshapeGrad2InDataAnchor == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reshapeGrad2InDataAnchor is null"),
                                                   return FAILED);
  auto reshapeGrad2PeerOutDataAnchor = reshapeGrad2InDataAnchor->GetPeerOutAnchor();

  auto mulInDataAnchor = mul->GetInDataAnchor(0);
  FUSION_PASS_CHECK(mulInDataAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "mulInDataAnchor is null"),
                    return FAILED);

  auto mulPeerOutDataAnchor = mulInDataAnchor->GetPeerOutAnchor();
  FUSION_PASS_CHECK(mulPeerOutDataAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "mulPeerOutDataAnchor is null"),
                    return FAILED);

  auto mulOutDataAnchor = mul->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(mulOutDataAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "mulOutDataAnchor is null"),
                    return FAILED);

  auto reshapeGradOutDataAnchor = reshape_grad->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(reshapeGradOutDataAnchor == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reshapeGradOutDataAnchor is null"),
                                                   return FAILED);

  auto reshapeGradPeerInDataAnchors = reshapeGradOutDataAnchor->GetPeerInDataAnchors();
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(mulPeerOutDataAnchor, mulInDataAnchor) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove inputdata edge error"),
                                                   return FAILED);

  // delete reshape node
  FUSION_PASS_CHECK(RemoveNode(reshape_grad_2, graph) == FAILED,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove reshape_grad_2 node failed"),
                                                   return FAILED);
  FUSION_PASS_CHECK(RemoveNode(expandDim, graph) == FAILED,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove expandDims node failed"),
                                                   return FAILED);
  FUSION_PASS_CHECK(RemoveNode(reshape_grad, graph) == FAILED,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove reshape_grad node failed"),
                                                   return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(reshapeGrad2PeerOutDataAnchor, mulInDataAnchor) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input edge 0 error"), return FAILED);

  for (unsigned int i = 0; i < reshapeGradPeerInDataAnchors.size(); ++i) {
    ge::InDataAnchorPtr dstAnchor = reshapeGradPeerInDataAnchors.at(i);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mulOutDataAnchor, dstAnchor) != ge::GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output anchor Failed."),
                                                     return FAILED);
  }
  fusionNodes.push_back(xentropy);
  fusionNodes.push_back(mul);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxCrossEntropyWithLogitsGradPass fusion end");
  return SUCCESS;
}

REGISTER_PASS("SoftmaxCrossEntropyWithLogitsGradPass", BUILT_IN_GRAPH_PASS, SoftmaxCrossEntropyWithLogitsGradPass);
}  // namespace fe
