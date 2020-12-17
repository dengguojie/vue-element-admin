/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file biasadd_conv_fusion_pass.cpp
 * \brief conv-biasadd fusion pass(conv-biasadd --> conv)
 */
#include "biasadd_conv_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "op_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
namespace fe {
static const string PATTERN_SRC = "src";
static const string PATTERN_BIASADD = "biasadd";
static const char* CONVOLUTION = "Conv2D";
static const char* DEPTHWISECONVOLUTION = "DepthwiseConv2D";
static const char* BIASADD = "BiasAdd";
static const char* CONVOLUTION_3D = "Conv3D";
static const char* ADD_3D = "Add";
static const int64_t DIM1 = 1;
static const int64_t DIM_COUNT = 4;

vector<FusionPattern*> BiasaddConvFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("BiasaddFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);

  pattern->AddOpDesc(PATTERN_BIASADD, {BIASADD, ADD_3D})
      .AddOpDesc(PATTERN_SRC, {CONVOLUTION, DEPTHWISECONVOLUTION, CONVOLUTION_3D})
      .SetInputs(PATTERN_BIASADD, {PATTERN_SRC})
      .SetOutput(PATTERN_BIASADD);
  patterns.push_back(pattern);

  return patterns;
}

vector<ge::NodePtr> BiasaddConvFusionPass::GetConstOrDataInputs(const ge::Node &node) {
  vector<ge::NodePtr> ret;
  auto in_anchors = node.GetAllInDataAnchors();
  for (const auto &in_anchor : in_anchors) {
    auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) continue;

    auto in_node = out_anchor->GetOwnerNode();
    if (in_node->GetType() == "Const") {
      ret.push_back(in_node);
    } else if (in_node->GetType() == "Switch" && node.GetType() == "Matmul") {
      // const --> switch --> matmul
      auto switch_input = GetConstOrDataInputs(*in_node);
      if (switch_input.size() > 0) {
        ret.insert(ret.end(), switch_input.begin(), switch_input.end());
      }
    } else if (in_node->GetType() == "Data") {
      auto parent = NodeUtils::GetParentInput(in_node);
      if ((parent != nullptr) && (parent->GetType() == "Const")) {
        ret.push_back(in_node);
      }
    }
  }
  return ret;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status BiasaddConvFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter BiasaddConvFusionPass");
  ge::NodePtr src_node = GetNodeFromMapping(PATTERN_SRC, mapping);
  ge::NodePtr biasadd_node = GetNodeFromMapping(PATTERN_BIASADD, mapping);
  FUSION_PASS_CHECK(src_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Node conv2d is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(biasadd_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Node BiasAdd is null, fusion failed."),
                    return PARAM_INVALID);

  if (src_node->GetOutDataNodes().size() > 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "out data size is invalid.");
    return NOT_CHANGED;
  }
  vector<ge::ConstGeTensorPtr> weights = ge::OpDescUtils::GetWeights(biasadd_node);
  vector<ge::NodePtr> const_input_nodes = GetConstOrDataInputs(*biasadd_node);
  auto biasAddWeightSize = weights.size();
  if (biasAddWeightSize == 0 && biasadd_node->GetType() == ADD_3D && src_node->GetType() == CONVOLUTION_3D) {
    bool checkNull = biasadd_node->GetInDataAnchor(1) == nullptr ||
                     biasadd_node->GetInDataAnchor(1)->GetPeerOutAnchor() == nullptr ||
                     biasadd_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode() == nullptr;

    FUSION_PASS_CHECK(checkNull,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "The input of add %s is null!", biasadd_node->GetName().c_str()),
                      return NOT_CHANGED);
    auto nodeInfrontOfAdd = biasadd_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();

    if (nodeInfrontOfAdd->GetType() == "Reshape") {
      /* This case, the BiasAdd is Add and the input of Add is Reshape,
       * we just get the first weight of reshape as the bias. */
      weights = ge::OpDescUtils::GetWeights(nodeInfrontOfAdd);
      const_input_nodes = GetConstOrDataInputs(*nodeInfrontOfAdd);
      FUSION_PASS_CHECK(weights.empty(), OP_LOGI(FUSED_OP_TYPE.c_str(), "Node Add:[%s]'s weight size %u is invalid.",
                                                 nodeInfrontOfAdd->GetName().c_str(), weights.size()),
                        return NOT_CHANGED);
    } else {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The input of biasadd %s is invalid.", biasadd_node->GetName().c_str());
      return NOT_CHANGED;
    }
  } else {
    FUSION_PASS_CHECK(biasAddWeightSize != 1,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node BiasAdd:[%s]'s weight size %u is invalid.",
                              biasadd_node->GetName().c_str(), biasAddWeightSize),
                      return NOT_CHANGED);
    /* The weights will be the weight of BiasAdd node */
  }

  ge::OpDescPtr src_op = src_node->GetOpDesc();
  FUSION_PASS_CHECK(src_op == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                                               src_node->GetName().c_str()),
                    return PARAM_INVALID);
  std::map<string, uint32_t> inputNameMap = src_op->GetAllInputName();
  ge::OpDescPtr biasadd_op = biasadd_node->GetOpDesc();
  FUSION_PASS_CHECK(biasadd_op == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                                                   biasadd_node->GetName().c_str()),
                    return PARAM_INVALID);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "BiasaddConvFusionPass: conv2d [%s] has %u input anchor.", src_node->GetName().c_str(),
          src_node->GetAllInDataAnchors().size());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "BiasaddConvFusionPass: conv2d [%s] has %u input desc.", src_node->GetName().c_str(),
          src_op->GetAllInputsDesc().size());
  int32_t in_edges_size = src_node->GetInDataNodes().size();
  if (in_edges_size < 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "inEdges size is invalid.");
    return NOT_CHANGED;
  }

  ge::ConstGeTensorPtr biases = weights[0];
  FUSION_PASS_CHECK(biases == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Biasadd node's weight is null, fusion failed."),
                    return PARAM_INVALID);

  int64_t dim1Count = 0;
  int64_t newShape = 1;
  if (biases->GetTensorDesc().GetShape().GetDims().size() != 1) {
    for (int64_t dim : biases->GetTensorDesc().GetShape().GetDims()) {
      if (PatternFusionUtil::IsUnknownShape(dim)) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "BiasaddConvFusionPass cannot be applied for unknown shape.");
        return NOT_CHANGED;
      }
      if (dim == DIM1) {
        dim1Count++;
      } else {
        newShape = dim;
      }
    }
    if (dim1Count < DIM_COUNT) {
      return NOT_CHANGED;
    }
  } else {
    newShape = biases->GetTensorDesc().GetShape().GetDims()[0];
  }

  FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::LinkControlEdge(biasadd_node, src_node),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Link control edge from [%s] to [%s] failed",
                            biasadd_node->GetName().c_str(), src_node->GetName().c_str()),
                    return FAILED);

  std::vector<int64_t> newDimVec;
  newDimVec.push_back(newShape);
  ge::GeShape biasShape(newDimVec);
  if (const_input_nodes.empty()) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to get const node.");
    return FAILED;
  }
  ge::NodePtr const_node = const_input_nodes[0];
  GeTensorDesc constTensor = const_node->GetOpDesc()->GetOutputDesc(0);
  constTensor.SetShape(biasShape);
  constTensor.SetOriginShape(biasShape);
  ge::Format inputFormat = biasadd_op->GetInputDesc(0).GetFormat();
  constTensor.SetFormat(inputFormat);
  inputFormat = biasadd_op->GetInputDesc(0).GetOriginFormat();
  constTensor.SetOriginFormat(inputFormat);

  const_node->GetOpDesc()->UpdateOutputDesc(0, constTensor);

  FUSION_PASS_CHECK(src_node->AddLinkFrom(2, const_node) != ge::GRAPH_SUCCESS,
                    OP_LOGE(src_node->GetName().c_str(), "Fail to link const node with conv node."),
                    return FAILED);

  FUSION_PASS_CHECK(true != ge::AttrUtils::SetBool(src_node->GetOpDesc(), ge::MATMUL_HAS_BIAS, true),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Biasadd op weight should be 1-D."), return FAILED);
  GeTensorDesc convbiasTensor = src_op->GetInputDesc(2);
  GeTensorDesc inputTensor = src_op->GetInputDesc(0);
  convbiasTensor.SetOriginShape(convbiasTensor.GetShape());
  convbiasTensor.SetOriginDataType(convbiasTensor.GetDataType());
  convbiasTensor.SetOriginFormat(inputTensor.GetOriginFormat());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 2nd input datatype is %d.", src_op->GetInputDesc(2).GetDataType());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 2nd input origin datatype is %d.",
          src_op->GetInputDesc(2).GetOriginDataType());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 2nd input format is %d.", src_op->GetInputDesc(2).GetFormat());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 2nd input origin format is %d.", src_op->GetInputDesc(2).GetOriginFormat());
  src_op->UpdateInputDesc(2, convbiasTensor);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 2nd input datatype is %d.", src_op->GetInputDesc(2).GetDataType());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 2nd input origin datatype is %d.",
          src_op->GetInputDesc(2).GetOriginDataType());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 2nd input format is %d.", src_op->GetInputDesc(2).GetFormat());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D's 2nd input origin format is %d.", src_op->GetInputDesc(2).GetOriginFormat());

  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(biasadd_node),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove node:[%s] failed", biasadd_node->GetName().c_str()),
                    return FAILED);
  fusionNodes.push_back(src_node);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "BiasaddConvFusionPass fusion success.");

  return SUCCESS;
}

REGISTER_PASS("AABiasaddConvFusion", BUILT_IN_GRAPH_PASS, BiasaddConvFusionPass);
}  // namespace fe
