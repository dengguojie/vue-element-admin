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
 * \file biasadd_conv_fusion_pass.cpp
 * \brief conv-biasadd fusion pass(conv-biasadd --> conv)
 */
#include "biasadd_conv_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "op_log.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "error_util.h"

using namespace ge;
namespace fe {
static const string PATTERN_SRC = "src";
static const string PATTERN_BIASADD = "biasadd";
static const char* CONVOLUTION = "Conv2D";
static const char* DEPTHWISECONVOLUTION = "DepthwiseConv2D";
static const char* BIASADD = "BiasAdd";
static const char* CONVOLUTION_3D = "Conv3D";
static const char *VARIABLE = "Variable";
static const char* ADD_3D = "Add";
static const int64_t DIM1 = 1;
static const int64_t DIM_COUNT = 4;

vector<FusionPattern*> BiasaddConvFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("BiasaddFusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);

  pattern->AddOpDesc(PATTERN_BIASADD, {BIASADD, ADD_3D})
      .AddOpDesc(PATTERN_SRC, {CONVOLUTION, DEPTHWISECONVOLUTION, CONVOLUTION_3D})
      .SetInputs(PATTERN_BIASADD, {PATTERN_SRC})
      .SetOutput(PATTERN_BIASADD);
  patterns.push_back(pattern);

  return patterns;
}

vector<ge::NodePtr> BiasaddConvFusionPass::GetConstOrDataInputs(const ge::NodePtr &node) {
  vector<ge::NodePtr> ret;
  auto in_anchors = node->GetAllInDataAnchors();
  for (const auto &in_anchor : in_anchors) {
    auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) continue;

    auto in_node = out_anchor->GetOwnerNode();
    if (in_node->GetType() == "Const") {
      ret.push_back(in_node);
    } else if (in_node->GetType() == "Switch" && node->GetType() == "Matmul") {
      // const --> switch --> matmul
      auto switch_input = GetConstOrDataInputs(in_node);
      if (!switch_input.empty()) {
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

Status BiasaddConvFusionPass::AdjustShapeOfBiasWeight(
    const vector<ge::ConstGeTensorPtr> &weights, ge::GeShape &bias_shape) {
  ge::ConstGeTensorPtr biases = weights[0];

  if (biases == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Biasadd node weight is null, fusion failed.");
    return NOT_CHANGED;
  }
  int64_t new_dim = 1;
  int64_t dim1Count = 0;

  if (biases->GetTensorDesc().GetShape().GetDims().size() != 1) {
    for (int64_t dim : biases->GetTensorDesc().GetShape().GetDims()) {
      if (PatternFusionUtil::IsUnknownShape(dim)) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "BiasaddConvFusionPass cannot be applied for unknown shape.");
        return NOT_CHANGED;
      }
      if (dim == DIM1) {
        dim1Count++;
      } else {
        new_dim = dim;
      }
    }
    if (dim1Count < DIM_COUNT) {
      return NOT_CHANGED;
    }
    std::vector<int64_t> newDimVec;
    newDimVec.emplace_back(new_dim);
    bias_shape = ge::GeShape(newDimVec);
  } else {
    bias_shape = biases->GetTensorDesc().GetShape();
  }

  return SUCCESS;
}

Status BiasaddConvFusionPass::GetWeightNode(const ge::NodePtr &biasadd_node, const ge::NodePtr conv,
    ge::NodePtr &weight_node, ge::GeShape &bias_shape) {
  auto biasaddPeerAnchor = biasadd_node->GetInDataAnchor(1)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(biasaddPeerAnchor == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "biasaddPeerAnchor is null, fusion failed."),
                    return NOT_CHANGED);
  auto nodeInfrontOfAdd = biasadd_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
  bool case_training = (nodeInfrontOfAdd->GetType() == VARIABLE);

  if (case_training) {
    FUSION_PASS_CHECK(biasadd_node->GetType() == ADD_3D,
                                          OP_LOGI(FUSED_OP_TYPE.c_str(),
                                                          "We do not support fusion of Conv2D + Add in training mode."),
                                          return NOT_CHANGED);
    bias_shape = nodeInfrontOfAdd->GetOpDesc()->MutableOutputDesc(0)->GetShape();
    weight_node = nodeInfrontOfAdd;
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Do BiasaddConvFusionPass for op %s in training mode.", conv->GetName().c_str());
  } else {
    // Inference mode
    vector<ge::ConstGeTensorPtr> weights = ge::OpDescUtils::GetWeights(biasadd_node);
    vector<ge::NodePtr> const_input_nodes = GetConstOrDataInputs(biasadd_node);

    bool case_conv3d_with_reshape = (biasadd_node->GetType() == ADD_3D && conv->GetType() == CONVOLUTION_3D);
    if (case_conv3d_with_reshape) {
      bool checkNull = biasadd_node->GetInDataAnchor(1) == nullptr ||
                       biasadd_node->GetInDataAnchor(1)->GetPeerOutAnchor() == nullptr ||
                       biasadd_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode() == nullptr;

      FUSION_PASS_CHECK(checkNull,
                        OP_LOGI(FUSED_OP_TYPE.c_str(), "The input of add %s is null!", biasadd_node->GetName().c_str()),
                        return NOT_CHANGED);

      if (nodeInfrontOfAdd->GetType() == "Reshape") {
        ge::OpDescPtr input_tensor_op = nodeInfrontOfAdd->GetOpDesc();
        auto input_tensor_0 = input_tensor_op->GetInputDesc(0);
        std::vector<int64_t> input_size = input_tensor_0.GetShape().GetDims();
        FUSION_PASS_CHECK(input_size.size() != 1, OP_LOGI(FUSED_OP_TYPE.c_str(),
                          "Node Add:[%s]'s input size %zu is invalid.",
                          nodeInfrontOfAdd->GetName().c_str(), input_size.size()),
                          return NOT_CHANGED);
        /* This case, the BiasAdd is Add and the input of Add is Reshape,
         * we just get the first weight of reshape as the bias. */
        weights = ge::OpDescUtils::GetWeights(nodeInfrontOfAdd);
        const_input_nodes = GetConstOrDataInputs(nodeInfrontOfAdd);
        FUSION_PASS_CHECK(const_input_nodes[0]->GetOpDesc()->GetOutputDesc(0).GetDataType() != ge::DT_FLOAT16 &&
                          const_input_nodes[0]->GetOpDesc()->GetOutputDesc(0).GetDataType() != ge::DT_FLOAT,
                          OP_LOGI(FUSED_OP_TYPE.c_str(), "Node Add:[%s]'s dtype %u is invalid.",
                          nodeInfrontOfAdd->GetName().c_str(),
                          const_input_nodes[0]->GetOpDesc()->GetOutputDesc(0).GetDataType()),
                          return NOT_CHANGED);
      }
    } // for other cases, we just use the weight of BiasAdd

    FUSION_PASS_CHECK(weights.size() != 1, OP_LOGI(FUSED_OP_TYPE.c_str(), "Node Add:[%s]'s weight size %u is invalid.",
                                                                                           nodeInfrontOfAdd->GetName().c_str(), weights.size()),
                                          return NOT_CHANGED);
    FUSION_PASS_CHECK(AdjustShapeOfBiasWeight(weights, bias_shape) != SUCCESS,
                                          OP_LOGI(FUSED_OP_TYPE.c_str(), "The bias shape is not valid for node %s!", biasadd_node->GetName().c_str()),
                                          return NOT_CHANGED);
    FUSION_PASS_CHECK(const_input_nodes.empty(), OP_LOGI(FUSED_OP_TYPE.c_str(), "The bias %s's weight is empty!",
                                                         biasadd_node->GetName().c_str()),
                      return NOT_CHANGED);
    weight_node = const_input_nodes[0];
  }
  return SUCCESS;
}

Status BiasaddConvFusionPass::CheckParam(const ge::NodePtr &conv, const ge::NodePtr &biasadd_node) {
  FUSION_PASS_CHECK(conv == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node conv2d is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(biasadd_node == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node BiasAdd is null, fusion failed."),
                    return PARAM_INVALID);

  if (conv->GetOutDataNodes().size() > 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "out data size is invalid.");
    return NOT_CHANGED;
  }

  int32_t in_edges_size = conv->GetInDataNodes().size();
  if (in_edges_size < 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "inEdges size is invalid.");
    return NOT_CHANGED;
  }

  if (conv->GetInDataAnchor(2)->GetPeerOutAnchor() != nullptr) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Biasadd_conv_fusion can not support conv op which has already bias");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status BiasaddConvFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter BiasaddConvFusionPass");
  ge::NodePtr conv = GetNodeFromMapping(PATTERN_SRC, mapping);
  ge::NodePtr biasadd_node = GetNodeFromMapping(PATTERN_BIASADD, mapping);
  FUSION_PASS_CHECK(biasadd_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "biasadd_node is null, fusion failed."),
                    return PARAM_INVALID);
  Status result = CheckParam(conv, biasadd_node);
  FUSION_PASS_CHECK(result != SUCCESS, , return result);

  ge::OpDescPtr src_op = conv->GetOpDesc();
  OP_LOGI(FUSED_OP_TYPE.c_str(), "BiasaddConvFusionPass: conv2d [%s] has %u input anchor.", conv->GetName().c_str(),
          conv->GetAllInDataAnchors().size());
  OP_LOGI(FUSED_OP_TYPE.c_str(), "BiasaddConvFusionPass: conv2d [%s] has %u input desc.", conv->GetName().c_str(),
          src_op->GetAllInputsDesc().size());

  std::map<string, uint32_t> inputNameMap = src_op->GetAllInputName();
  ge::OpDescPtr biasadd_op = biasadd_node->GetOpDesc();
  FUSION_PASS_CHECK(biasadd_op == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                    biasadd_node->GetName().c_str()),
                    return PARAM_INVALID);

  ge::NodePtr weight_node;
  ge::GeShape bias_shape;
  result = GetWeightNode(biasadd_node, conv, weight_node, bias_shape);
  if (result != SUCCESS) {
    return result;
  }

  FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::LinkControlEdge(biasadd_node, conv),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Link control edge from [%s] to [%s] failed",
                                          biasadd_node->GetName().c_str(), conv->GetName().c_str()),
                    return FAILED);

  GeTensorDesc constTensor = weight_node->GetOpDesc()->GetOutputDesc(0);
  constTensor.SetShape(bias_shape);
  constTensor.SetOriginShape(bias_shape);
  ge::Format inputFormat = biasadd_op->GetInputDesc(0).GetFormat();
  constTensor.SetFormat(inputFormat);
  inputFormat = biasadd_op->GetInputDesc(0).GetOriginFormat();
  constTensor.SetOriginFormat(inputFormat);

  weight_node->GetOpDesc()->UpdateOutputDesc(0, constTensor);

  FUSION_PASS_CHECK(conv->AddLinkFrom(2, weight_node) != ge::GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(conv->GetName().c_str(), "Fail to link const node with conv node."),
                    return FAILED);

  FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(conv->GetOpDesc(), ge::MATMUL_HAS_BIAS, true),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Biasadd op weight should be 1-D."), return FAILED);
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
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                          "Remove node:[%s] failed", biasadd_node->GetName().c_str()),
                    return FAILED);
  fusionNodes.push_back(conv);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "BiasaddConvFusionPass fusion success.");

  return SUCCESS;
}

REGISTER_PASS("AABiasaddConvFusion", BUILT_IN_GRAPH_PASS, BiasaddConvFusionPass);
}  // namespace fe
