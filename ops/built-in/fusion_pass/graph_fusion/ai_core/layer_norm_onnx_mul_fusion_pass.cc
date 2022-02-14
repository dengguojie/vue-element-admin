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
 * \file layer_norm_fusion_pass.cpp
 * \brief layer norm fusion pass
 */
#include "layer_norm_onnx_mul_fusion_pass.h"

#include <numeric>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "fp16_t.hpp"

using namespace ge;
namespace fe {
static const char* MEAN = "ReduceMeanD";
static const char* SUB = "Sub";
static const char* MUL = "Mul";
static const char* POW = "Pow";
static const char* DIV = "RealDiv";
static const char* ADD = "Add";
static const char* RSQRT = "Sqrt";
static const std::string PATTERN_MEAN0 = "FusedNodeMean0";
static const std::string PATTERN_SUB0 = "FusedNodeSub0";
static const std::string PATTERN_SUB1 = "FusedNodeSub1";
static const std::string PATTERN_MUL0 = "FusedNodeMul0";
static const std::string PATTERN_POW0 = "FusedNodePow0";
static const std::string PATTERN_MEAN1 = "FusedNodeMean1";
static const std::string PATTERN_ADD0 = "FusedNodeAdd0";
static const std::string PATTERN_RSQRT0 = "FusedNodeSqrt0";
static const std::string PATTERN_DIV0 = "FusedNodeDIV0";
static const std::string PATTERN_MUL1 = "FusedNodeMul1";
static const std::string PATTERN_ADD1 = "FusedNodeAdd1";
static const std::string LAYERNORM = "LayerNorm";
/*
case 1: default single op net with Mul                  |case 2: default single op net with Mul (without affine)
               x                                        |                     x
           /   |   \                                    |                 /   |   \
         /     |     \                                  |               /     |     \
       /       |       \                                |             /       |       \
      |    ReduceMean---|-------                        |            |    ReduceMean---|-------
       \     /          \       |                       |            \     /           \       |
         \  /            \     /                        |             \  /              \     /
         Sub              Sub                    x      |              Sub               Sub                    x
         /                 |                     |      |              /                  |                     |
        |                 Mul           ==>  LayerNorm  |              |                 Mul           ==>  LayerNorm
        |             ReduceMean                 |      |              |             ReduceMean                 |
        |                 Add                    y      |              |                 Add                    y
        |                Sqrt                           |              |                Sqrt
         \             /                                |              \             /
          \          /                                  |               \          /
            \      /                                    |                 \      /
              Div                                       |                   Div
              Mul                                       |                    y
              Add                                       |
               y                                        |

case 3: default single op net with Pow                  |case 4: default single op net with Pow (without affine)
               x                                        |                     x
           /   |   \                                    |                 /   |   \
         /     |     \                                  |               /     |     \
       /       |       \                                |             /       |       \
      |    ReduceMean---|-------                        |            |    ReduceMean---|-------
       \     /          \       |                       |            \     /           \       |
         \  /            \     /                        |             \  /              \     /
         Sub              Sub                    x      |              Sub               Sub                    x
         /                 |                     |      |              /                  |                     |
        |                 Pow           ==>  LayerNorm  |              |                 Pow           ==>  LayerNorm
        |             ReduceMean                 |      |              |             ReduceMean                 |
        |                 Add                    y      |              |                 Add                    y
        |                Sqrt                           |              |                Sqrt
         \             /                                |              \             /
          \          /                                  |               \          /
            \      /                                    |                 \      /
              Div                                       |                   Div
              Mul                                       |                    y
              Add                                       |
               y                                        |
*/

Status LayerNormONNXMULFusionPass::AddEdge(const ge::NodePtr& pre_node, int pre_idx, const ge::NodePtr& cur_node,
                                           int cur_idx) {
  auto pre_anchor = pre_node->GetOutDataAnchor(pre_idx);
  auto cur_anchor = cur_node->GetInDataAnchor(cur_idx);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(pre_anchor, cur_anchor) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to AddEdge between %s and %s.", pre_node->GetName().c_str(),
                            cur_node->GetName().c_str()),
                    return FAILED);

  return SUCCESS;
}

template <class T>
Status LayerNormONNXMULFusionPass::CreatNode(ge::ComputeGraph& graph, const ge::NodePtr& previous_node,
                                             ge::NodePtr& cur_node, std::string opname, std::string optype, T value,
                                             vector<ge::NodePtr>& fusionNodes) {
  // step1: create major node
  ge::OpDescPtr new_desc_ptr = nullptr;
  FUSION_PASS_MAKE_SHARED((new_desc_ptr = std::make_shared<ge::OpDesc>(opname, optype)),
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "create %s_desc_ptr failed.", opname.c_str());
                          new_desc_ptr = nullptr;
                          return INTERNAL_ERROR);

  ge::GeTensorDesc input_descs = previous_node->GetOpDesc()->GetInputDesc(0);
  new_desc_ptr->AddInputDesc(input_descs);
  new_desc_ptr->AddOutputDesc(input_descs);
  cur_node = graph.AddNode(new_desc_ptr);
  fusionNodes.push_back(cur_node);
  // step2: add const node
  // update tensor_desc
  std::vector<int64_t> ori_shape = input_descs.GetShape().GetDims();
  std::vector<int64_t> const_shape(axes.size(), 0);
  for (size_t i = 0; i < axes.size(); i++) {
    const_shape[i] = ori_shape[axes[i]];
  }
  input_descs.SetShape(GeShape(const_shape));
  input_descs.SetOriginShape(GeShape(const_shape));
  // create const array and set value
  int32_t const_numel = std::accumulate(const_shape.begin(), const_shape.end(), 1, std::multiplies<int32_t>());
  unique_ptr<T[]> const_array(new (std::nothrow) T[const_numel]());
  FUSION_PASS_CHECK(const_array.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "const_array is NULL"),
                    return PARAM_INVALID);
  Status ret = NnSet(const_numel, value, *reinterpret_cast<T*>(const_array.get()));
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);
  // create const node and set weights
  ge::GeTensorPtr const_desc_ptr = nullptr;
  FUSION_PASS_MAKE_SHARED((const_desc_ptr = std::make_shared<ge::GeTensor>(
                               input_descs, reinterpret_cast<uint8_t*>(const_array.get()), const_numel * sizeof(T))),
                          const_desc_ptr = nullptr;
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "const_desc_ptr failed.");
                          return PARAM_INVALID);
  ge::OpDescUtils::SetWeights(cur_node, {const_desc_ptr});
  auto const_nodes = OpDescUtils::GetConstInputs(cur_node);
  FUSION_PASS_CHECK(const_nodes.size() < 1, OP_LOGE(FUSED_OP_TYPE.c_str(), " const node size less than 1."),
                    return FAILED);
  NodePtr const_input = const_nodes[0];
  FUSION_PASS_CHECK(const_input == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "const input is null."),
                    return PARAM_INVALID);
  const_input->GetOpDesc()->SetType("Const");

  // update name
  std::map<string, uint32_t> input_name_idx = {{"x1", 0}, {"x2", 1}};
  new_desc_ptr->UpdateInputName(input_name_idx);
  return SUCCESS;
}

Status LayerNormONNXMULFusionPass::CreateMulAndAddNode(ge::ComputeGraph& graph, const ge::NodePtr div0_node,
                                                       ge::NodePtr& mul1_node, ge::NodePtr& add1_node,
                                                       vector<ge::NodePtr>& fusionNodes) {
  // step 1: create isolated mul and add op
  DataType dtype = div0_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
  if (dtype == ge::DT_FLOAT16) {
    uint16_t mul1_value = 1;
    Status ret =
        CreatNode<uint16_t>(graph, div0_node, mul1_node, div0_node->GetName() + "/Mul", "Mul", mul1_value, fusionNodes);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Creat Mul node fail"), return FAILED);

    uint16_t add1_value = 0;
    ret =
        CreatNode<uint16_t>(graph, mul1_node, add1_node, mul1_node->GetName() + "/Add", "Add", add1_value, fusionNodes);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Creat Add node fail"), return FAILED);
  } else if (dtype == ge::DT_FLOAT) {
    float mul1_value = 1.;
    Status ret =
        CreatNode<float>(graph, div0_node, mul1_node, div0_node->GetName() + "/Mul", "Mul", mul1_value, fusionNodes);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Creat Mul node fail"), return FAILED);

    float add1_value = 0.;
    ret = CreatNode<float>(graph, mul1_node, add1_node, mul1_node->GetName() + "/Add", "Add", add1_value, fusionNodes);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Creat Add node fail"), return FAILED);
  } else {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Node:%s's dtype is not in (float16, float32), fusion failed.");
    return NOT_CHANGED;
  }

  // step 2: handle NetOutput
  auto add1_anchor = add1_node->GetOutDataAnchor(0);
  for (auto in_data_anchor : div0_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(div0_node->GetOutDataAnchor(0), in_data_anchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to remove edge between div0 and successor node."),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(add1_anchor, in_data_anchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to add edge between div0 successor node and add1."),
                      return FAILED);
  }

  // step 3: add edge between two nodes
  FUSION_PASS_CHECK(SUCCESS != AddEdge(div0_node, 0, mul1_node, 0),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to AddEdge between %s and %s.", div0_node->GetName().c_str(),
                            mul1_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != AddEdge(mul1_node, 0, add1_node, 0),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to AddEdge between %s and %s.", div0_node->GetName().c_str(),
                            mul1_node->GetName().c_str()),
                    return FAILED);

  return SUCCESS;
}

vector<FusionPattern*> LayerNormONNXMULFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("LayerNormONNXMULFusionPass");
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern1->AddOpDesc(PATTERN_MEAN0, {MEAN})
      .AddOpDesc(PATTERN_SUB0, {SUB})
      .AddOpDesc(PATTERN_SUB1, {SUB})
      .AddOpDesc(PATTERN_MUL0, {MUL})
      .AddOpDesc(PATTERN_MEAN1, {MEAN})
      .AddOpDesc(PATTERN_ADD0, {ADD})
      .AddOpDesc(PATTERN_RSQRT0, {RSQRT})
      .AddOpDesc(PATTERN_DIV0, {DIV})
      .AddOpDesc(PATTERN_MUL1, {MUL})
      .AddOpDesc(PATTERN_ADD1, {ADD})
      .SetInputs(PATTERN_SUB0, {PATTERN_MEAN0})
      .SetInputs(PATTERN_SUB1, {PATTERN_MEAN0})
      .SetInputs(PATTERN_MUL0, {PATTERN_SUB1})
      .SetInputs(PATTERN_MEAN1, {PATTERN_MUL0})
      .SetInputs(PATTERN_ADD0, {PATTERN_MEAN1})
      .SetInputs(PATTERN_RSQRT0, {PATTERN_ADD0})
      .SetInputs(PATTERN_DIV0, {PATTERN_SUB0, PATTERN_RSQRT0})
      .SetInputs(PATTERN_MUL1, {PATTERN_DIV0})
      .SetInputs(PATTERN_ADD1, {PATTERN_MUL1})
      .SetOutput(PATTERN_ADD1);
  patterns.push_back(pattern1);

  FusionPattern* pattern2 = new (std::nothrow) FusionPattern("LayerNormONNXMULFusionPass");
  FUSION_PASS_CHECK(pattern2 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern2->AddOpDesc(PATTERN_MEAN0, {MEAN})
      .AddOpDesc(PATTERN_SUB0, {SUB})
      .AddOpDesc(PATTERN_SUB1, {SUB})
      .AddOpDesc(PATTERN_MUL0, {MUL})
      .AddOpDesc(PATTERN_MEAN1, {MEAN})
      .AddOpDesc(PATTERN_ADD0, {ADD})
      .AddOpDesc(PATTERN_RSQRT0, {RSQRT})
      .AddOpDesc(PATTERN_DIV0, {DIV})
      .SetInputs(PATTERN_SUB0, {PATTERN_MEAN0})
      .SetInputs(PATTERN_SUB1, {PATTERN_MEAN0})
      .SetInputs(PATTERN_MUL0, {PATTERN_SUB1})
      .SetInputs(PATTERN_MEAN1, {PATTERN_MUL0})
      .SetInputs(PATTERN_ADD0, {PATTERN_MEAN1})
      .SetInputs(PATTERN_RSQRT0, {PATTERN_ADD0})
      .SetInputs(PATTERN_DIV0, {PATTERN_SUB0, PATTERN_RSQRT0})
      .SetOutput(PATTERN_DIV0);
  patterns.push_back(pattern2);

  FusionPattern* pattern3 = new (std::nothrow) FusionPattern("LayerNormONNXMULFusionPass");
  FUSION_PASS_CHECK(pattern3 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern3->AddOpDesc(PATTERN_MEAN0, {MEAN})
      .AddOpDesc(PATTERN_SUB0, {SUB})
      .AddOpDesc(PATTERN_SUB1, {SUB})
      .AddOpDesc(PATTERN_POW0, {POW})
      .AddOpDesc(PATTERN_MEAN1, {MEAN})
      .AddOpDesc(PATTERN_ADD0, {ADD})
      .AddOpDesc(PATTERN_RSQRT0, {RSQRT})
      .AddOpDesc(PATTERN_DIV0, {DIV})
      .AddOpDesc(PATTERN_MUL1, {MUL})
      .AddOpDesc(PATTERN_ADD1, {ADD})
      .SetInputs(PATTERN_SUB0, {PATTERN_MEAN0})
      .SetInputs(PATTERN_SUB1, {PATTERN_MEAN0})
      .SetInputs(PATTERN_POW0, {PATTERN_SUB1})
      .SetInputs(PATTERN_MEAN1, {PATTERN_POW0})
      .SetInputs(PATTERN_ADD0, {PATTERN_MEAN1})
      .SetInputs(PATTERN_RSQRT0, {PATTERN_ADD0})
      .SetInputs(PATTERN_DIV0, {PATTERN_SUB0, PATTERN_RSQRT0})
      .SetInputs(PATTERN_MUL1, {PATTERN_DIV0})
      .SetInputs(PATTERN_ADD1, {PATTERN_MUL1})
      .SetOutput(PATTERN_ADD1);
  patterns.push_back(pattern3);

  FusionPattern* pattern4 = new (std::nothrow) FusionPattern("LayerNormONNXMULFusionPass");
  FUSION_PASS_CHECK(pattern4 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern4->AddOpDesc(PATTERN_MEAN0, {MEAN})
      .AddOpDesc(PATTERN_SUB0, {SUB})
      .AddOpDesc(PATTERN_SUB1, {SUB})
      .AddOpDesc(PATTERN_POW0, {POW})
      .AddOpDesc(PATTERN_MEAN1, {MEAN})
      .AddOpDesc(PATTERN_ADD0, {ADD})
      .AddOpDesc(PATTERN_RSQRT0, {RSQRT})
      .AddOpDesc(PATTERN_DIV0, {DIV})
      .SetInputs(PATTERN_SUB0, {PATTERN_MEAN0})
      .SetInputs(PATTERN_SUB1, {PATTERN_MEAN0})
      .SetInputs(PATTERN_POW0, {PATTERN_SUB1})
      .SetInputs(PATTERN_MEAN1, {PATTERN_POW0})
      .SetInputs(PATTERN_ADD0, {PATTERN_MEAN1})
      .SetInputs(PATTERN_RSQRT0, {PATTERN_ADD0})
      .SetInputs(PATTERN_DIV0, {PATTERN_SUB0, PATTERN_RSQRT0})
      .SetOutput(PATTERN_DIV0);
  patterns.push_back(pattern4);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define LayerNormONNXMULFusionPass pattern end");
  return patterns;
}

// get the scalar from the op inputs
/*
case 1: the first input of the op is a scalar |  case 2: the second input of the op is a scalar
    x1(scalar)      x2(tensor)                |      x1(tensor)      x2(scalar)
         \            /                       |           \            /
          \          /                        |            \          /
               op                             |                 op
               |                              |                 |
               |                              |                 |
*/
static Status GetTensorDescAndDtype(ge::NodePtr node, ge::Tensor& const_input,
                                    ge::GeTensorDesc& tensor_desc, DataType& dtype) {
  ge::Operator op_node = ge::OpDescUtils::CreateOperatorFromNode(node);
  dtype = op_node.GetInputDescByName("x2").GetDataType();
  std::string OpType = node->GetType();
  if (OpType == ADD) {
    if (GRAPH_SUCCESS != op_node.GetInputConstData("x2", const_input)) {
      OP_LOGI("LayerNorm", "get const data of %s failed for x2, need to check x1", node->GetName().c_str());
      if (GRAPH_SUCCESS != op_node.GetInputConstData("x1", const_input)) {
        OP_LOGD("LayerNorm", "get const data of %s failed for x1, not change", node->GetName().c_str());
        return NOT_CHANGED;
      } else {
        tensor_desc = node->GetOpDesc()->GetInputDesc("x1");
      }
    } else {
      tensor_desc = node->GetOpDesc()->GetInputDesc("x2");
    }
  } else if (OpType == POW) {
    if (GRAPH_SUCCESS != op_node.GetInputConstData("x2", const_input)) {
      OP_LOGI("LayerNorm", "get const data of %s failed for x2, need to check x1", node->GetName().c_str());
      return NOT_CHANGED;
    } else {
      tensor_desc = node->GetOpDesc()->GetInputDesc("x2");
    }
  } else {
    OP_LOGI("LayerNorm", "%s op cannot be fused.", node->GetName().c_str());
    return NOT_CHANGED;
  }
  return SUCCESS;
}

static Status GetScalarFromOp(ge::NodePtr node, float& value) {
  ge::Tensor const_input;
  ge::GeTensorDesc tensor_desc;
  DataType dtype;
  if (SUCCESS != GetTensorDescAndDtype(node, const_input, tensor_desc, dtype)) {
    return NOT_CHANGED;
  }
  std::vector<int64_t> tensor_dims = tensor_desc.GetShape().GetDims();
  if (tensor_dims.size() != 0) {
    if (PatternFusionUtil::IsUnknownShape(tensor_dims[0])) {
      OP_LOGD("LayerNorm", "LayerNormOnnxFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    if (!((tensor_dims.size() == 1) && (tensor_dims[0] == 1))) {
      OP_LOGD("LayerNorm", "the const input of %s must be scalar, not change", node->GetName().c_str());
      return NOT_CHANGED;
    }
  }
  if (!(dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16)) {
    OP_LOGD("LayerNorm", "add0_node type is not float or fp16, not change");
    return NOT_CHANGED;
  }
  if (const_input.GetData() != nullptr) {
    float* tensor_data_ptr = reinterpret_cast<float*>(const_input.GetData());
    value = *tensor_data_ptr;
  } else {
    OP_LOGE("LayerNorm", "const data of %s node is null.", node->GetName().c_str());
    return GRAPH_FAILED;
  }
  return SUCCESS;
}

Status LayerNormONNXMULFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI("LayerNormONNXMULFusionPass Fusion Start!");
  // get all nodes
  ge::NodePtr mean0_node = GetNodeFromMapping(PATTERN_MEAN0, mapping);
  ge::NodePtr sub0_node = GetNodeFromMapping(PATTERN_SUB0, mapping);
  ge::NodePtr sub1_node = GetNodeFromMapping(PATTERN_SUB1, mapping);
  ge::NodePtr mul0_node = GetNodeFromMapping(PATTERN_MUL0, mapping);
  ge::NodePtr pow0_node = GetNodeFromMapping(PATTERN_POW0, mapping);
  ge::NodePtr mean1_node = GetNodeFromMapping(PATTERN_MEAN1, mapping);
  ge::NodePtr add0_node = GetNodeFromMapping(PATTERN_ADD0, mapping);
  ge::NodePtr rsqrt0_node = GetNodeFromMapping(PATTERN_RSQRT0, mapping);
  ge::NodePtr div0_node = GetNodeFromMapping(PATTERN_DIV0, mapping);
  ge::NodePtr mul1_node = GetNodeFromMapping(PATTERN_MUL1, mapping);
  ge::NodePtr add1_node = GetNodeFromMapping(PATTERN_ADD1, mapping);

  // set the flag of pow0
  if ((mul0_node != nullptr) && (pow0_node == nullptr)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul0_node is in the graph.");
    with_pow = false;
  } else if ((mul0_node == nullptr) && (pow0_node != nullptr)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "pow0_node is in the graph.");
    with_pow = true;
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul0_node and pow0_node are invalid, fusion failed.");
    return NOT_CHANGED;
  }

  // set the flag of without affine
  if ((mul1_node == nullptr) && (add1_node == nullptr)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Both mul1_node and add1_node are null, it's belong to without affine scene.");
    with_affine = false;
  } else if ((mul1_node == nullptr) || (add1_node == nullptr)) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Either mul1_node or add1_node is null, fusion failed.");
    return NOT_CHANGED;
  }
  FUSION_PASS_CHECK(mean0_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mean0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(sub0_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "sub0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(sub1_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "sub1_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mean1_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mean1_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(add0_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "add0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(rsqrt0_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "rsqrt0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(div0_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "div0_node is null, fusion failed."),
                    return PARAM_INVALID);
  if (with_affine) {
    FUSION_PASS_CHECK(mul1_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mul1_node is null, fusion failed."),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(add1_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "add1_node is null, fusion failed."),
                      return PARAM_INVALID);
  }
  // check input link
  std::string mean0_input_name = mean0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  std::string sub0_input_name = sub0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  std::string sub1_input_name = sub1_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  if (strcmp(mean0_input_name.c_str(), sub0_input_name.c_str()) != 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mean0_input and sub0_input are not same, not change");
    return NOT_CHANGED;
  }
  if ((strcmp(mean0_input_name.c_str(), sub1_input_name.c_str()) != 0)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mean0_input and sub1_input are not same, not change");
    return NOT_CHANGED;
  }
  // check output link
  FUSION_PASS_CHECK(mean0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "mean0_node output size is [%d], which not equal to 2.",
                            mean0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(sub0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "sub0_node output size is [%d], which not equal to 1.",
                            sub0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  if (with_pow) {
    FUSION_PASS_CHECK(pow0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "pow0_node output size is [%d], which not equal to 1.",
                              pow0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                      return NOT_CHANGED);
    FUSION_PASS_CHECK(sub1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "sub1_node output size is [%d], which not equal to 1.",
                              sub1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                      return NOT_CHANGED);
  } else {
    FUSION_PASS_CHECK(mul0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "mul0_node output size is [%d], which not equal to 1.",
                              mul0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                      return NOT_CHANGED);
    FUSION_PASS_CHECK(sub1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 2,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "sub1_node output size is [%d], which not equal to 2.",
                              sub1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                      return NOT_CHANGED);
  }
  FUSION_PASS_CHECK(mean1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "mean1_node output size is [%d], which not equal to 1.",
                            mean1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(add0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add0_node output size is [%d], which not equal to 1.",
                            add0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(rsqrt0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "rsqrt0_node output size is [%d], which not equal to 1.",
                            rsqrt0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  if (with_affine) {
    FUSION_PASS_CHECK(div0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "div0_node output size is [%d], which not equal to 1.",
                              div0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                      return NOT_CHANGED);
    FUSION_PASS_CHECK(mul1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "mul1_node output size is [%d], which not equal to 1.",
                              mul1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                      return NOT_CHANGED);
  }
  // check input and attr
  ge::Operator op_mean0 = ge::OpDescUtils::CreateOperatorFromNode(mean0_node);
  std::vector<int64_t> axes0;
  if (GRAPH_SUCCESS != op_mean0.GetAttr("axes", axes0)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr axes failed.");
    return GRAPH_FAILED;
  }
  bool keep_dims0;
  if (GRAPH_SUCCESS != op_mean0.GetAttr("keep_dims", keep_dims0)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr keep_dims failed.");
    return GRAPH_FAILED;
  }
  Operator op_mean1 = ge::OpDescUtils::CreateOperatorFromNode(mean1_node);
  std::vector<int64_t> axes1;
  if (GRAPH_SUCCESS != op_mean1.GetAttr("axes", axes1)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr axes failed.");
    return GRAPH_FAILED;
  }
  bool keep_dims1;
  if (GRAPH_SUCCESS != op_mean0.GetAttr("keep_dims", keep_dims1)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr keep_dims failed.");
    return GRAPH_FAILED;
  }
  if ((axes0.size() != 1) || (axes1.size() != 1) || (axes0[0] != axes1[0])) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the axes of mean are not same, not change");
    return NOT_CHANGED;
  }
  if (!keep_dims0 || !keep_dims1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the attr keep_dims of mean is not true, not change");
    return NOT_CHANGED;
  }
  ge::GeTensorDesc input_desc = mean0_node->GetOpDesc()->GetInputDesc(0);
  std::vector<int64_t> input_dims = input_desc.GetShape().GetDims();
  size_t dims_size = input_dims.size();
  if (dims_size < 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "input shape must be greater to one, not change");
    return NOT_CHANGED;
  }
  if ((axes0[0] != -1) && (axes0[0] != (int64_t)(input_dims.size() - 1))) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the axes of mean is not the last dim of input, not change");
    return NOT_CHANGED;
  }

  // check const input
  if (with_pow) {
    float exp = 0.;
    FUSION_PASS_CHECK(SUCCESS != GetScalarFromOp(pow0_node, exp),
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "Fail to get value from const node of %s.", PATTERN_POW0.c_str()),
                      return NOT_CHANGED);
    // judge whether exp is equal to 2
    if (std::fabs(exp - 2.0) > std::numeric_limits<float>::epsilon()) {
      OP_LOGD("LayerNorm", "the exp of pow is %f, which should be equal to 2, not change", exp);
      return NOT_CHANGED;
    }
  }
  float add0_const = 0.;
  FUSION_PASS_CHECK(SUCCESS != GetScalarFromOp(add0_node, add0_const),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Fail to get value from const node of %s.", PATTERN_POW0.c_str()),
                    return NOT_CHANGED);
  if (!with_affine) {
    FUSION_PASS_CHECK(SUCCESS != CreateMulAndAddNode(graph, div0_node, mul1_node, add1_node, fusionNodes),
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "Fail to create mul1_node and add1_node, fusion failed."),
                      return NOT_CHANGED);
  }
  // copy Opdesc
  std::shared_ptr<ge::OpDesc> layer_desc = nullptr;
  layer_desc = std::make_shared<ge::OpDesc>(add1_node->GetName() + "/" + LAYERNORM, LAYERNORM);
  FUSION_PASS_CHECK(layer_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "layer_desc is null, fusion failed."),
                    return PARAM_INVALID);
  // add input
  FUSION_PASS_CHECK(layer_desc->AddInputDesc(0, input_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add input1_desc failed."), return FAILED);
  ge::GeTensorDesc input2_desc = mul1_node->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddInputDesc(1, input2_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add input2_desc failed."), return FAILED);
  ge::GeTensorDesc input3_desc = add1_node->GetOpDesc()->GetInputDesc(1);
  FUSION_PASS_CHECK(layer_desc->AddInputDesc(2, input3_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add input3_desc failed."), return FAILED);
  // add output
  ge::GeTensorDesc output1_desc = add1_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddOutputDesc("y", output1_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add output1_desc failed."), return FAILED);
  ge::GeTensorDesc output2_desc = mean0_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddOutputDesc("mean", output2_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add output2_desc failed."), return FAILED);
  ge::GeTensorDesc output3_desc = mean1_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddOutputDesc("variance", output3_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add output3_desc failed."), return FAILED);
  // add layer_norm node
  ge::NodePtr layer_node = graph.AddNode(layer_desc);
  fusionNodes.push_back(layer_node);

  // add attr
  Operator op_layer = ge::OpDescUtils::CreateOperatorFromNode(layer_node);
  op_layer.SetAttr("begin_norm_axis", axes0[0]);
  op_layer.SetAttr("begin_params_axis", -1);
  op_layer.SetAttr("epsilon", add0_const);

  // connect input edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mean0_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            layer_node->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            mean0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            layer_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mul1_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            layer_node->GetInDataAnchor(1)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            mul1_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            layer_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(add1_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            layer_node->GetInDataAnchor(2)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            add1_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            layer_node->GetName().c_str()),
                    return FAILED);

  // connect output edge
  for (auto& inDataAnchor : add1_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(add1_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(layer_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }

  // set node type
  layer_node->GetOpDesc()->SetType(LAYERNORM);

  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(mean0_node) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mean0_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(sub0_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove sub0_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(sub1_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove sub1_node failed."),
                    return FAILED);
  if (with_pow) {
    FUSION_PASS_CHECK(graph.RemoveNode(pow0_node) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove pow0_node failed."), return FAILED);
  } else {
    FUSION_PASS_CHECK(graph.RemoveNode(mul0_node) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mul0_node failed."), return FAILED);
  }
  FUSION_PASS_CHECK(graph.RemoveNode(mean1_node) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mean1_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(add0_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove add0_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(rsqrt0_node) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove rsqrt0_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(div0_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove div0_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul1_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mul1_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(add1_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove add1_node failed."),
                    return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "LayerNormONNXMULFusionPass graph fusion success!");
  return SUCCESS;
}
REGISTER_PASS("LayerNormONNXMULFusionPass", BUILT_IN_GRAPH_PASS, LayerNormONNXMULFusionPass);
}  // namespace fe
