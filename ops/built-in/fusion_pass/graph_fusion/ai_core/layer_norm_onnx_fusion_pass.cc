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
 * \file layer_norm_onnx_fusion_pass.cpp
 * \brief layer norm onnx fusion pass
 */

#include <numeric>
#include <sstream>
#include "layer_norm_onnx_fusion_pass.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "fp16_t.hpp"

namespace fe {
static const char* REDUCEMEAN = "ReduceMeanD";
static const char* SUB = "Sub";
static const char* CAST = "Cast";
static const char* POW = "Pow";
static const char* ADD = "Add";
static const char* SQRT = "Sqrt";
static const char* DIV = "RealDiv";
static const char* MUL = "Mul";
static const std::string PATTERN_INPUT = "Input0";
static const std::string PATTERN_REDUCEMEAN0 = "FusedNodeReduceMean0";
static const std::string PATTERN_SUB0 = "FusedNodeSub0";
static const std::string PATTERN_CAST0 = "FusedNodeCast0";
static const std::string PATTERN_POW0 = "FusedNodePow0";
static const std::string PATTERN_REDUCEMEAN1 = "FusedNodeReduceMean1";
static const std::string PATTERN_ADD0 = "FusedNodeAdd0";
static const std::string PATTERN_SQRT0 = "FusedNodeSqrt0";
static const std::string PATTERN_DIV0 = "FusedNodeDiv0";
static const std::string PATTERN_MUL0 = "FusedNodeMul0";
static const std::string PATTERN_ADD1 = "FusedNodeAdd1";
static const std::string LAYERNORM = "LayerNorm";
/*
case 1: default single op net                       |case 2: default single op net (without affine)
        x                                           |        x
     /     \                                        |     /     \
    |      ReduceMean                               |    |     ReduceMean
     \     /                                        |     \     /
        Sub                           x             |       Sub                           x
      /    \                          |             |      /    \                         |
     |     Pow               ==>  LayerNorm         |     |     Pow               ==>  LayerNorm
     |     ReduceMean                 |             |     |     ReduceMean                |
     |     Add                        y             |     |     Add                       y
     |     Sqrt                                     |     |     Sqrt
      \    /                                        |      \    /
        Div                                         |        Div
        Mul                                         |        y
        Add                                         |
         y                                          |
case 3: contain cast                                |case 4: contain cast (without affine)
        x                                           |        x
     /     \                                        |     /     \
    |      ReduceMean                               |    |     ReduceMean
     \     /                                        |     \     /
        Sub                           x             |       Sub                           x
      /    \                          |             |      /    \                         |
     |     Cast               ==>  LayerNorm        |     |     Cast               ==>  LayerNorm
     |     Pow                        |             |     |     Pow                       |
     |     ReduceMean                 y             |     |     ReduceMean                y
     |     Add                                      |     |     Add
     |     Sqrt                                     |     |     Sqrt
      \    /                                        |      \    /
        Div                                         |        Div
        Mul                                         |        y
        Add                                         |
         y                                          |
*/
vector<FusionPattern*> LayerNormONNXFusionPass::DefinePatterns() {
  std::vector<FusionPattern*> patterns;

  FusionPattern* case1 = new (std::nothrow) FusionPattern("LayerNormOnnxFusionPass");
  FUSION_PASS_CHECK(case1 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."), return patterns);
  case1->AddOpDesc(PATTERN_INPUT)
      .AddOpDesc(PATTERN_REDUCEMEAN0, {REDUCEMEAN})
      .AddOpDesc(PATTERN_SUB0, {SUB})
      .AddOpDesc(PATTERN_POW0, {POW})
      .AddOpDesc(PATTERN_REDUCEMEAN1, {REDUCEMEAN})
      .AddOpDesc(PATTERN_ADD0, {ADD})
      .AddOpDesc(PATTERN_SQRT0, {SQRT})
      .AddOpDesc(PATTERN_DIV0, {DIV})
      .AddOpDesc(PATTERN_MUL0, {MUL})
      .AddOpDesc(PATTERN_ADD1, {ADD})
      .SetInputs(PATTERN_REDUCEMEAN0, {PATTERN_INPUT})
      .SetInputs(PATTERN_SUB0, {PATTERN_INPUT, PATTERN_REDUCEMEAN0})
      .SetInputs(PATTERN_POW0, {PATTERN_SUB0})
      .SetInputs(PATTERN_REDUCEMEAN1, {PATTERN_POW0})
      .SetInputs(PATTERN_ADD0, {PATTERN_REDUCEMEAN1})
      .SetInputs(PATTERN_SQRT0, {PATTERN_ADD0})
      .SetInputs(PATTERN_DIV0, {PATTERN_SUB0, PATTERN_SQRT0})
      .SetInputs(PATTERN_MUL0, {PATTERN_DIV0})
      .SetInputs(PATTERN_ADD1, {PATTERN_MUL0})
      .SetOutput(PATTERN_ADD1);
  patterns.push_back(case1);
  FusionPattern* case2 = new (std::nothrow) FusionPattern("LayerNormOnnxFusionPass");
  FUSION_PASS_CHECK(case2 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."), return patterns);
  case2->AddOpDesc(PATTERN_INPUT)
      .AddOpDesc(PATTERN_REDUCEMEAN0, {REDUCEMEAN})
      .AddOpDesc(PATTERN_SUB0, {SUB})
      .AddOpDesc(PATTERN_POW0, {POW})
      .AddOpDesc(PATTERN_REDUCEMEAN1, {REDUCEMEAN})
      .AddOpDesc(PATTERN_ADD0, {ADD})
      .AddOpDesc(PATTERN_SQRT0, {SQRT})
      .AddOpDesc(PATTERN_DIV0, {DIV})
      .SetInputs(PATTERN_REDUCEMEAN0, {PATTERN_INPUT})
      .SetInputs(PATTERN_SUB0, {PATTERN_INPUT, PATTERN_REDUCEMEAN0})
      .SetInputs(PATTERN_POW0, {PATTERN_SUB0})
      .SetInputs(PATTERN_REDUCEMEAN1, {PATTERN_POW0})
      .SetInputs(PATTERN_ADD0, {PATTERN_REDUCEMEAN1})
      .SetInputs(PATTERN_SQRT0, {PATTERN_ADD0})
      .SetInputs(PATTERN_DIV0, {PATTERN_SUB0, PATTERN_SQRT0})
      .SetOutput(PATTERN_DIV0);
  patterns.push_back(case2);
  FusionPattern* case3 = new (std::nothrow) FusionPattern("LayerNormOnnxFusionPass");
  FUSION_PASS_CHECK(case3 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."), return patterns);
  case3->AddOpDesc(PATTERN_INPUT)
      .AddOpDesc(PATTERN_REDUCEMEAN0, {REDUCEMEAN})
      .AddOpDesc(PATTERN_SUB0, {SUB})
      .AddOpDesc(PATTERN_CAST0, {CAST})
      .AddOpDesc(PATTERN_POW0, {POW})
      .AddOpDesc(PATTERN_REDUCEMEAN1, {REDUCEMEAN})
      .AddOpDesc(PATTERN_ADD0, {ADD})
      .AddOpDesc(PATTERN_SQRT0, {SQRT})
      .AddOpDesc(PATTERN_DIV0, {DIV})
      .AddOpDesc(PATTERN_MUL0, {MUL})
      .AddOpDesc(PATTERN_ADD1, {ADD})
      .SetInputs(PATTERN_REDUCEMEAN0, {PATTERN_INPUT})
      .SetInputs(PATTERN_SUB0, {PATTERN_INPUT, PATTERN_REDUCEMEAN0})
      .SetInputs(PATTERN_CAST0, {PATTERN_SUB0})
      .SetInputs(PATTERN_POW0, {PATTERN_CAST0})
      .SetInputs(PATTERN_REDUCEMEAN1, {PATTERN_POW0})
      .SetInputs(PATTERN_ADD0, {PATTERN_REDUCEMEAN1})
      .SetInputs(PATTERN_SQRT0, {PATTERN_ADD0})
      .SetInputs(PATTERN_DIV0, {PATTERN_SUB0, PATTERN_SQRT0})
      .SetInputs(PATTERN_MUL0, {PATTERN_DIV0})
      .SetInputs(PATTERN_ADD1, {PATTERN_MUL0})
      .SetOutput(PATTERN_ADD1);
  patterns.push_back(case3);
  FusionPattern* case4 = new (std::nothrow) FusionPattern("LayerNormOnnxFusionPass");
  FUSION_PASS_CHECK(case4 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."), return patterns);
  case4->AddOpDesc(PATTERN_INPUT)
      .AddOpDesc(PATTERN_REDUCEMEAN0, {REDUCEMEAN})
      .AddOpDesc(PATTERN_SUB0, {SUB})
      .AddOpDesc(PATTERN_CAST0, {CAST})
      .AddOpDesc(PATTERN_POW0, {POW})
      .AddOpDesc(PATTERN_REDUCEMEAN1, {REDUCEMEAN})
      .AddOpDesc(PATTERN_ADD0, {ADD})
      .AddOpDesc(PATTERN_SQRT0, {SQRT})
      .AddOpDesc(PATTERN_DIV0, {DIV})
      .SetInputs(PATTERN_REDUCEMEAN0, {PATTERN_INPUT})
      .SetInputs(PATTERN_SUB0, {PATTERN_INPUT, PATTERN_REDUCEMEAN0})
      .SetInputs(PATTERN_CAST0, {PATTERN_SUB0})
      .SetInputs(PATTERN_POW0, {PATTERN_CAST0})
      .SetInputs(PATTERN_REDUCEMEAN1, {PATTERN_POW0})
      .SetInputs(PATTERN_ADD0, {PATTERN_REDUCEMEAN1})
      .SetInputs(PATTERN_SQRT0, {PATTERN_ADD0})
      .SetInputs(PATTERN_DIV0, {PATTERN_SUB0, PATTERN_SQRT0})
      .SetOutput(PATTERN_DIV0);
  patterns.push_back(case4);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define LayerNormOnnxFusionPass pattern end");
  return patterns;
}

Status LayerNormONNXFusionPass::AddEdge(const ge::NodePtr& pre_node, int pre_idx, const ge::NodePtr& cur_node,
                                        int cur_idx) {
  auto pre_anchor = pre_node->GetOutDataAnchor(pre_idx);
  auto cur_anchor = cur_node->GetInDataAnchor(cur_idx);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(pre_anchor, cur_anchor) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to AddEdge between %s and %s.", pre_node->GetName().c_str(),
                            cur_node->GetName().c_str()),
                    return FAILED);

  return SUCCESS;
}

template<class T>
Status LayerNormONNXFusionPass::CreatNode(ge::ComputeGraph& graph, const ge::NodePtr& previous_node,
                                          ge::NodePtr& cur_node, std::string opname, std::string optype, T value,
                                          vector<ge::NodePtr>& fusionNodes) {
  // step1: create major node
  ge::OpDescPtr new_desc_ptr = nullptr;
  FUSION_PASS_MAKE_SHARED((new_desc_ptr = std::make_shared<ge::OpDesc>(opname, optype)),
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "create %s_desc_ptr failed.", opname.c_str());
                          new_desc_ptr = nullptr; return INTERNAL_ERROR);

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
  FUSION_PASS_MAKE_SHARED(
      (const_desc_ptr = std::make_shared<ge::GeTensor>(input_descs, reinterpret_cast<uint8_t*>(const_array.get()),
                                                       const_numel * sizeof(T))),
      const_desc_ptr = nullptr;
      OP_LOGE(FUSED_OP_TYPE.c_str(), "const_desc_ptr failed."); return PARAM_INVALID);
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

Status LayerNormONNXFusionPass::CreateMulAndAddNode(ge::ComputeGraph& graph, const ge::NodePtr div0_node,
                                                    ge::NodePtr& mul0_node, ge::NodePtr& add1_node,
                                                    vector<ge::NodePtr>& fusionNodes) {
  // step 1: create isolated mul and add op
  DataType dtype = div0_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
  if (dtype == ge::DT_FLOAT16){
    uint16_t mul0_value = 1;
    Status ret =
        CreatNode<uint16_t>(graph, div0_node, mul0_node, div0_node->GetName() + "/Mul", "Mul", mul0_value, fusionNodes);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Creat Mul node fail"), return FAILED);

    uint16_t add1_value = 0;
    ret =
        CreatNode<uint16_t>(graph, mul0_node, add1_node, mul0_node->GetName() + "/Add", "Add", add1_value, fusionNodes);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Creat Mul node fail"), return FAILED);
  } else if (dtype == ge::DT_FLOAT) {
    float mul0_value = 1.;
    Status ret =
        CreatNode<float>(graph, div0_node, mul0_node, div0_node->GetName() + "/Mul", "Mul", mul0_value, fusionNodes);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Creat Mul node fail"), return FAILED);

    float add1_value = 0.;
    ret = CreatNode<float>(graph, mul0_node, add1_node, mul0_node->GetName() + "/Add", "Add", add1_value, fusionNodes);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Creat Mul node fail"), return FAILED);
  } else {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Node:%s's dtype is not in (float16, float32), fusion failed.");
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
  FUSION_PASS_CHECK(SUCCESS != AddEdge(div0_node, 0, mul0_node, 0),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to AddEdge between %s and %s.", div0_node->GetName().c_str(),
                            mul0_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != AddEdge(mul0_node, 0, add1_node, 0),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to AddEdge between %s and %s.", div0_node->GetName().c_str(),
                            mul0_node->GetName().c_str()),
                    return FAILED);

  return SUCCESS;
}

Status LayerNormONNXFusionPass::CheckEdges(std::map<std::string, ge::NodePtr>& nodes_map) {
  // check input edge
  std::string reducemean0_input_name =
      nodes_map[PATTERN_REDUCEMEAN0]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  std::string sub0_input0_name =
      nodes_map[PATTERN_SUB0]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  FUSION_PASS_CHECK(reducemean0_input_name != sub0_input0_name,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "reducemean0_input and sub0_input are different, not change."),
                    return NOT_CHANGED);
  std::string sub0_output_name = "";
  if (nodes_map[PATTERN_CAST0] != nullptr) {
    sub0_output_name = nodes_map[PATTERN_CAST0]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  } else {
    sub0_output_name = nodes_map[PATTERN_POW0]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  }
  std::string div0_input0_name =
      nodes_map[PATTERN_DIV0]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  FUSION_PASS_CHECK(sub0_output_name != div0_input0_name,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "pow0_input0 and div0_input0 are different, not change."),
                    return NOT_CHANGED);

  // check output edge
  std::string out_node = with_affine ? PATTERN_ADD1 : PATTERN_DIV0;
  for (auto& it : nodes_map) {
    if ((it.second != nullptr) && (it.first != PATTERN_SUB0) && (it.first != out_node)) {
      FUSION_PASS_CHECK(it.second->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                        OP_LOGW(FUSED_OP_TYPE.c_str(), "The output edge of %s is [%d], which not equal to 1.",
                                it.first.c_str(), it.second->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                        return NOT_CHANGED);
    } else if (it.first == PATTERN_SUB0) {
      FUSION_PASS_CHECK(it.second->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 2,
                        OP_LOGW(FUSED_OP_TYPE.c_str(), "The output edge of %s is [%d], which not equal to 2.",
                                it.first.c_str(), it.second->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                        return NOT_CHANGED);
    }
  }

  return SUCCESS;
}

static Status GetScalarFromOp(ge::NodePtr node, float& value) {
  ge::Tensor const_input;
  ge::Operator op_node = ge::OpDescUtils::CreateOperatorFromNode(node);
  ge::GeTensorDesc tensor_desc = node->GetOpDesc()->GetInputDesc("x2");
  if (GRAPH_SUCCESS != op_node.GetInputConstData("x2", const_input)) {
    OP_LOGI("LayerNorm", "get const data of %s failed for x2, need to check x1", node->GetName().c_str());
    if (GRAPH_SUCCESS != op_node.GetInputConstData("x1", const_input)) {
      OP_LOGW("LayerNorm", "get const data of %s failed for x1, not change", node->GetName().c_str());
      return NOT_CHANGED;
    } else {
      tensor_desc = node->GetOpDesc()->GetInputDesc("x1");
    }
  }

  std::vector<int64_t> tensor_dims = tensor_desc.GetShape().GetDims();
  if (tensor_dims.size() != 0) {
    if (PatternFusionUtil::IsUnknownShape(tensor_dims[0])) {
      OP_LOGW("LayerNorm", "LayerNormOnnxFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    if (!((tensor_dims.size() == 1) && (tensor_dims[0] == 1))) {
      OP_LOGW("LayerNorm", "the const input of %s must be scalar, not change", node->GetName().c_str());
      return NOT_CHANGED;
    }
  }
  DataType dtype = op_node.GetInputDescByName("x2").GetDataType();
  if (!(dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16)) {
    OP_LOGW("LayerNorm", "add0_node type is not float or fp16, not change");
    return NOT_CHANGED;
  }
  float* tensor_data_ptr = (float*)const_input.GetData();
  if (tensor_data_ptr == nullptr) {
    OP_LOGE("LayerNorm", "const data of %s node is null.", node->GetName().c_str());
    return GRAPH_FAILED;
  }
  value = *tensor_data_ptr;
  return SUCCESS;
}

static Status CheckShape(ge::NodePtr node, std::vector<int64_t> axes, std::vector<int64_t> expect) {
  ge::Tensor const_input;
  ge::Operator op_node = ge::OpDescUtils::CreateOperatorFromNode(node);
  ge::GeTensorDesc tensor_desc = node->GetOpDesc()->GetInputDesc("x2");
  if (op_node.GetInputConstData("x2", const_input) != GRAPH_SUCCESS) {
    OP_LOGI("LayerNorm", "get const data of %s failed for x2, need to check x1", node->GetName().c_str());
    if (op_node.GetInputConstData("x1", const_input) != GRAPH_SUCCESS) {
      OP_LOGI("LayerNorm", "get const data of %s failed for x1, not change", node->GetName().c_str());
      return NOT_CHANGED;
    } else {
      tensor_desc = node->GetOpDesc()->GetInputDesc("x1");
    }
  }
  std::vector<int64_t> tensor_dims = tensor_desc.GetShape().GetDims();

  FUSION_PASS_CHECK(tensor_dims.size() != axes.size(),
                    OP_LOGW("LayerNorm", "The dim of const %s is not equal to the length of reducemean0 attr",
                            node->GetName().c_str()),
                    return NOT_CHANGED);
  for (size_t i = 0; i < axes.size(); i++) {
    if (tensor_dims[i] != expect[axes[i]]) {
      OP_LOGE("LayerNorm", "The value of %s is not equal to the length of reducemean0 attr, not change",
              node->GetName().c_str());
      return NOT_CHANGED;
    }
  }

  return SUCCESS;
}

Status LayerNormONNXFusionPass::CheckValue(std::map<std::string, ge::NodePtr>& nodes_map) {
  // check value of attr
  ge::Operator op_reducemean0 = ge::OpDescUtils::CreateOperatorFromNode(nodes_map[PATTERN_REDUCEMEAN0]);
  bool keep_dims = false;
  if (GRAPH_SUCCESS != op_reducemean0.GetAttr("keep_dims", keep_dims)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to get keep_dims from %s.", nodes_map[PATTERN_REDUCEMEAN0]->GetName().c_str());
    return GRAPH_FAILED;
  }
  if (!keep_dims) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "the attr keep_dims in reducemean must be true, but it is false, not change");
    return NOT_CHANGED;
  }

  // check const input
  float exp = 0.;
  FUSION_PASS_CHECK(SUCCESS != GetScalarFromOp(nodes_map[PATTERN_POW0], exp),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Fail to get value from const node of %s.", PATTERN_POW0.c_str()),
                    return NOT_CHANGED);
  if (std::fabs(exp - 2.0) > std::numeric_limits<float>::epsilon()) {
    OP_LOGW("LayerNorm", "the exp of pow is %f, which should be equal to 2, not change", exp);
    return NOT_CHANGED;
  }

  FUSION_PASS_CHECK(SUCCESS != GetScalarFromOp(nodes_map[PATTERN_ADD0], epsilon),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Fail to get epsilon from const node of %s.", PATTERN_ADD0.c_str()),
                    return NOT_CHANGED);
  if (epsilon > 0.1) {
    OP_LOGW("LayerNorm", "the epsilon of Add0 is %f, which should be close to 0, not change", epsilon);
    return NOT_CHANGED;
  }
  ge::GeTensorDesc input_desc = nodes_map[PATTERN_REDUCEMEAN0]->GetOpDesc()->GetInputDesc(0);
  std::vector<int64_t> expect = input_desc.GetShape().GetDims();
  if (nodes_map[PATTERN_MUL0] != nullptr) {
    FUSION_PASS_CHECK(SUCCESS != CheckShape(nodes_map[PATTERN_MUL0], axes, expect),
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "The shape of %s is not as expect.", PATTERN_MUL0.c_str()),
                      return NOT_CHANGED);
  }
  if (nodes_map[PATTERN_ADD1] != nullptr) {
    FUSION_PASS_CHECK(SUCCESS != CheckShape(nodes_map[PATTERN_ADD1], axes, expect),
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "The shape of %s is not as expect.", PATTERN_ADD1.c_str()),
                      return NOT_CHANGED);
  }

  return SUCCESS;
}

Status LayerNormONNXFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "start running in LayerNormONNX fusion pass");

  // step1: get all the nodes from mapping.
  std::map<std::string, ge::NodePtr> nodes_map = {
      {PATTERN_REDUCEMEAN0, nullptr}, {PATTERN_SUB0, nullptr}, {PATTERN_CAST0, nullptr}, {PATTERN_POW0, nullptr},
      {PATTERN_REDUCEMEAN1, nullptr}, {PATTERN_ADD0, nullptr}, {PATTERN_SQRT0, nullptr}, {PATTERN_DIV0, nullptr},
      {PATTERN_MUL0, nullptr},        {PATTERN_ADD1, nullptr}};
  std::vector<std::string> exclude = {PATTERN_CAST0, PATTERN_MUL0, PATTERN_ADD1};
  for (auto& it : nodes_map) {
    ge::NodePtr node = GetNodeFromMapping(it.first, mapping);
    if (std::find(exclude.begin(), exclude.end(), it.first) == exclude.end()) {
      FUSION_PASS_CHECK(node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "%s is null, fusion failed.", it.first.c_str()),
                        return PARAM_INVALID);
    } else if (node == nullptr) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "%s is null.", it.first.c_str());
    }
    it.second = node;
  }

  // step2: get axes from reducemean0, which is used by CreateMulAndAddNode and CheckValue
  ge::Operator op_reducemean0 = ge::OpDescUtils::CreateOperatorFromNode(nodes_map[PATTERN_REDUCEMEAN0]);
  if (op_reducemean0.GetAttr("axes", axes) != GRAPH_SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to get axes from %s.", nodes_map[PATTERN_REDUCEMEAN0]->GetName().c_str());
    return GRAPH_FAILED;
  }
  size_t dims_size = nodes_map[PATTERN_REDUCEMEAN0]->GetOpDesc()->GetInputDesc(0).GetShape().GetDims().size();
  if (dims_size < 1) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "input shape must be greater than one, not change");
    return NOT_CHANGED;
  }
  for (size_t i = 0; i < axes.size(); i++) {
    axes[i] = axes[i] > 0 ? axes[i] : axes[i] + dims_size;
  }
  begin_norm_axis = axes[0];

  // set the flag of without affine
  if ((nodes_map[PATTERN_MUL0] == nullptr) && (nodes_map[PATTERN_ADD1] == nullptr)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Both mul0_node and add1_node are null, it's belong to without affine scene.");
    with_affine = false;
  } else if ((nodes_map[PATTERN_MUL0] == nullptr) || (nodes_map[PATTERN_ADD1] == nullptr)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Either mul0_node or add1_node is null, fusion failed.");
    return NOT_CHANGED;
  }

  // step3: check the connection relationship and value of attr or const node
  FUSION_PASS_CHECK(CheckEdges(nodes_map) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(),
                            "The internal connection relationship is not as expected, fusion failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckValue(nodes_map) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(),
                            "The shape of const input or the value of attribute is not as expected, fusion failed."),
                    return NOT_CHANGED);

  // step4: creat mul and add op, convert the without affine scene into the with affine scene.
  if (!with_affine) {
    FUSION_PASS_CHECK(SUCCESS != CreateMulAndAddNode(graph, nodes_map[PATTERN_DIV0], nodes_map[PATTERN_MUL0],
                                                      nodes_map[PATTERN_ADD1], fusionNodes),
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "Fail to create mul0_node and add1_node, fusion failed."),
                      return NOT_CHANGED);
  }

  // step5: create(copy) output Opdesc
  ge::OpDescPtr layer_desc =
      std::make_shared<ge::OpDesc>(nodes_map[PATTERN_ADD1]->GetName() + "/" + LAYERNORM, LAYERNORM);
  FUSION_PASS_CHECK(layer_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "layer_desc is null, fusion failed."),
                    return FAILED);

  // step6: add input or output
  // add input desc
  ge::GeTensorDesc x_desc = nodes_map[PATTERN_REDUCEMEAN0]->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddInputDesc(0, x_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add x_desc failed."), return FAILED);
  ge::GeTensorDesc gamma_desc = nodes_map[PATTERN_MUL0]->GetOpDesc()->GetInputDesc(1);
  FUSION_PASS_CHECK(layer_desc->AddInputDesc(1, gamma_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add gamma_desc failed."), return FAILED);
  ge::GeTensorDesc beta_desc = nodes_map[PATTERN_ADD1]->GetOpDesc()->GetInputDesc(1);
  FUSION_PASS_CHECK(layer_desc->AddInputDesc(2, beta_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add beta_desc failed."), return FAILED);
  // add output desc
  ge::GeTensorDesc y_desc = nodes_map[PATTERN_ADD1]->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddOutputDesc("y", y_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add output1_desc failed."), return FAILED);
  ge::GeTensorDesc mean_desc = nodes_map[PATTERN_REDUCEMEAN0]->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddOutputDesc("mean", mean_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add mean_desc failed."), return FAILED);
  ge::GeTensorDesc var_desc = nodes_map[PATTERN_REDUCEMEAN1]->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddOutputDesc("variance", var_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add var_desc failed."), return FAILED);

  // step7: add layer_norm node and set value for attr
  ge::NodePtr layer_node = graph.AddNode(layer_desc);
  fusionNodes.push_back(layer_node);
  Operator op_layer = ge::OpDescUtils::CreateOperatorFromNode(layer_node);
  op_layer.SetAttr("begin_norm_axis", begin_norm_axis);  // reducemean0的axes属性
  op_layer.SetAttr("begin_params_axis", -1);
  op_layer.SetAttr("epsilon", epsilon);  // add0的const输入值

  // step8: add edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(nodes_map[PATTERN_REDUCEMEAN0]->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            layer_node->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            PATTERN_REDUCEMEAN0.c_str(), layer_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(nodes_map[PATTERN_MUL0]->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            layer_node->GetInDataAnchor(1)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            PATTERN_MUL0.c_str(), layer_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(nodes_map[PATTERN_ADD1]->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            layer_node->GetInDataAnchor(2)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            PATTERN_ADD1.c_str(), layer_node->GetName().c_str()),
                    return FAILED);
  for (auto& inDataAnchor : nodes_map[PATTERN_ADD1]->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(nodes_map[PATTERN_ADD1]->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(layer_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }

  // step9: set node type
  layer_node->GetOpDesc()->SetType(LAYERNORM);

  // step10: delete fused nodes
  for (auto& it : nodes_map) {
    if (it.second != nullptr) {
      FUSION_PASS_CHECK(graph.RemoveNode(it.second) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove %s failed.", it.first.c_str()), return FAILED);
    }
    it.second = nullptr;
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "LayerNormFusionPass graph fusion success!");
  return SUCCESS;
}

REGISTER_PASS("LayerNormONNXFusionPass", BUILT_IN_GRAPH_PASS, LayerNormONNXFusionPass);
}  // namespace fe