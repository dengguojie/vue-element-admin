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
 * \file gelu_fusion_pass.cpp
 * \brief gelu usion pass
 */
#include "gelu_onnx_fusion_pass.h"

#include <numeric>
#include <sstream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "fp16_t.hpp"

using namespace ge;
namespace fe {
static const char* DIV = "RealDiv";
static const char* ADD = "Add";
static const char* MUL = "Mul";
static const char* ERF = "Erf";
static const std::string PATTERN_INPUT = "Input0";
static const std::string PATTERN_DIV0 = "FusedNodeDiv0";
static const std::string PATTERN_ERF0 = "FusedNodeErf0";
static const std::string PATTERN_ADD0 = "FusedNodeAdd0";
static const std::string PATTERN_MUL0 = "FusedNodeMul0";
static const std::string PATTERN_MUL1 = "FusedNodeMul1";
static const std::string GELU = "Gelu";
/*
case1: mul0 has const input       | case2:mul1 has const input                           
        x                         |              x                  
     /     \                      |           /     \          
    |      RealDiv                |          |      RealDiv            
    |        |                    |          |        |                    
    |       Erf           x       |          |       Erf           x    
    |        |            |       |          |        |            |
   Mul0     Add   ==>    Gelu     |          |       Add   ==>    Gelu                
    \        /            |       |          \        /            |        
     \      /             y       |           \      /             y
      \    /                      |             Mul0                        
       Mul1                       |              |
        |                         |             Mul1
        y                         |              y
*/
vector<FusionPattern*> GeluONNXFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("GeluONNXFusionPass");
  FUSION_PASS_CHECK(pattern1 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern1->AddOpDesc(PATTERN_DIV0, {DIV})
      .AddOpDesc(PATTERN_ERF0, {ERF})
      .AddOpDesc(PATTERN_ADD0, {ADD})
      .AddOpDesc(PATTERN_MUL0, {MUL})
      .AddOpDesc(PATTERN_MUL1, {MUL})
      .SetInputs(PATTERN_ERF0, {PATTERN_DIV0})
      .SetInputs(PATTERN_ADD0, {PATTERN_ERF0})
      .SetInputs(PATTERN_MUL1, {PATTERN_MUL0, PATTERN_ADD0})
      .SetOutput(PATTERN_MUL1);
  patterns.push_back(pattern1);

  FusionPattern* pattern2 = new (std::nothrow) FusionPattern("GeluONNXFusionPass");
  FUSION_PASS_CHECK(pattern2 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern2->AddOpDesc(PATTERN_INPUT)
      .AddOpDesc(PATTERN_DIV0, {DIV})
      .AddOpDesc(PATTERN_ERF0, {ERF})
      .AddOpDesc(PATTERN_ADD0, {ADD})
      .AddOpDesc(PATTERN_MUL0, {MUL})
      .AddOpDesc(PATTERN_MUL1, {MUL})
      .SetInputs(PATTERN_DIV0, {PATTERN_INPUT})
      .SetInputs(PATTERN_ERF0, {PATTERN_DIV0})
      .SetInputs(PATTERN_ADD0, {PATTERN_ERF0})
      .SetInputs(PATTERN_MUL0, {PATTERN_INPUT, PATTERN_ADD0})
      .SetInputs(PATTERN_MUL1, {PATTERN_MUL0})
      .SetOutput(PATTERN_MUL1);

  patterns.push_back(pattern2);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define GeluONNXFusionPass pattern end");
  return patterns;
}

static Status GetScalarFromOp(ge::NodePtr node, float& value) {
  ge::Tensor const_input;
  ge::Operator op_node = ge::OpDescUtils::CreateOperatorFromNode(node);
  ge::GeTensorDesc tensor_desc = node->GetOpDesc()->GetInputDesc("x2");
  if (GRAPH_SUCCESS != op_node.GetInputConstData("x2", const_input)) {
    OP_LOGI("Gelu", "get const data of %s failed for x2, need to check x1", node->GetName().c_str());
    if (GRAPH_SUCCESS != op_node.GetInputConstData("x1", const_input)) {
      OP_LOGW("Gelu", "get const data of %s failed for x1, not change", node->GetName().c_str());
      return NOT_CHANGED;
    } else {
      tensor_desc = node->GetOpDesc()->GetInputDesc("x1");
    }
  }

  std::vector<int64_t> tensor_dims = tensor_desc.GetShape().GetDims();
  if (tensor_dims.size() != 0) {
    if (PatternFusionUtil::IsUnknownShape(tensor_dims[0])) {
      OP_LOGW("Gelu", "GeluONNXFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    if (!((tensor_dims.size() == 1) && (tensor_dims[0] == 1))) {
      OP_LOGW("Gelu", "the const input of %s must be scalar, not change", node->GetName().c_str());
      return NOT_CHANGED;
    }
  }
  DataType dtype = op_node.GetInputDescByName("x2").GetDataType();
  if (!(dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16)) {
    OP_LOGW("Gelu", "add0_node type is not float or fp16, not change");
    return NOT_CHANGED;
  }
  float* tensor_data_ptr = (float*)const_input.GetData();
  if (tensor_data_ptr == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT("Gelu", "const data of %s node is null.", node->GetName().c_str());
    return GRAPH_FAILED;
  }
  value = *tensor_data_ptr;
  return SUCCESS;
}

Status GeluONNXFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // get all nodes
  ge::NodePtr div0_node = GetNodeFromMapping(PATTERN_DIV0, mapping);
  ge::NodePtr erf0_node = GetNodeFromMapping(PATTERN_ERF0, mapping);
  ge::NodePtr add0_node = GetNodeFromMapping(PATTERN_ADD0, mapping);
  ge::NodePtr mul0_node = GetNodeFromMapping(PATTERN_MUL0, mapping);
  ge::NodePtr mul1_node = GetNodeFromMapping(PATTERN_MUL1, mapping);
  FUSION_PASS_CHECK(div0_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "div0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(erf0_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "erf0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(add0_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul0_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mul0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul1_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mul1_node is null, fusion failed."),
                    return PARAM_INVALID);

  // check input and output link relation
  FUSION_PASS_CHECK(div0_node->GetInDataNodes().size() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of Div0 node size is [%lu], which not equal to 2.",
                            div0_node->GetInDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(div0_node->GetOutDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node of Div0 node size is [%d], which not equal to 1.",
                            div0_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(erf0_node->GetInDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of Erf0 node size is [%lu], which not equal to 1.",
                            erf0_node->GetInDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(erf0_node->GetOutDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node of Erf0 node size is [%d], which not equal to 1.",
                            erf0_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(add0_node->GetInDataNodes().size() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of Add0 node size is [%lu], which not equal to 2.",
                            add0_node->GetInDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(add0_node->GetOutDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node of Add0 node size is [%d], which not equal to 1.",
                            add0_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(mul0_node->GetInDataNodes().size() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of Mul0 node size is [%lu], which not equal to 2.",
                            mul0_node->GetInDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(mul0_node->GetOutDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node of Mul0 node size is [%d], which not equal to 1.",
                            mul0_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);

  // check const input
  float div0_exp, add0_exp, mul0_exp, mul1_exp = 0.;
  float div0_const_input = 1.41421;
  FUSION_PASS_CHECK(SUCCESS != GetScalarFromOp(div0_node, div0_exp),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Fail to get value from const node of %s.", PATTERN_DIV0.c_str()),
                    return NOT_CHANGED);
  if (std::fabs(div0_exp - div0_const_input) > 0.00001) {
      OP_LOGW("Gelu", "the exp of div0 is %f, which should be equal to 1.41421, not change", div0_exp);
      return NOT_CHANGED;
  }
  FUSION_PASS_CHECK(SUCCESS != GetScalarFromOp(add0_node, add0_exp),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Fail to get value from const node of %s.", PATTERN_ADD0.c_str()),
                    return NOT_CHANGED);
  if (std::fabs(add0_exp - 1.0) > std::numeric_limits<float>::epsilon()) {
      OP_LOGW("Gelu", "the exp of add0 is %f, which should be equal to 1.0, not change", add0_exp);
      return NOT_CHANGED;
  }

  Status status_mul0 = GetScalarFromOp(mul0_node, mul0_exp);
  Status status_mul1 = GetScalarFromOp(mul1_node, mul1_exp);
  mulConstNodeType = status_mul0 == SUCCESS ? true : false;
  if (mulConstNodeType) {
    FUSION_PASS_CHECK(SUCCESS != status_mul0,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Fail to get value from const node of %s.", PATTERN_MUL0.c_str()),
                    return NOT_CHANGED);
    if (std::fabs(mul0_exp - 0.5) > std::numeric_limits<float>::epsilon()) {
      OP_LOGW("Gelu", "the exp of mul0 is %f, which should be equal to 0.5, not change", mul0_exp);
      return NOT_CHANGED;
    }
  } else {
    FUSION_PASS_CHECK(SUCCESS != status_mul1,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Fail to get value from const node of %s.", PATTERN_MUL1.c_str()),
                    return NOT_CHANGED);
    if (std::fabs(mul1_exp - 0.5) > std::numeric_limits<float>::epsilon()) {
      OP_LOGW("Gelu", "the exp of mul1 is %f, which should be equal to 0.5, not change", mul1_exp);
      return NOT_CHANGED;
    }
  }
  // copy Opdesc
  std::shared_ptr<ge::OpDesc> gelu_desc = nullptr;
  gelu_desc = std::make_shared<ge::OpDesc>(mul1_node->GetName() + "/" + GELU, GELU);
  FUSION_PASS_CHECK(gelu_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "gelu_desc is null, fusion failed."),
                    return PARAM_INVALID);

  // add input
  ge::GeTensorDesc input_desc = div0_node->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(gelu_desc->AddInputDesc(input_desc) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input failed."),
                    return FAILED);

  // add output
  ge::GeTensorDesc output_desc = mul1_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(gelu_desc->AddOutputDesc(output_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output failed."), return FAILED);

  // add gelu node
  ge::NodePtr gelu_node = graph.AddNode(gelu_desc);
  fusionNodes.push_back(gelu_node);

  // connect input edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(div0_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            gelu_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            div0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            gelu_node->GetName().c_str()),
                    return FAILED);

  // connect output edge
  for (auto &inDataAnchor : mul1_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(mul1_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(gelu_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }

  // set node type
  gelu_node->GetOpDesc()->SetType(GELU);
  std::map<string, uint32_t> input_name_id = {{"x", 0}};
  gelu_node->GetOpDesc()->UpdateInputName(input_name_id);
  std::map<string, uint32_t> output_name_id = {{"y", 0}};
  gelu_node->GetOpDesc()->UpdateOutputName(output_name_id);

  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(div0_node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove div0_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(erf0_node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove erf0_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(add0_node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove add0_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul0_node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul0_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul1_node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul1_node failed."),
                    return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "GeluONNXFusionPass graph fusion success!");
  return SUCCESS;
}
REGISTER_PASS("GeluONNXFusionPass", BUILT_IN_GRAPH_PASS, GeluONNXFusionPass);
}  // namespace fe
