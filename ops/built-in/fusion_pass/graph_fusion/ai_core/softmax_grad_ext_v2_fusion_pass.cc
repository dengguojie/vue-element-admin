/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief BatchNormGrad BnInferGrad fusion pass
 *
 */

#include <memory>
#include <string>
#include "softmax_grad_ext_v2_fusion_pass.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {
static const string PATTERN_MUL = "mul";
static const string PATTERN_MUL1 = "mul1";
static const string PATTERN_MUL_Grad = "mulGrad";
static const string PATTERN_SUM = "sum";
static const string PATTERN_SUB = "sub";
static const string MUL = "Mul";
static const string SUM = "ReduceSumD";
static const string SUB = "Sub";
static const string SOFTMAXGRADEXT = "SoftmaxGradExt";
static const string AXIS = "axes";
static const string KEEPDIMS = "keep_dims";

vector<FusionPattern *> SoftmaxGradExtV2FusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxGradExtV2FusionPass pattern begin");
  vector<FusionPattern *> patterns;
  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("SoftmaxGradExtV2FusionPass");

  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

/*
                      input1 input0
                          \   /
                           mul               input0   input1  input2
                            |                   \        |      /
                    input0  sum        ----->     softmax_grad_ext
                       \    /
             input1      sub
                 \      /
       input2      mul1
          \         /
            mul_grad
*/
/*
                    input1 input0
                        \   /
                         mul               input0   input1  input2
                          |                   \        |      /
                 input0  sum        ----->     softmax_grad_ext
                     \    /
           input1     sub
               \      /
                 mul1      input2
                   \        /
                    mul_grad
*/

  pattern->AddOpDesc(PATTERN_MUL, {MUL})
      .AddOpDesc(PATTERN_MUL1, {MUL})
      .AddOpDesc(PATTERN_MUL_Grad, {MUL})
      .AddOpDesc(PATTERN_SUM, {SUM})
      .AddOpDesc(PATTERN_SUB, {SUB})
      .SetInputs(PATTERN_SUM, {PATTERN_MUL})
      .SetInputs(PATTERN_SUB, {PATTERN_SUM})
      .SetInputs(PATTERN_MUL1, {PATTERN_SUB})
      .SetInputs(PATTERN_MUL_Grad, {PATTERN_MUL1})
      .SetOutput(PATTERN_MUL_Grad);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxGradExtV2FusionPass pattern end");

  return patterns;
}

Status SoftmaxGradExtV2FusionPass::Fusion(
    ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes)
{
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxGradExtV2FusionPass fusion begin");
  ge::NodePtr mulNode = GetNodeFromMapping(PATTERN_MUL, mapping);
  ge::NodePtr mul1Node = GetNodeFromMapping(PATTERN_MUL1, mapping);
  ge::NodePtr mulGradNode = GetNodeFromMapping(PATTERN_MUL_Grad, mapping);
  ge::NodePtr subNode = GetNodeFromMapping(PATTERN_SUB, mapping);
  ge::NodePtr sumNode = GetNodeFromMapping(PATTERN_SUM, mapping);

  //check pattern
  if(subNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode() !=
     mulNode->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()) {
    return NOT_CHANGED;
  }

  FUSION_PASS_CHECK(mulNode == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "mulNode is null, fusion failed."),
           return PARAM_INVALID);
  FUSION_PASS_CHECK(mul1Node == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "mul1Node is null, fusion failed."),
           return PARAM_INVALID);
  FUSION_PASS_CHECK(mulGradNode == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "mulGradNode is null, fusion failed."),
           return PARAM_INVALID);
  FUSION_PASS_CHECK(subNode == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "subNode is null, fusion failed."),
           return PARAM_INVALID);
  FUSION_PASS_CHECK(sumNode == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "sumNode is null, fusion failed."),
           return PARAM_INVALID);

  // copy Opdesc
  std::shared_ptr<ge::OpDesc> newOpdesc = nullptr;
  newOpdesc =
      std::make_shared<ge::OpDesc>(mul1Node->GetName() + "/" + SOFTMAXGRADEXT,
                                   SOFTMAXGRADEXT);

  FUSION_PASS_CHECK(newOpdesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "newOpdesc is null, fusion failed."),
           return PARAM_INVALID);

  // add inputs
  ge::GeTensorDesc input_tensor0 = mulNode->GetOpDesc()->GetInputDesc(1);
  FUSION_PASS_CHECK(newOpdesc->AddInputDesc(input_tensor0) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add input failed."), return FAILED);
  ge::GeTensorDesc input_tensor1 = mulNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(newOpdesc->AddInputDesc(input_tensor1) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add input failed."), return FAILED);
  if (mulGradNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode() != mul1Node) {
    ge::GeTensorDesc input_tensor2 = mulGradNode->GetOpDesc()->GetInputDesc(0);
    FUSION_PASS_CHECK(newOpdesc->AddInputDesc(input_tensor2) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "add input failed."), return FAILED);
  } else {
    ge::GeTensorDesc input_tensor2 = mulGradNode->GetOpDesc()->GetInputDesc(1);
    FUSION_PASS_CHECK(newOpdesc->AddInputDesc(input_tensor2) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "add input failed."), return FAILED);
  }

  // add output
  ge::GeTensorDesc output_tensor = mulGradNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(newOpdesc->AddOutputDesc(output_tensor) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add output failed."), return FAILED);

  ge::NodePtr newNode = graph.AddNode(newOpdesc);
  newNodes.push_back(newNode);

  // copy attr
  vector<int32_t> axis;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(sumNode->GetOpDesc(), AXIS, axis),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Get attr axis failed"), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(newNode->GetOpDesc(), AXIS, axis),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Set attr axis failed"), return FAILED);
  bool keep_dims;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(sumNode->GetOpDesc(), KEEPDIMS, keep_dims),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Get attr keep_dims failed"), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(newNode->GetOpDesc(), KEEPDIMS, keep_dims),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Set attr keep_dims failed"), return FAILED);

  // connect output edge
  for (auto inDataAnchor :
       mulGradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(mulGradNode->GetOutDataAnchor(0),
                                        inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(newNode->GetOutDataAnchor(0),
                                     inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }

  if (mulGradNode->GetOutControlAnchor()) {
    for (auto inControlAnchor :
         mulGradNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::RemoveEdge(mulGradNode->GetOutControlAnchor(),
                                     inControlAnchor) != SUCCESS,
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out control edge failed."), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(newNode->GetOutControlAnchor(),
                                       inControlAnchor) != SUCCESS,
               OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out control edge failed."), return FAILED);
    }
  }

  // connect input edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
               mulNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
               newNode->GetInDataAnchor(0)) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                   mulNode->GetInDataAnchor(1)
                       ->GetPeerOutAnchor()
                       ->GetOwnerNode()
                       ->GetName()
                       .c_str(),
                   newNode->GetName().c_str()),
           return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
               mulNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
               newNode->GetInDataAnchor(1)) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                   mulNode->GetInDataAnchor(0)
                       ->GetPeerOutAnchor()
                       ->GetOwnerNode()
                       ->GetName()
                       .c_str(),
                   newNode->GetName().c_str()),
           return FAILED);

  if (mulGradNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode() != mul1Node) {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
        mulGradNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
        newNode->GetInDataAnchor(2)) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                     mulGradNode->GetInDataAnchor(0)
                         ->GetPeerOutAnchor()
                         ->GetOwnerNode()
                         ->GetName()
                         .c_str(),
                     newNode->GetName().c_str()),
    return FAILED);
  } else {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
        mulGradNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
        newNode->GetInDataAnchor(2)) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                     mulGradNode->GetInDataAnchor(1)
                         ->GetPeerOutAnchor()
                         ->GetOwnerNode()
                         ->GetName()
                         .c_str(),
                     newNode->GetName().c_str()),
    return FAILED);
  }

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mulGradNode->GetOutControlAnchor(),
                                   newNode->GetInControlAnchor()) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add control edge between node %s. and node %s failed.",
                   mulGradNode->GetName().c_str(),
                   newNode->GetName().c_str()),
           return FAILED);

  // set grad op type to BNInferGrad
  newNode->GetOpDesc()->SetType(SOFTMAXGRADEXT);

  FUSION_PASS_CHECK(graph.RemoveNode(mulNode) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mul node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul1Node) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mul1 node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mulGradNode) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mulGrad node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(subNode) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove sub node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(sumNode) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove sum node failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxGradExtV2FusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("SoftmaxGradExtFusionV2", BUILT_IN_GRAPH_PASS,
              SoftmaxGradExtV2FusionPass);
}  // namespace fe
