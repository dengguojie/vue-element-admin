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
 * \file batchnorm_bninfer_fusion_pass.cpp
 * \brief BatchNorm BnInfer fusion pass
 */
#include <memory>
#include <string>
#include "batchnorm_bninfer_fusion_pass.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

#include "op_log.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "pattern_fusion_util.h"

namespace fe {
static const string PATTERN_BATCHNORM = "batchNorm";
static const string PATTERN_INPUTS1 = "input1";
static const string PATTERN_INPUTS2 = "input2";
static const string PATTERN_INPUTS3 = "input3";
static const string PATTERN_INPUTS4 = "input4";
static const string PATTERN_INPUTS5 = "input5";
static const string BATCHNORM = "BatchNorm";
static const string BNINFER = "BNInference";
static const string EPSILON = "epsilon";
static const string IS_TRAING = "is_training";
static const string USE_GLOBAL_STATS = "use_global_stats";
static const string MODE = "mode";
static const uint32_t INPUT_IDX_2 = 2;
static const uint32_t INPUT_IDX_3 = 3;
static const uint32_t INPUT_IDX_4 = 4;
static const int64_t INPUT_IDX_5 = 5;
static const float FLOAT_NUM_ONE = 1;

vector<FusionPattern*> BatchNormBnInferFusionPass::DefinePatterns() {
  // batch_norm ------> bninference
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormBnInferFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("BatchNormBnInferFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_BATCHNORM, {BATCHNORM})
      .AddOpDesc(PATTERN_INPUTS1)
      .AddOpDesc(PATTERN_INPUTS2)
      .AddOpDesc(PATTERN_INPUTS3)
      .AddOpDesc(PATTERN_INPUTS4)
      .AddOpDesc(PATTERN_INPUTS5)
      .SetInputs(PATTERN_BATCHNORM,
                 {PATTERN_INPUTS1, PATTERN_INPUTS2, PATTERN_INPUTS3, PATTERN_INPUTS4, PATTERN_INPUTS5})
      .SetOutput(PATTERN_BATCHNORM);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormBnInferFusionPass pattern end");
  return patterns;
}

Status BatchNormBnInferFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormBnInferFusionPass fusion begin");
  ge::NodePtr batchNormNode = GetNodeFromMapping(PATTERN_BATCHNORM, mapping);

  FUSION_PASS_CHECK(batchNormNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "batchNorm is null, fusion failed."),
                    return PARAM_INVALID);
  bool is_traing = true;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(batchNormNode->GetOpDesc(), IS_TRAING, is_traing),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Get is_traing attr failed."), return FAILED);

  FUSION_PASS_CHECK(is_traing, OP_LOGI(FUSED_OP_TYPE.c_str(), "is_traing is true, no need fusion."), return NOT_CHANGED);

  FUSION_PASS_CHECK(batchNormNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() != 0 ||
                        batchNormNode->GetOutDataAnchor(2)->GetPeerInDataAnchors().size() != 0 ||
                        batchNormNode->GetOutDataAnchor(3)->GetPeerInDataAnchors().size() != 0 ||
                        batchNormNode->GetOutDataAnchor(4)->GetPeerInDataAnchors().size() != 0,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "batchNorm Node's some output's peer is null, no need fusion."),
                    return NOT_CHANGED);

  // validate const input
  bool is_not_const = false;
  ge::NodePtr inputNode;
  for (int i = 1; i < INPUT_IDX_5; i++) {
    inputNode = batchNormNode->GetInDataAnchor(i)->GetPeerOutAnchor()->GetOwnerNode();
    std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(inputNode);
    if (type != "Const" && type != "Constant") {
      is_not_const = true;
      break;
    }
  }

  // copy Opdesc
  std::shared_ptr<ge::OpDesc> newOpdesc = nullptr;
  newOpdesc = std::make_shared<ge::OpDesc>(batchNormNode->GetName(), BNINFER);

  FUSION_PASS_CHECK(newOpdesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "newOpdesc is null, fusion failed."),
                    return PARAM_INVALID);

  string newOpName = newOpdesc->GetName();

  // input 2-5 not all const, fused to bn_infer
  if (is_not_const == true) {
    string BNTYPE = "BNInfer";
    // add input
    ge::GeTensorDesc input_tensor_x = batchNormNode->GetOpDesc()->GetInputDesc(0);
    FUSION_PASS_CHECK(newOpdesc->AddInputDesc("x", input_tensor_x) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "BNInfer add input x failed."), return FAILED);

    ge::GeTensorDesc input_tensor_mean = batchNormNode->GetOpDesc()->GetInputDesc(1);
    FUSION_PASS_CHECK(newOpdesc->AddInputDesc("scale", input_tensor_mean) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "BNInfer add input scale failed."), return FAILED);

    ge::GeTensorDesc input_tensor_variance = batchNormNode->GetOpDesc()->GetInputDesc(INPUT_IDX_2);
    FUSION_PASS_CHECK(newOpdesc->AddInputDesc("offset", input_tensor_variance) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "BNInfer add input offset failed."), return FAILED);

    ge::GeTensorDesc input_tensor_scale = batchNormNode->GetOpDesc()->GetInputDesc(INPUT_IDX_3);
    FUSION_PASS_CHECK(newOpdesc->AddInputDesc("mean", input_tensor_scale) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "BNInfer add input mean failed."), return FAILED);

    ge::GeTensorDesc input_tensor_offset = batchNormNode->GetOpDesc()->GetInputDesc(INPUT_IDX_4);
    FUSION_PASS_CHECK(newOpdesc->AddInputDesc("variance", input_tensor_offset) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "BNInfer add input variance failed."), return FAILED);

    // add output
    ge::GeTensorDesc tensor1 = batchNormNode->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(newOpdesc->AddOutputDesc("y", tensor1) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "BNInfer add output failed."), return FAILED);

    ge::NodePtr newNode = graph.AddNode(newOpdesc);
    FUSION_PASS_CHECK(newNode == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "newNode is null,fusion failed"),
                      return PARAM_INVALID);
    fusionNodes.push_back(newNode);

    // copy attr
    float epsilon;
    FUSION_PASS_CHECK(!ge::AttrUtils::GetFloat(batchNormNode->GetOpDesc(), EPSILON, epsilon),
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "Get epsilon attr failed."), return NOT_CHANGED);

    FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(newNode->GetOpDesc(), EPSILON, epsilon),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Set epsilon attr failed"), return FAILED);

    // copy output edge
    for (auto inDataAnchor : batchNormNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(batchNormNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(newNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
    }

    if (batchNormNode->GetOutControlAnchor()) {
      for (auto inControlAnchor : batchNormNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
        FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(batchNormNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out control edge failed."), return FAILED);
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(newNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out control edge failed."), return FAILED);
      }
    }

    // copy BatchNorm inputs to BNInfer nodes
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                              newNode->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              batchNormNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                              newNode->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                              newNode->GetInDataAnchor(1)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              batchNormNode->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                              newNode->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(2)->GetPeerOutAnchor(),
                                              newNode->GetInDataAnchor(2)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              batchNormNode->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                              newNode->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(3)->GetPeerOutAnchor(),
                                              newNode->GetInDataAnchor(3)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              batchNormNode->GetInDataAnchor(3)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                              newNode->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(4)->GetPeerOutAnchor(),
                                              newNode->GetInDataAnchor(4)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              batchNormNode->GetInDataAnchor(4)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                              newNode->GetName().c_str()),
                      return FAILED);

    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(batchNormNode->GetOutControlAnchor(), newNode->GetInControlAnchor()) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add control edge between node %s. and node %s failed.",
                batchNormNode->GetName().c_str(), newNode->GetName().c_str()),
        return FAILED);

    // set grad op type to BNInferGrad
    newNode->GetOpDesc()->SetType(BNTYPE);

    FUSION_PASS_CHECK(graph.RemoveNode(batchNormNode) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove batchnorm node failed."), return FAILED);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormBnInferFusionPass fusion end");
    return SUCCESS;
  }

  // input 2-5 not all const, fused to bn_inference & add input
  ge::GeTensorDesc input_tensor_x = batchNormNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(
      newOpdesc->AddInputDesc("x", input_tensor_x) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: add the input desc for the input x failed.", newOpName.c_str()),
      return FAILED);

  ge::GeTensorDesc input_tensor_mean = batchNormNode->GetOpDesc()->GetInputDesc(INPUT_IDX_3);
  FUSION_PASS_CHECK(
      newOpdesc->AddInputDesc("mean", input_tensor_mean) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: add the input desc for the input mean failed.", newOpName.c_str()),
      return FAILED);

  ge::GeTensorDesc input_tensor_variance = batchNormNode->GetOpDesc()->GetInputDesc(INPUT_IDX_4);
  FUSION_PASS_CHECK(
      newOpdesc->AddInputDesc("variance", input_tensor_variance) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: add the input desc for the input variance failed.", newOpName.c_str()),
      return FAILED);

  // need to add momentum input
  // new the tensordesc for momentum
  vector<int64_t> assistShapeVec = {};
  ge::GeShape assistShape(assistShapeVec);
  ge::Format assistFormat = input_tensor_x.GetFormat();

  ge::GeTensorPtr momentumPtr = nullptr;
  unique_ptr<float[]> inputAssit(new (std::nothrow) float[1]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                    return PARAM_INVALID);
  Status ret = NnSet(1, FLOAT_NUM_ONE, *reinterpret_cast<float*>(inputAssit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

  ge::GeTensorDesc tensorDesc(ge::GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  tensorDesc.SetShape(assistShape);
  tensorDesc.SetFormat(assistFormat);

  FUSION_PASS_MAKE_SHARED((momentumPtr = std::make_shared<ge::GeTensor>(
                               tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), 1 * sizeof(float))),
                          momentumPtr = nullptr;
                          return PARAM_INVALID);

  // new the momentum node
  std::shared_ptr<ge::OpDesc> newConstantOp = nullptr;
  FUSION_PASS_MAKE_SHARED((newConstantOp = std::make_shared<ge::OpDesc>(newOpdesc->GetName() + "_momentum", "Constant")),
                          newConstantOp = nullptr;
                          return PARAM_INVALID);
  ge::AttrUtils::SetTensor(newConstantOp, "value", momentumPtr);
  (void)newConstantOp->AddOutputDesc(momentumPtr->GetTensorDesc());
  ge::NodePtr momentumNode = graph.AddNode(newConstantOp);

  (void)newOpdesc->AddInputDesc("momentum", newConstantOp->GetOutputDesc(0));

  // continue to add other inputs tensor_desc
  ge::GeTensorDesc input_tensor_scale = batchNormNode->GetOpDesc()->GetInputDesc(1);
  if (newOpdesc->AddInputDesc("scale", input_tensor_scale) != SUCCESS){
    momentumPtr = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: add the input desc for the input scale failed.", newOpName.c_str());
    return FAILED;
  }
  ge::GeTensorDesc input_tensor_offset = batchNormNode->GetOpDesc()->GetInputDesc(INPUT_IDX_2);
  if (newOpdesc->AddInputDesc("offset", input_tensor_offset) != SUCCESS){
    momentumPtr = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: add the input desc for the input offset failed.", newOpName.c_str());
    return FAILED;
  }
  // add output
  ge::GeTensorDesc tensor1 = batchNormNode->GetOpDesc()->GetOutputDesc(0);
  if (newOpdesc->AddOutputDesc("y", tensor1) != SUCCESS){
    momentumPtr = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Op[%s]: add the output desc for the output y failed.", newOpName.c_str());
    return FAILED;
  }
  ge::NodePtr newNode = graph.AddNode(newOpdesc);
  fusionNodes.push_back(newNode);

  // copy attr
  float epsilon;
  if (!ge::AttrUtils::GetFloat(batchNormNode->GetOpDesc(), EPSILON, epsilon)){
    momentumPtr = nullptr;
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Get epsilon attr failed.");
    return NOT_CHANGED;
  }
  if (!ge::AttrUtils::SetFloat(newNode->GetOpDesc(), EPSILON, epsilon)){
    momentumPtr = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Set epsilon attr failed");
    return FAILED;
  }
  if (!ge::AttrUtils::SetBool(newNode->GetOpDesc(), USE_GLOBAL_STATS, true)){
    momentumPtr = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Set use_global_stats attr failed");
    return FAILED;
  }
  if (!ge::AttrUtils::SetInt(newNode->GetOpDesc(), MODE, 1)){
    momentumPtr = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Set mode attr failed");
    return FAILED;
  }
  // add edge for momentum node and bninference node
  if (ge::GraphUtils::AddEdge(momentumNode->GetOutDataAnchor(0), newNode->GetInDataAnchor(3)) != SUCCESS){
    momentumPtr = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add input data edge failed.");
    return FAILED;
  }
  // copy output edge
  for (auto inDataAnchor : batchNormNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    if (ge::GraphUtils::RemoveEdge(batchNormNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS){
      momentumPtr = nullptr;
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed.");
      return FAILED;
    }
    if (ge::GraphUtils::AddEdge(newNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS){
      momentumPtr = nullptr;
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed.");
      return FAILED;
    }
  }

  if (batchNormNode->GetOutControlAnchor()) {
    for (auto inControlAnchor : batchNormNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      if (ge::GraphUtils::RemoveEdge(batchNormNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS){
        momentumPtr = nullptr;
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out control edge failed.");
        return FAILED;
      }
      if (ge::GraphUtils::AddEdge(newNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS){
        momentumPtr = nullptr;
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out control edge failed.");
        return FAILED;
      }
    }
  }

  // copy BatchNorm inputs to BNInfer nodes
  if (ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                              newNode->GetInDataAnchor(0)) != SUCCESS){
    momentumPtr = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
            batchNormNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
            newNode->GetName().c_str());
    return FAILED;
  }
  if (ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                              newNode->GetInDataAnchor(4)) != SUCCESS){
    momentumPtr = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
            batchNormNode->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
            newNode->GetName().c_str());
    return FAILED;
  }
  if (ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(2)->GetPeerOutAnchor(),
                              newNode->GetInDataAnchor(5)) != SUCCESS){
    momentumPtr = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
            batchNormNode->GetInDataAnchor(2)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
            newNode->GetName().c_str());
    return FAILED;
  }
  if (ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(3)->GetPeerOutAnchor(),
                              newNode->GetInDataAnchor(1)) != SUCCESS){
    momentumPtr = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
            batchNormNode->GetInDataAnchor(3)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
            newNode->GetName().c_str());
    return FAILED;
  }
  if (ge::GraphUtils::AddEdge(batchNormNode->GetInDataAnchor(4)->GetPeerOutAnchor(),
                              newNode->GetInDataAnchor(2)) != SUCCESS){
    momentumPtr = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
            batchNormNode->GetInDataAnchor(4)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
            newNode->GetName().c_str());
    return FAILED;
  }
  if (ge::GraphUtils::AddEdge(batchNormNode->GetOutControlAnchor(), newNode->GetInControlAnchor()) != SUCCESS){
    momentumPtr = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add control edge between node %s. and node %s failed.",
            batchNormNode->GetName().c_str(), newNode->GetName().c_str());
    return FAILED;
  }
  // set grad op type to BNInferGrad
  newNode->GetOpDesc()->SetType(BNINFER);

  if (graph.RemoveNode(batchNormNode) != SUCCESS){
    momentumPtr = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove identity node failed.");
    return FAILED;
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormBnInferFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("BatchNormBnInferFusionPass", BUILT_IN_GRAPH_PASS, BatchNormBnInferFusionPass);
}  // namespace fe
