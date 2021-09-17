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
 * \file softmax_with_drop_out_do_mask_fusion_pass.cpp
 * \brief SoftmaxWithDropOutDoMask fusion pass
 */
#include "softmax_with_drop_out_do_mask_fusion_pass.h"

#include <memory>
#include <string>
#include "op_log.h"
#include "error_util.h"
#include "common/util/platform_info.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const uint32_t CORE_NUM = 32;
static const string AXIS = "axes";
static const string PATTERN_DROPOUT = "DropOutDoMaskV3D";
static const string PATTERN_SOFTMAX = "SoftmaxV2";
static const string KEEPPROB = "keep_prob";
static const string SOFTMAXWITHDROPOUTDOMASK = "SoftmaxV2WithDropOutDoMaskV3D";

vector<FusionPattern*> SoftmaxWithDropOutDoMaskFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxWithDropOutDoMaskFusionPass pattern begin.");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SoftmaxWithDropOutDoMaskFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_SOFTMAX, {"SoftmaxV2"})
          .AddOpDesc(PATTERN_DROPOUT, {"DropOutDoMaskV3D"})
          .SetInputs(PATTERN_DROPOUT, {PATTERN_SOFTMAX})
          .SetOutput(PATTERN_DROPOUT);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxWithDropOutDoMaskFusionPass pattern end.");

  return patterns;
}

Status SoftmaxWithDropOutDoMaskFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "SoftmaxWithDropOutDoMaskFusionPass fusion begin.");
  ge::NodePtr softmaxNode = GetNodeFromMapping(PATTERN_SOFTMAX, mapping);
  ge::NodePtr dropoutNode = GetNodeFromMapping(PATTERN_DROPOUT, mapping);

  FUSION_PASS_CHECK(softmaxNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "softmaxNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(dropoutNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dropoutNode is null, fusion failed."),
                    return PARAM_INVALID);

  if (softmaxNode->GetOpDesc()->GetInputDesc(0).GetDataType() != ge::DT_FLOAT16) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "type is not fp16.");
    return NOT_CHANGED;
  }

  PlatformInfo platform_info;
  OptionalInfo optional_info;
  FUSION_PASS_CHECK(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info,
                                                                                     optional_info) != fe::SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get platformInfo failed."), return false);
  uint32_t core_num = platform_info.soc_info.ai_core_cnt;
  if (core_num != CORE_NUM) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "platform is not support.");
    return NOT_CHANGED;
  }

  // copy Opdesc
  std::shared_ptr<ge::OpDesc> newOpdesc = nullptr;
  newOpdesc = std::make_shared<ge::OpDesc>(dropoutNode->GetName() + "/" + SOFTMAXWITHDROPOUTDOMASK, SOFTMAXWITHDROPOUTDOMASK);

  FUSION_PASS_CHECK(newOpdesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "newOpdesc is null, fusion failed."),
                    return PARAM_INVALID);

  // add inputs
  string newOpName = newOpdesc->GetName();
  ge::GeTensorDesc input_tensor0 = softmaxNode->GetOpDesc()->GetInputDesc(0);
  ge::GeShape softmaxInputShape = input_tensor0.GetShape();

  vector<int64_t> dimInfo = softmaxInputShape.GetDims();
  vector<int64_t> assitDimInfoOrigin = {dimInfo[0], dimInfo[1], 512, 512};
  vector<int64_t> assitDimInfo = {dimInfo[0], dimInfo[1], 32, 32, 16, 16};
  ge::GeShape assitShape(assitDimInfo);
  ge::GeShape assitShapeOrigin(assitDimInfoOrigin);

  input_tensor0.SetShape(assitShape);
  input_tensor0.SetOriginShape(assitShapeOrigin);
  input_tensor0.SetFormat(ge::FORMAT_FRACTAL_NZ);
  input_tensor0.SetOriginFormat(ge::FORMAT_ND);
  input_tensor0.SetDataType(ge::DT_FLOAT16);
  FUSION_PASS_CHECK(
      newOpdesc->AddInputDesc(input_tensor0) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add the input desc for the input x failed.", newOpName.c_str()),
      return FAILED);

  ge::GeTensorDesc input_tensor1 = dropoutNode->GetOpDesc()->GetInputDesc(1);
  input_tensor1.SetShape(assitShape);
  input_tensor1.SetOriginShape(assitShape);
  input_tensor1.SetFormat(ge::FORMAT_ND);
  input_tensor1.SetOriginFormat(ge::FORMAT_ND);
  input_tensor1.SetDataType(ge::DT_UINT8);

  FUSION_PASS_CHECK(
      newOpdesc->AddInputDesc(input_tensor1) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add the input desc for the input mask failed.", newOpName.c_str()),
      return FAILED);

  // add output
  ge::GeTensorDesc output_tensor0 = softmaxNode->GetOpDesc()->GetOutputDesc(0);
  output_tensor0.SetShape(assitShape);
  output_tensor0.SetOriginShape(assitShapeOrigin);
  output_tensor0.SetFormat(ge::FORMAT_FRACTAL_NZ);
  output_tensor0.SetOriginFormat(ge::FORMAT_ND);
  output_tensor0.SetDataType(ge::DT_FLOAT16);
  FUSION_PASS_CHECK(
      newOpdesc->AddOutputDesc(output_tensor0) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add the output desc for the output y1 failed.", newOpName.c_str()),
      return FAILED);

  ge::GeTensorDesc output_tensor1 = dropoutNode->GetOpDesc()->GetOutputDesc(0);
  output_tensor1.SetShape(assitShape);
  output_tensor1.SetOriginShape(assitShapeOrigin);
  output_tensor1.SetFormat(ge::FORMAT_FRACTAL_NZ);
  output_tensor1.SetOriginFormat(ge::FORMAT_ND);
  output_tensor1.SetDataType(ge::DT_FLOAT16);
  FUSION_PASS_CHECK(
      newOpdesc->AddOutputDesc(output_tensor1) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[%s]: add the output desc for the output y2 failed.", newOpName.c_str()),
      return FAILED);

  ge::NodePtr newNode = graph.AddNode(newOpdesc);
  newNodes.push_back(newNode);

  // copy attr
  vector<int32_t> axis;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(softmaxNode->GetOpDesc(), AXIS, axis),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Get attr axis failed"), return NOT_CHANGED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(newNode->GetOpDesc(), AXIS, axis),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set attr axis failed"), return FAILED);
  float keep_prob;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetFloat(dropoutNode->GetOpDesc(), KEEPPROB, keep_prob),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Get attr keep_dims failed"), return NOT_CHANGED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(newNode->GetOpDesc(), KEEPPROB, keep_prob),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set attr keep_dims failed"), return FAILED);

  // connect output edge
  for (auto &inDataAnchor : softmaxNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    if (inDataAnchor->GetOwnerNode()->GetType() == "DropOutDoMaskV3D") {
      continue;
    }
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(softmaxNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge1 failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(newNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge1 failed."), return FAILED);
  }

  for (auto &inDataAnchor : dropoutNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(dropoutNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge2 failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(newNode->GetOutDataAnchor(1), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge2 failed."), return FAILED);
  }

  if (dropoutNode->GetOutControlAnchor()) {
    for (auto &inControlAnchor : dropoutNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(dropoutNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out control edge failed."), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(newNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out control edge failed."), return FAILED);
    }
  }

  // connect input edge
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(softmaxNode->GetInDataAnchor(0)->GetPeerOutAnchor(), newNode->GetInDataAnchor(0)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
              softmaxNode->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
              newNode->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(dropoutNode->GetInDataAnchor(1)->GetPeerOutAnchor(), newNode->GetInDataAnchor(1)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
              dropoutNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
              newNode->GetName().c_str()),
      return FAILED);

  // set grad op type to
  newNode->GetOpDesc()->SetType(SOFTMAXWITHDROPOUTDOMASK);

  FUSION_PASS_CHECK(graph.RemoveNode(softmaxNode) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove softmax node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(dropoutNode) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove dropout node failed."),
                    return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxWithDropOutDoMaskFusionPass fusion end");

  return SUCCESS;
}

REGISTER_PASS("SoftmaxWithDropOutDoMaskFusion", BUILT_IN_GRAPH_PASS, SoftmaxWithDropOutDoMaskFusionPass);
}  // namespace fe