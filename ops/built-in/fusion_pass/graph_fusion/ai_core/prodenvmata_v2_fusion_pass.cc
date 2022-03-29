/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "prodenvmata_v2_fusion_pass.h"

#include <math.h>
#include <iostream>
#include <map>
#include <algorithm>
#include "external/graph/operator_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"
#include "common/util/platform_info.h"

using namespace ge;
namespace fe {
static const std::string PATTERN_PRODENVMATA = "ProdEnvMatA";
static const char* PRODENVMATA = "ProdEnvMatA";
static const std::vector<std::string> SUPPORT_PLATFORM_PATTERN = {"Ascend910"};
float rcutaAttr = 0.0;
float rcutrAttr = 0.0;
float rcutrSmthAttr = 0.0;
vector<int32_t> selaAttr;
vector<int32_t> selrAttr;

vector<FusionPattern*> ProdEnvMatAV2FusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ProdEnvMatAV2Fusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_PRODENVMATA, {PRODENVMATA}).SetOutput(PATTERN_PRODENVMATA);
  patterns.push_back(pattern);
  return patterns;
}

Status ProdEnvMatAV2FusionPass::CheckPlatformInfo() {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckPlatformInfo begin");

  PlatformInfo platformInfo;
  OptionalInfo optionalInfo;
  FUSION_PASS_CHECK(
      PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != fe::SUCCESS,
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Failed to get platform info"), return NOT_CHANGED);

  std::string socVersion = optionalInfo.soc_version;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Get soc version: %s", socVersion.c_str());

  bool isSupport = false;
  for (string pattern : SUPPORT_PLATFORM_PATTERN) {
    if (socVersion == pattern || socVersion.find(pattern) != string::npos) {
      isSupport = true;
      break;
    }
  }
  FUSION_PASS_CHECK(!isSupport, OP_LOGD(FUSED_OP_TYPE.c_str(), "Only support 910 series platform"), return NOT_CHANGED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckPlatformInfo end");
  return SUCCESS;
}

Status ProdEnvMatAV2FusionPass::RemoveFusedNode(ge::ComputeGraph& graph, ge::NodePtr& fusedNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "RemovefusedNode begin");

  for (auto inAnchor : fusedNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  if (fusedNode->GetInControlAnchor() != nullptr) {
    fusedNode->GetInControlAnchor()->UnlinkAll();
  }
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove Node:%s", fusedNode->GetName().c_str()),
      return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "RemovefusedNode end");
  return SUCCESS;
}

Status ProdEnvMatAV2FusionPass::CreatAiCpuNode(ge::ComputeGraph& graph, ge::NodePtr& envmatNode,
                                               ge::NodePtr& envmatNodeAiCpu, vector<ge::GeTensorDesc>& inputDesc,
                                               vector<ge::GeTensorDesc>& outputDesc) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreatAiCpuNode begin");

  ge::OpDescPtr envMatACalcRijDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((envMatACalcRijDesc = std::make_shared<ge::OpDesc>(
                               envmatNode->GetName() + '_' + "ProdEnvMatACalcRij", "ProdEnvMatACalcRij")),
                          return INTERNAL_ERROR);

  FUSION_PASS_CHECK(
      envMatACalcRijDesc->AddInputDesc("coord", inputDesc[0]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add coord desc for envMatACalcRijDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcRijDesc->AddInputDesc("type", inputDesc[1]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add type desc for envMatACalcRijDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcRijDesc->AddInputDesc("natoms", inputDesc[2]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add natoms desc for envMatACalcRijDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcRijDesc->AddInputDesc("box", inputDesc[3]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add box desc for envMatACalcRijDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcRijDesc->AddInputDesc("mesh", inputDesc[4]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add mesh desc for envMatACalcRijDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcRijDesc->AddOutputDesc("rij", outputDesc[0]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add rij desc for envMatACalcRijDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcRijDesc->AddOutputDesc("nlist", outputDesc[1]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add nlist desc for envMatACalcRijDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcRijDesc->AddOutputDesc("distance", outputDesc[2]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add distance desc for envMatACalcRijDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcRijDesc->AddOutputDesc("rij_x", outputDesc[2]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add rij_x desc for envMatACalcRijDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcRijDesc->AddOutputDesc("rij_y", outputDesc[2]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add rij_y desc for envMatACalcRijDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcRijDesc->AddOutputDesc("rij_z", outputDesc[2]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add rij_z desc for envMatACalcRijDesc."),
      return FAILED);

  FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(envMatACalcRijDesc, "rcut_a", rcutaAttr),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to set envMatACalcRijDesc attr rcut_a[%f].", rcutaAttr),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(envMatACalcRijDesc, "rcut_r", rcutrAttr),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to set envMatACalcRijDesc attr rcut_r[%f].", rcutrAttr),
                    return FAILED);
  FUSION_PASS_CHECK(
      !ge::AttrUtils::SetFloat(envMatACalcRijDesc, "rcut_r_smth", rcutrSmthAttr),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to set envMatACalcRijDesc attr rcut_r_smth[%f].", rcutrSmthAttr),
      return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(envMatACalcRijDesc, "sel_a", selaAttr),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to set envMatACalcRijDesc attr sel_a"), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(envMatACalcRijDesc, "sel_r", selrAttr),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to set envMatACalcRijDesc attr sel_r"), return FAILED);

  envmatNodeAiCpu = graph.AddNode(envMatACalcRijDesc);
  FUSION_PASS_CHECK(envmatNodeAiCpu == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "envmatNodeAiCpu is null, fusion failed."),
                    return PARAM_INVALID);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreatAiCpuNode end");
  return SUCCESS;
}

Status ProdEnvMatAV2FusionPass::CreatAiCoreNode(ge::ComputeGraph& graph, ge::NodePtr& envmatNode,
                                                ge::NodePtr& envmatNodeAiCore, vector<ge::GeTensorDesc>& inputDesc,
                                                vector<ge::GeTensorDesc>& outputDesc) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreatAiCoreNode begin");

  ge::OpDescPtr envMatACalcDescrptDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((envMatACalcDescrptDesc = std::make_shared<ge::OpDesc>(
                               envmatNode->GetName() + '_' + "ProdEnvMatACalcDescrpt", "ProdEnvMatACalcDescrpt")),
                          return INTERNAL_ERROR);

  FUSION_PASS_CHECK(
      envMatACalcDescrptDesc->AddInputDesc("distance", inputDesc[0]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add distance desc for envMatACalcDescrptDesc."),
      return FAILED);

  FUSION_PASS_CHECK(
      envMatACalcDescrptDesc->AddInputDesc("rij_x", inputDesc[0]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add rij_x desc for envMatACalcDescrptDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcDescrptDesc->AddInputDesc("rij_y", inputDesc[0]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add rij_y desc for envMatACalcDescrptDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcDescrptDesc->AddInputDesc("rij_z", inputDesc[0]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add rij_z desc for envMatACalcDescrptDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcDescrptDesc->AddInputDesc("type", inputDesc[1]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add type desc for envMatACalcDescrptDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcDescrptDesc->AddInputDesc("natoms", inputDesc[2]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add natoms desc for envMatACalcDescrptDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcDescrptDesc->AddInputDesc("mesh", inputDesc[3]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add mesh desc for envMatACalcDescrptDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcDescrptDesc->AddInputDesc("davg", inputDesc[4]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add davg desc for envMatACalcDescrptDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcDescrptDesc->AddInputDesc("dstd", inputDesc[5]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add dstd desc for envMatACalcDescrptDesc."),
      return FAILED);
  FUSION_PASS_CHECK(
      envMatACalcDescrptDesc->AddOutputDesc("descrpt", outputDesc[0]) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add descrpt desc for envMatACalcDescrptDesc."),
      return FAILED);
  FUSION_PASS_CHECK(envMatACalcDescrptDesc->AddOutputDesc("descrpt_deriv", outputDesc[1]) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Failed to add descrpt_deriv desc for envMatACalcDescrptDesc."),
                    return FAILED);

  FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(envMatACalcDescrptDesc, "rcut_a", rcutaAttr),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to set envMatACalcDescrptDesc attr rcut_a[%f].", rcutaAttr),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(envMatACalcDescrptDesc, "rcut_r", rcutrAttr),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to set envMatACalcDescrptDesc attr rcut_r[%f].", rcutrAttr),
                    return FAILED);
  FUSION_PASS_CHECK(
      !ge::AttrUtils::SetFloat(envMatACalcDescrptDesc, "rcut_r_smth", rcutrSmthAttr),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to set envMatACalcDescrptDesc attr rcut_r_smth[%f].", rcutrSmthAttr),
      return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(envMatACalcDescrptDesc, "sel_a", selaAttr),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to set envMatACalcDescrptDesc attr sel_a."), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(envMatACalcDescrptDesc, "sel_r", selrAttr),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to set envMatACalcDescrptDesc attr sel_r."), return FAILED);

  envmatNodeAiCore = graph.AddNode(envMatACalcDescrptDesc);
  FUSION_PASS_CHECK(envmatNodeAiCore == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "envmatNodeAiCore is null, fusion failed."),
                    return PARAM_INVALID);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreatAiCoreNode end");
  return SUCCESS;
}

Status ProdEnvMatAV2FusionPass::AddAndDeleteEdge(ge::NodePtr& envmatNode, ge::NodePtr& envmatNodeAiCpu,
                                                 ge::NodePtr& envmatNodeAiCore) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "AddAndDeleteEdge begin");

  for (size_t i = 0; i < envmatNode->GetOpDesc()->GetAllInputsDesc().size() - 2; i++) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(envmatNode->GetInDataAnchor(i)->GetPeerOutAnchor(),
                                                         envmatNodeAiCpu->GetInDataAnchor(i)),
                      VECTOR_FUSION_INNER_ERR_REPORT(
                          FUSED_OP_TYPE.c_str(), "Failed to add edge from ProdEnvMatA node to envmatNodeAiCpu node."),
                      return FAILED);
  }

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(envmatNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                                       envmatNodeAiCore->GetInDataAnchor(4)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Failed to add edge from ProdEnvMatA node to envmatNodeAiCpu node."),
                    return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(envmatNode->GetInDataAnchor(2)->GetPeerOutAnchor(),
                                                       envmatNodeAiCore->GetInDataAnchor(5)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Failed to add edge from ProdEnvMatA node to envmatNodeAiCpu node."),
                    return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(envmatNode->GetInDataAnchor(4)->GetPeerOutAnchor(),
                                                       envmatNodeAiCore->GetInDataAnchor(6)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Failed to add edge from ProdEnvMatA node to envmatNodeAiCpu node."),
                    return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(envmatNode->GetInDataAnchor(5)->GetPeerOutAnchor(),
                                                       envmatNodeAiCore->GetInDataAnchor(7)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Failed to add edge from ProdEnvMatA node to envmatNodeAiCpu node."),
                    return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(envmatNode->GetInDataAnchor(6)->GetPeerOutAnchor(),
                                                       envmatNodeAiCore->GetInDataAnchor(8)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Failed to add edge from ProdEnvMatA node to envmatNodeAiCpu node."),
                    return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(envmatNodeAiCpu->GetOutDataAnchor(2), envmatNodeAiCore->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "Failed to add edge from output distance to input distance."),
      return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(envmatNodeAiCpu->GetOutDataAnchor(3), envmatNodeAiCore->GetInDataAnchor(1)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output rij_x to input rij_x."),
      return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(envmatNodeAiCpu->GetOutDataAnchor(4), envmatNodeAiCore->GetInDataAnchor(2)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output rij_y to input rij_y."),
      return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(envmatNodeAiCpu->GetOutDataAnchor(5), envmatNodeAiCore->GetInDataAnchor(3)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output rij_z to input rij_z."),
      return FAILED);

  for (auto inAnchor : envmatNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(envmatNodeAiCore->GetOutDataAnchor(0), inAnchor),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Failed to add edge from envmatNode to envmatNodeAiCore."),
                      return FAILED);
  }

  for (auto inAnchor : envmatNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(envmatNodeAiCore->GetOutDataAnchor(1), inAnchor),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Failed to add edge from envmatNode to envmatNodeAiCore."),
                      return FAILED);
  }

  for (auto inAnchor : envmatNode->GetOutDataAnchor(2)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(envmatNodeAiCpu->GetOutDataAnchor(0), inAnchor),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from envmatNode to envmatNodeAiCpu."),
        return FAILED);
  }

  for (auto inAnchor : envmatNode->GetOutDataAnchor(3)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(envmatNodeAiCpu->GetOutDataAnchor(1), inAnchor),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from envmatNode to envmatNodeAiCpu."),
        return FAILED);
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "AddAndDeleteEdge end");
  return SUCCESS;
}

Status ProdEnvMatAV2FusionPass::FusionAiScense(ge::ComputeGraph& graph, ge::NodePtr& envmatNode,
                                               vector<ge::NodePtr>& newEnvmatNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "FusionAiScense begin");

  ge::OpDescPtr envmatNodeDescPtr = envmatNode->GetOpDesc();
  FUSION_PASS_CHECK(envmatNodeDescPtr == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "envmatNode get desc failed."),
                    return NOT_CHANGED);

  size_t envmatNodeInputNum = 7;
  FUSION_PASS_CHECK(envmatNode->GetInDataNodes().size() != envmatNodeInputNum,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "the number of envmatNode input node is not equal to 7"),
                    return FAILED);
  size_t envmatNodeOutputNum = 4;
  FUSION_PASS_CHECK(envmatNode->GetAllOutDataAnchors().size() != envmatNodeOutputNum,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "the number of envmatNode output node is not equal to 4"),
                    return FAILED);

  ge::GeTensorDesc coordInputDesc = envmatNodeDescPtr->GetInputDesc(0);
  ge::GeTensorDesc typeInputDesc = envmatNodeDescPtr->GetInputDesc(1);
  ge::GeTensorDesc natomsInputDesc = envmatNodeDescPtr->GetInputDesc(2);
  ge::GeTensorDesc boxInputDesc = envmatNodeDescPtr->GetInputDesc(3);
  ge::GeTensorDesc meshInputDesc = envmatNodeDescPtr->GetInputDesc(4);
  ge::GeTensorDesc davgInputDesc = envmatNodeDescPtr->GetInputDesc(5);
  ge::GeTensorDesc dstdInputDesc = envmatNodeDescPtr->GetInputDesc(6);

  ge::GeTensorDesc descrptOutDesc = envmatNodeDescPtr->GetOutputDesc(0);
  ge::GeTensorDesc descrptDerivOutDesc = envmatNodeDescPtr->GetOutputDesc(1);
  ge::GeTensorDesc rijOutDesc = envmatNodeDescPtr->GetOutputDesc(2);
  ge::GeTensorDesc nlistOutDesc = envmatNodeDescPtr->GetOutputDesc(3);

  FUSION_PASS_CHECK(!ge::AttrUtils::GetFloat(envmatNodeDescPtr, "rcut_a", rcutaAttr),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Get rcut_a attr failed."), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::GetFloat(envmatNodeDescPtr, "rcut_r", rcutrAttr),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Get rcut_r attr failed."), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::GetFloat(envmatNodeDescPtr, "rcut_r_smth", rcutrSmthAttr),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Get rcut_r_smth attr failed."), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(envmatNodeDescPtr, "sel_a", selaAttr),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Get sel_a attr failed."), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(envmatNodeDescPtr, "sel_r", selrAttr),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Get sel_r attr failed."), return FAILED);

  int32_t nlocNnei = 0;
  int32_t nsample = 0;
  int32_t divisorNumberThree = 3;
  size_t rijOutDescShapeSize = 2;

  FUSION_PASS_CHECK(rijOutDesc.GetShape().GetDims().size() < rijOutDescShapeSize,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "the dim size of rijOutDesc is less than 2"), return FAILED);

  nsample = rijOutDesc.GetShape().GetDim(0);
  // nlocNnei is a multiple of 3 or -1
  nlocNnei = rijOutDesc.GetShape().GetDim(1);
  if (nlocNnei != -1) {
    nlocNnei = nlocNnei / divisorNumberThree;
  }

  ge::GeShape outputShape({nsample, nlocNnei});
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, ge::DT_FLOAT);
  outputTensorDesc.SetOriginShape(outputShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);

  ge::NodePtr envmatNodeAiCpu = nullptr;
  vector<ge::GeTensorDesc> aiCpuInputDesc = {coordInputDesc, typeInputDesc, natomsInputDesc, boxInputDesc,
                                             meshInputDesc};
  vector<ge::GeTensorDesc> aiCpuOutputDesc = {rijOutDesc, nlistOutDesc, outputTensorDesc};
  FUSION_PASS_CHECK(CreatAiCpuNode(graph, envmatNode, envmatNodeAiCpu, aiCpuInputDesc, aiCpuOutputDesc) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "CreatAiCpuNode failed"), return FAILED);

  ge::NodePtr envmatNodeAiCore = nullptr;
  vector<ge::GeTensorDesc> aiCoreInputDesc = {outputTensorDesc, typeInputDesc, natomsInputDesc,
                                              meshInputDesc, davgInputDesc, dstdInputDesc};
  vector<ge::GeTensorDesc> aiCoreOutputDesc = {descrptOutDesc, descrptDerivOutDesc};
  FUSION_PASS_CHECK(CreatAiCoreNode(graph, envmatNode, envmatNodeAiCore, aiCoreInputDesc, aiCoreOutputDesc) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "CreatAiCoreNode failed"), return FAILED);

  FUSION_PASS_CHECK(AddAndDeleteEdge(envmatNode, envmatNodeAiCpu, envmatNodeAiCore) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add and delete edge failed."),
                    return FAILED);

  newEnvmatNodes.push_back(envmatNodeAiCpu);
  newEnvmatNodes.push_back(envmatNodeAiCore);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "FusionAiScense end");
  return SUCCESS;
}

Status ProdEnvMatAV2FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into ProdEnvMatAV2FusionPass");

  FUSION_PASS_CHECK(CheckPlatformInfo() != SUCCESS, OP_LOGD(FUSED_OP_TYPE.c_str(), "Failed to check platform info"),
                    return NOT_CHANGED);

  ge::NodePtr envmatNode = GetNodeFromMapping(PATTERN_PRODENVMATA, mapping);
  FUSION_PASS_CHECK(envmatNode == nullptr, OP_LOGI("envmat Node is null."), return NOT_CHANGED);

  vector<ge::NodePtr> newEnvmatNodes;
  FUSION_PASS_CHECK(FusionAiScense(graph, envmatNode, newEnvmatNodes) != ge::GRAPH_SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Ai Scense fusion failed"), return FAILED);
  ge::NodePtr envmatNodeAiCpu = newEnvmatNodes[0];
  ge::NodePtr envmatNodeAiCore = newEnvmatNodes[1];

  FUSION_PASS_CHECK(RemoveFusedNode(graph, envmatNode) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove envmatNode!"),
                    return FAILED);

  newNodes.push_back(envmatNodeAiCpu);
  newNodes.push_back(envmatNodeAiCore);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to ProdEnvMatAV2FusionPass");
  return SUCCESS;
}
REGISTER_PASS("ProdEnvMatAV2FusionPass", BUILT_IN_GRAPH_PASS, ProdEnvMatAV2FusionPass);
}  // namespace fe