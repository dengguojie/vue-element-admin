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

/*!
 * \file randomdsa_fusion_pass.cpp
 * \brief Dsa op fusion pass
 *  DropOutGenMask       RandomUniformInt       TruncatedNormal              RandomStandardNormal
 *        |                     |                      |                              |
 *        |                     |                      |                              |
 *        V                     V                      V                              V
 *  DsaGenBitMask        DSARandomUniform       DSARandomTruncatedNormal       DSARandomNormal
 */
#include "randomdsa_fusion_pass.h"

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
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "common/util/platform_info.h"

using namespace ge;
namespace fe {
// define op DropOutGenMask------>DSAGenBitMask
static const std::string PATTERN_FUSEDNODE_DROP = "DropOutGenMask";
static const std::string PATTERN_FUSEDNODE_DSAGEN = "DSAGenBitMask";

// define op RandomUniformInt------>DSARandomUniform
static const std::string PATTERN_FUSEDNODE_UNIFORM = "RandomUniformInt";
static const std::string PATTERN_FUSEDNODE_DSAUNIFORM = "DSARandomUniform";

// define op reduceprod
static const string kPatternDSA = "RandomDSA";
static const string kTypeReduceProd = "ReduceProd";
static const int32_t kCountIndex = 0;
static const int32_t MEAN_NODE_EDGE = 2;
static const int32_t STDDEV_NODE_EDGE = 3;
static const int32_t UNIFORM_INPUT_SIZE = 3;
static const int32_t MOVE_32_NUM = 32;

// define pattern
vector<FusionPattern*> RandomDsaFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("RandomDsaFusionPass1");
  FUSION_PASS_CHECK(pattern1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern1->AddOpDesc(PATTERN_FUSEDNODE_DROP, {"DropOutGenMask"}).SetOutput(PATTERN_FUSEDNODE_DROP);

  FusionPattern* pattern2 = new (std::nothrow) FusionPattern("RandomDsaFusionPass2");
  FUSION_PASS_CHECK(pattern2 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern2->AddOpDesc(PATTERN_FUSEDNODE_UNIFORM, {"RandomUniformInt"}).SetOutput(PATTERN_FUSEDNODE_UNIFORM);

  patterns.push_back(pattern1);
  patterns.push_back(pattern2);
  return patterns;
}

Status RandomDsaFusionPass::FlatternShape(ge::ComputeGraph& graph, ge::NodePtr& fusion_node, int32_t index) {
  ge::OpDescPtr reduceProd = nullptr;
  FUSION_PASS_MAKE_SHARED(
      reduceProd = std::make_shared<ge::OpDesc>(fusion_node->GetName() + "_reduceProd", kTypeReduceProd),
      return PARAM_INVALID);
  (void)ge::AttrUtils::SetBool(reduceProd, "keep_dims", false);
  if (fusion_node->GetAllInDataAnchors().empty() || fusion_node->GetInDataAnchor(index) == nullptr ||
      fusion_node->GetInDataAnchor(index)->GetPeerOutAnchor() == nullptr ||
      fusion_node->GetInDataAnchor(index)->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
    OP_LOGD(FUSED_OP_TYPE_PROD.c_str(), "RandomInsertReduceProdFusionPass fusion pass exit.");
    return NOT_CHANGED;
  }
  // set axes
  ge::GeTensorDesc tmp_desc;
  ge::GeTensorPtr seed_ptr = nullptr;
  std::unique_ptr<int32_t> seed_data_tmp(new (std::nothrow) int32_t(0));
  FUSION_PASS_MAKE_SHARED((seed_ptr = std::make_shared<ge::GeTensor>(
                               tmp_desc, reinterpret_cast<uint8_t*>(seed_data_tmp.get()), sizeof(int32_t))),
                          seed_ptr = nullptr;
                          return PARAM_INVALID);
  Status ret = seed_ptr->SetData(reinterpret_cast<uint8_t*>(seed_data_tmp.get()), sizeof(int32_t));
  if (ret != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE_PROD.c_str(), "set axes data failed.");
    return ret;
  }
  seed_ptr->MutableTensorDesc().SetDataType(ge::DT_INT32);
  ge::OpDescPtr const_opdesc = ge::OpDescUtils::CreateConstOp(seed_ptr);
  FUSION_PASS_CHECK(const_opdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a const object failed."), return FAILED);
  string constOpName = reduceProd->GetName() + "_axes";
  const_opdesc->SetName(constOpName);
  ge::NodePtr const_node = graph.AddNode(const_opdesc);

  ge::NodePtr countNode = fusion_node->GetInDataAnchor(index)->GetPeerOutAnchor()->GetOwnerNode();
  int outIndex = fusion_node->GetInDataAnchor(index)->GetPeerOutAnchor()->GetIdx();
  reduceProd->AddInputDesc(countNode->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(outIndex)));
  auto reduceProdOutDesc = fusion_node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(index));
  reduceProdOutDesc.SetShape(GeShape({1}));
  reduceProdOutDesc.SetOriginShape(GeShape({1}));
  reduceProd->AddOutputDesc(reduceProdOutDesc);
  fusion_node->GetOpDesc()->UpdateInputDesc(static_cast<uint32_t>(index), reduceProdOutDesc);
  ge::NodePtr reduceProdNode = graph.AddNode(reduceProd);

  (void)ge::GraphUtils::RemoveEdge(countNode->GetOutDataAnchor(outIndex), fusion_node->GetInDataAnchor(index));
  (void)ge::GraphUtils::AddEdge(countNode->GetOutDataAnchor(outIndex), reduceProdNode->GetInDataAnchor(0));
  (void)ge::GraphUtils::AddEdge(reduceProdNode->GetOutDataAnchor(0), fusion_node->GetInDataAnchor(index));
  if (reduceProdNode->AddLinkFrom(const_node) != ge::GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT("ReduceProd", "Addlink from failed.");
    return FAILED;
  }
  OP_LOGD(FUSED_OP_TYPE_PROD.c_str(), "RandomInsertReduceProdFusionPass fusion pass end");
  return SUCCESS;
}

Status RandomDsaFusionPass::CreateConstOperator(ge::ComputeGraph& graph, ge::NodePtr& fusion_node,
                                                ge::NodePtr& const_node) {
  // Get attr value
  int32_t seed0 = 0;
  int32_t seed1 = 0;
  Operator op_drop = ge::OpDescUtils::CreateOperatorFromNode(fusion_node);
  if (GRAPH_SUCCESS != op_drop.GetAttr("seed", seed0)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get attr seed0 failed.");
    return GRAPH_FAILED;
  }

  if (GRAPH_SUCCESS != op_drop.GetAttr("seed2", seed1)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get attr seed1 failed.");
    return GRAPH_FAILED;
  }
  uint64_t seed_tmp = seed0;
  seed_tmp=(seed_tmp<<MOVE_32_NUM) | seed1;
  const uint64_t seed = seed_tmp;
  // set seed
  ge::GeTensorDesc tmp_desc;
  ge::GeTensorPtr seed_ptr = nullptr;
  std::unique_ptr<uint64_t> seed_data_tmp(new (std::nothrow) uint64_t(seed));
  FUSION_PASS_MAKE_SHARED((seed_ptr = std::make_shared<ge::GeTensor>(
                               tmp_desc, reinterpret_cast<uint8_t*>(seed_data_tmp.get()), sizeof(uint64_t))),
                          seed_ptr = nullptr;
                          return PARAM_INVALID);
  Status ret = seed_ptr->SetData(reinterpret_cast<uint8_t*>(seed_data_tmp.get()), sizeof(uint64_t));
  if (ret != SUCCESS) {
    OP_LOGW("set seed data failed.");
    return ret;
  }
  seed_ptr->MutableTensorDesc().SetDataType(ge::DT_UINT64);
  ge::OpDescPtr const_opdesc = ge::OpDescUtils::CreateConstOp(seed_ptr);
  FUSION_PASS_CHECK(const_opdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a const object failed."), return FAILED);
  string constOpName = fusion_node->GetName() + "_seed";
  const_opdesc->SetName(constOpName);
  const_node = graph.AddNode(const_opdesc);
  return SUCCESS;
}

Status RandomDsaFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGD("DSA OP Start Fusion");
  PlatformInfo platformInfo;
  OptionalInfo optionalInfo;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Get platform info failed, not fusion.");
    return SUCCESS;
  }
  if (optionalInfo.soc_version != "Ascend920A") {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "DSA fusion pass not support this soc version[%s].",
            optionalInfo.soc_version.c_str());
    return SUCCESS;
  }
  ge::NodePtr fusedNode_Drop = GetNodeFromMapping(PATTERN_FUSEDNODE_DROP, mapping);
  ge::NodePtr fusedNode_Uniform = GetNodeFromMapping(PATTERN_FUSEDNODE_UNIFORM, mapping);
  if (fusedNode_Drop == nullptr && fusedNode_Uniform == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "all related fusedNode randomdsa  is null, fusion failed.");
    return NOT_CHANGED;
  }

  // Get DropOut Input Info
  if (fusedNode_Drop != nullptr) {
    ge::OpDescPtr fusedDesc_Drop = fusedNode_Drop->GetOpDesc();
    // set dsa engine
    ge::GeTensorDesc dropout_input_desc0 = fusedNode_Drop->GetOpDesc()->GetInputDesc(0);
    ge::GeTensorDesc dropout_input_desc1 = fusedNode_Drop->GetOpDesc()->GetInputDesc(1);
    ge::GeTensorDesc output_desc0 = fusedNode_Drop->GetOpDesc()->GetOutputDesc(0);
    DataType datatype_drop = dropout_input_desc0.GetDataType();
    // DropoutGenMask input size must be 2
    if (fusedDesc_Drop->GetInputsSize() < 2) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "DropoutGenMask Index is beyond the size[%d] of output desc",
              fusedDesc_Drop->GetInputsSize());
      return NOT_CHANGED;
    }
    // define support dtype
    std::set<DataType> supported_dtypes = {ge::DT_INT64, ge::DT_INT32};
    if (supported_dtypes.count(datatype_drop) == 0) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input dtype not supported, fusion failed.");
      return NOT_CHANGED;
    }
    ge::NodePtr const_node = nullptr;
    // create const node
    CreateConstOperator(graph, fusedNode_Drop, const_node);
    ge::OpDescPtr dsagenbitmask_op = nullptr;
    FUSION_PASS_MAKE_SHARED((dsagenbitmask_op = std::make_shared<ge::OpDesc>(
                                 fusedNode_Drop->GetName() + "/" + "DSAGENBITMASK", "DSAGenBitMask")),
                            return INTERNAL_ERROR);
    ge::GeTensorDesc dropout_seed_desc;
    dropout_seed_desc.SetDataType(ge::DT_UINT64);
    dropout_seed_desc.SetOriginDataType(ge::DT_UINT64);
    dsagenbitmask_op->AddInputDesc("count", dropout_input_desc0);
    dsagenbitmask_op->AddInputDesc("seed", dropout_seed_desc);
    dsagenbitmask_op->AddInputDesc("dropout", dropout_input_desc1);
    dsagenbitmask_op->AddOutputDesc("y", output_desc0);
    ge::NodePtr dsagenbitmask_node = graph.AddNode(dsagenbitmask_op);
    // add dropout in edge to dsagenbitmask
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode_Drop->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                           dsagenbitmask_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                                       fusedNode_Drop->GetName().c_str(), 0, dsagenbitmask_node->GetName().c_str(), 0),
        return FAILED);

    if (dsagenbitmask_node->AddLinkFrom(1, const_node) != SUCCESS) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                                     const_node->GetName().c_str(), 1, dsagenbitmask_node->GetName().c_str(), 1);
      return FAILED;
    }

    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode_Drop->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                           dsagenbitmask_node->GetInDataAnchor(2)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                                       fusedNode_Drop->GetName().c_str(), 2, dsagenbitmask_node->GetName().c_str(), 2),
        return FAILED);

    for (auto in_ctl_anchor : fusedNode_Drop->GetInControlAnchor()->GetPeerOutControlAnchors()) {
      (void)ge::GraphUtils::AddEdge(in_ctl_anchor, dsagenbitmask_node->GetInControlAnchor());
    }

    // unlink dropout
    if (fusedNode_Drop->GetInControlAnchor() != nullptr) {
      fusedNode_Drop->GetInControlAnchor()->UnlinkAll();
    }
    for (auto& inAnchor : fusedNode_Drop->GetAllInDataAnchors()) {
      if (inAnchor != nullptr) {
        inAnchor->UnlinkAll();
      }
    }
    for (auto& out_anchor : fusedNode_Drop->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      (void)ge::GraphUtils::RemoveEdge(fusedNode_Drop->GetOutDataAnchor(0), out_anchor);
      (void)ge::GraphUtils::AddEdge(dsagenbitmask_node->GetOutDataAnchor(0), out_anchor);
    }
    // flattern shape to count by insert reduceprod
    FlatternShape(graph, dsagenbitmask_node, kCountIndex);

    // set count dtype
    dsagenbitmask_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_INT64);
    dsagenbitmask_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(ge::DT_INT64);
    // remove dropout node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode_Drop),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed",
                                                     fusedNode_Drop->GetName().c_str()),
                      return FAILED);
    fusionNodes.push_back(dsagenbitmask_node);
    return SUCCESS;
  }

  if (fusedNode_Uniform != nullptr) {
    ge::OpDescPtr fusedDesc_Uniform = fusedNode_Uniform->GetOpDesc();
    // set dsa engine
    ge::GeTensorDesc uniform_input_desc0 = fusedNode_Uniform->GetOpDesc()->GetInputDesc(0);
    ge::GeTensorDesc uniform_input_desc1 = fusedNode_Uniform->GetOpDesc()->GetInputDesc(1);
    ge::GeTensorDesc uniform_input_desc2 = fusedNode_Uniform->GetOpDesc()->GetInputDesc(2);
    ge::GeTensorDesc output_desc0 = fusedNode_Uniform->GetOpDesc()->GetOutputDesc(0);
    DataType datatype_uniform = uniform_input_desc0.GetDataType();

    // define support dtype
    std::set<DataType> supported_dtypes = {ge::DT_INT64, ge::DT_INT32};
    if (supported_dtypes.count(datatype_uniform) == 0) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "input dtype not supported, fusion failed.");
      return NOT_CHANGED;
    }

    // uniform input size must be 3
    if (fusedDesc_Uniform->GetInputsSize() < UNIFORM_INPUT_SIZE) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "RandomUniform Index is beyond the size[%d] of output desc",
              fusedDesc_Uniform->GetInputsSize());
      return NOT_CHANGED;
    }

    ge::NodePtr const_node = nullptr;
    // create const node
    CreateConstOperator(graph, fusedNode_Uniform, const_node);
    ge::OpDescPtr dsauniform_op = nullptr;
    FUSION_PASS_MAKE_SHARED((dsauniform_op = std::make_shared<ge::OpDesc>(
                                 fusedDesc_Uniform->GetName() + "/" + "DSARANDOMUNIFOTM", "DSARandomUniform")),
                            return INTERNAL_ERROR);
    ge::GeTensorDesc uniform_seed_desc;
    uniform_seed_desc.SetDataType(ge::DT_UINT64);
    uniform_seed_desc.SetOriginDataType(ge::DT_UINT64);
    dsauniform_op->AddInputDesc("count", uniform_input_desc0);
    dsauniform_op->AddInputDesc("seed", uniform_seed_desc);
    dsauniform_op->AddInputDesc("low", uniform_input_desc1);
    dsauniform_op->AddInputDesc("high", uniform_input_desc2);
    dsauniform_op->AddOutputDesc("y", output_desc0);
    ge::NodePtr dsauniform_node = graph.AddNode(dsauniform_op);
    // add dropout in edge to dsagenbitmask
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode_Uniform->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                           dsauniform_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                                       fusedNode_Uniform->GetName().c_str(), 0, dsauniform_node->GetName().c_str(), 0),
        return FAILED);

    if (dsauniform_node->AddLinkFrom(1, const_node) != SUCCESS) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                                     const_node->GetName().c_str(), 1, dsauniform_node->GetName().c_str(), 1);
      return FAILED;
    }

    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode_Uniform->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                           dsauniform_node->GetInDataAnchor(2)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                                       fusedNode_Uniform->GetName().c_str(), 1, dsauniform_node->GetName().c_str(), 2),
        return FAILED);

    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode_Uniform->GetInDataAnchor(2)->GetPeerOutAnchor(),
                                           dsauniform_node->GetInDataAnchor(3)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                                       fusedNode_Uniform->GetName().c_str(), 2, dsauniform_node->GetName().c_str(), 3),
        return FAILED);
    for (auto in_ctl_anchor : fusedNode_Uniform->GetInControlAnchor()->GetPeerOutControlAnchors()) {
      (void)ge::GraphUtils::AddEdge(in_ctl_anchor, dsauniform_node->GetInControlAnchor());
    }

    // unlink dropout
    if (fusedNode_Uniform->GetInControlAnchor() != nullptr) {
      fusedNode_Uniform->GetInControlAnchor()->UnlinkAll();
    }
    for (auto& inAnchor : fusedNode_Uniform->GetAllInDataAnchors()) {
      if (inAnchor != nullptr) {
        inAnchor->UnlinkAll();
      }
    }
    for (auto& out_anchor : fusedNode_Uniform->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      (void)ge::GraphUtils::RemoveEdge(fusedNode_Uniform->GetOutDataAnchor(0), out_anchor);
      (void)ge::GraphUtils::AddEdge(dsauniform_node->GetOutDataAnchor(0), out_anchor);
    }
    // flattern shape to count by insert reduceprod
    FlatternShape(graph, dsauniform_node, kCountIndex);

    // set count dtype
    dsauniform_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_INT64);
    dsauniform_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(ge::DT_INT64);

    // remove dropout node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode_Uniform),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed",
                                                     fusedNode_Uniform->GetName().c_str()),
                      return FAILED);
    fusionNodes.push_back(dsauniform_node);
    return SUCCESS;
  }
  return NOT_CHANGED;
}
REGISTER_PASS("RandomDsaFusionPass", BUILT_IN_GRAPH_PASS, RandomDsaFusionPass);
}  // namespace fe
