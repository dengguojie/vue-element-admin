/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file splitv_fusion_pass.cpp
 * \brief
 */
#include "splitv_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "SplitV";
static const std::string PATTERN_FUSEDNODE = "FusedNodeSplitV";
static const size_t SPLIT_MAX_NUM = 63;
/*
1:
SplitV ---> SplitVD

2:
          input                                                             input
            |                                                                 |
       S p l i t V D            ---  -fusion-  --->                    S  p  l  i  t  V  D
      / ...   |        \                                             /         |        \
     /  ...   |         \                                           /          |         \
output_1 ... output_m .. output_n                             SplitVD_1 ... SplitVD_M ... SplitVD_N
                                                             /    |   \    /    |     \    /    |    \
                                                      output_1  output_2 ... output_m  ...           output_n
*/
vector<FusionPattern*> SplitVFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SplitVFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

Status SplitVFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // build attr infos
  std::string fusionOpType = "SplitVD";
  std::vector<PassAttrInfo> splitvAttrInfo;
  // get node
  ge::NodePtr fused_node1 = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed"),
                    return PARAM_INVALID);

  ge::OpDescPtr fuseDesc1 = fused_node1->GetOpDesc();
  FUSION_PASS_CHECK(fuseDesc1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "OpDesc of node1 is null, fusion failed."),
                    return PARAM_INVALID);

  if (HasUnKnowShape(fused_node1)) {
    FUSION_PASS_CHECK(CheckOpSupported(fuseDesc1), OP_LOGI(FUSED_NODE, "split_v dynamic shape supported"),
                      return NOT_CHANGED);
    OP_LOGI(FUSED_NODE, "CheckOpSupported fail, split_v dynamic");
  }

  PassAttrInfo size_splits_st = {1, "size_splits", "SetListInt"};
  splitvAttrInfo.push_back(size_splits_st);
  PassAttrInfo split_dim_st = {2, "split_dim", "SetInt"};
  splitvAttrInfo.push_back(split_dim_st);

  // build a fusion node op desc
  OpDescPtr fusion_desc = PatternFusionUtil::GetFusionOpDesc(fused_node1, fusionOpType, splitvAttrInfo);
  FUSION_PASS_CHECK(fusion_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusion op desc is nullptr."),
                    return NOT_CHANGED);

  // check op support
  FUSION_PASS_CHECK(!CheckOpSupported(fusion_desc), OP_LOGI(FUSED_OP_TYPE.c_str(), "Split not supported."),
                    return NOT_CHANGED);

  ge::NodePtr fused_node = nullptr;
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, fused_node1, fusionOpType, splitvAttrInfo, fused_node);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "SplitV has input which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }
  ClearOpInferDepends(fused_node1);

  ge::OpDescPtr fusedDesc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(
      fusedDesc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fused_node's OpDesc is null, fusion failed."),
      return PARAM_INVALID);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "SplitV fusion SUCCESSS!!!!!");

  // A maximum of 63 tensors are supported in mini mode.
  int64_t num_split;
  ge::AttrUtils::GetInt(fusedDesc, "num_split", num_split);
  FUSION_PASS_CHECK(num_split <= SPLIT_MAX_NUM,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The amount of num_split of SplitVD node is less than 63."),
                    return SUCCESS);

  if (num_split > SPLIT_MAX_NUM) {
    size_t nodes_num;
    size_t nodes_num1 = num_split % SPLIT_MAX_NUM;
    if (nodes_num1 == 0) {
      nodes_num = num_split / SPLIT_MAX_NUM;
    } else {
      nodes_num = num_split / SPLIT_MAX_NUM + 1;
    }
    size_t last_node_num_split = num_split - (SPLIT_MAX_NUM * (nodes_num - 1));

    vector<int64_t> size_splits;
    ge::AttrUtils::GetListInt(fusedDesc, "size_splits", size_splits);

    int64_t split_dim_num;
    ge::AttrUtils::GetInt(fusedDesc, "split_dim", split_dim_num);

    vector<int64_t> size_splits_new;
    for (size_t i = 0; i < nodes_num; i++) {
      if (i < nodes_num - 1) {
        int64_t size = 0;
        for (size_t m = 0; m < SPLIT_MAX_NUM; m++) {
          size += size_splits[SPLIT_MAX_NUM * i + m];
        }
        size_splits_new.push_back(size);
      } else {
        int64_t size = 0;
        for (size_t m = 0; m < last_node_num_split; m++) {
          size += size_splits[SPLIT_MAX_NUM * i + m];
        }
        size_splits_new.push_back(size);
      }
    }

    ge::OpDescPtr SplitVDBaseDesc = AttrUtils::CopyOpDesc(fusedDesc);
    SplitVDBaseDesc->SetName(SplitVDBaseDesc->GetName() + "/SplitVD" + "Base_node");
    SplitVDBaseDesc->SetType("SplitVD");

    std::vector<ge::GeTensorDesc> outputDesc;
    for (size_t c = num_split - 1; c >= nodes_num; c--) {
      OpDescUtils::ClearOutputDesc(SplitVDBaseDesc, c);
    }

    ge::NodePtr splitvd_base_node = graph.AddNode(SplitVDBaseDesc);
    FUSION_PASS_CHECK(splitvd_base_node == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "splitvd_base_node is null, fusion failed."),
                      return PARAM_INVALID);
    fusionNodes.push_back(splitvd_base_node);
    ge::AttrUtils::SetListInt(splitvd_base_node->GetOpDesc(), "size_splits", size_splits_new);
    ge::AttrUtils::SetInt(splitvd_base_node->GetOpDesc(), "split_dim", split_dim_num);
    ge::AttrUtils::SetInt(splitvd_base_node->GetOpDesc(), "num_split", nodes_num);

    ge::GeTensorDesc SplitVDInputTensor_1 = SplitVDBaseDesc->GetInputDesc(0);
    ge::GeShape SplitVDInputShape_1 = SplitVDInputTensor_1.GetShape();
    int64_t dimnum = SplitVDInputShape_1.GetDimNum();
    if (split_dim_num < 0) {
      split_dim_num += dimnum;
    }
    for (size_t h = 0; h < nodes_num; h++) {
      ge::GeTensorDesc SplitVDOutputTensor_1 = SplitVDBaseDesc->GetOutputDesc(h);
      ge::GeShape SplitVDOutputShape_1 = SplitVDOutputTensor_1.GetShape();

      SplitVDOutputShape_1.SetDim(split_dim_num, size_splits_new[h]);
      SplitVDOutputTensor_1.SetShape(SplitVDOutputShape_1);
      SplitVDOutputTensor_1.SetOriginShape(SplitVDOutputShape_1);

      SplitVDBaseDesc->UpdateOutputDesc(h, SplitVDOutputTensor_1);
      outputDesc.push_back(SplitVDOutputTensor_1);
    }

    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                           splitvd_base_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                                       fused_node->GetName().c_str(), (0), splitvd_base_node->GetName().c_str(), 0),
        return FAILED);

    for (size_t i = 0; i < nodes_num; i++) {
      if (i < nodes_num - 1) {
        ge::OpDescPtr SplitVDDesc = AttrUtils::CopyOpDesc(fusedDesc);
        SplitVDDesc->SetName(SplitVDDesc->GetName() + "/SplitVD_" + to_string(i));
        SplitVDDesc->SetType("SplitVD");
        for (size_t c = num_split - 1; c >= SPLIT_MAX_NUM; c--) {
          OpDescUtils::ClearOutputDesc(SplitVDDesc, c);
        }
        ge::NodePtr splitvd_node = graph.AddNode(SplitVDDesc);
        FUSION_PASS_CHECK(
            splitvd_node == nullptr,
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitvd_node is null, fusion failed."),
            return PARAM_INVALID);
        fusionNodes.push_back(splitvd_node);

        vector<int64_t> size_splits_new2;
        for (size_t n = 0; n < SPLIT_MAX_NUM; n++) {
          size_splits_new2.push_back(size_splits[SPLIT_MAX_NUM * i + n]);
        }

        ge::AttrUtils::SetInt(splitvd_node->GetOpDesc(), "split_dim", split_dim_num);
        ge::AttrUtils::SetInt(splitvd_node->GetOpDesc(), "num_split", SPLIT_MAX_NUM);

        ge::AttrUtils::SetListInt(splitvd_node->GetOpDesc(), "size_splits", size_splits_new2);

        SplitVDDesc->UpdateInputDesc(0, outputDesc[i]);
        for (int64_t h = 0; h < SPLIT_MAX_NUM; h++) {
          ge::GeTensorDesc SplitVDOutputTensor_2 = SplitVDDesc->GetOutputDesc(h);
          ge::GeShape SplitVDOutputShape_2 = SplitVDOutputTensor_2.GetShape();
          SplitVDOutputShape_2.SetDim(split_dim_num, size_splits_new2[h]);
          SplitVDOutputTensor_2.SetShape(SplitVDOutputShape_2);
          SplitVDDesc->UpdateOutputDesc(h, SplitVDOutputTensor_2);
        }

        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitvd_base_node->GetOutDataAnchor(i),
                                                             splitvd_node->GetInDataAnchor(0)),
                          VECTOR_FUSION_INNER_ERR_REPORT(
                              FUSED_OP_TYPE.c_str(),
                              "Add edge from fused node:%s's index[%lu] to fusion node:%s's index[%lu] failed.",
                              splitvd_base_node->GetName().c_str(), i, splitvd_node->GetName().c_str(), i),
                          return FAILED);

        for (size_t m = 0; m < SPLIT_MAX_NUM; m++) {
          for (InDataAnchorPtr inAnchorPtr :
               fused_node->GetOutDataAnchor(SPLIT_MAX_NUM * i + m)->GetPeerInDataAnchors()) {
            FUSION_PASS_CHECK(
                SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(SPLIT_MAX_NUM * i + m), inAnchorPtr),
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitvd_node->GetOutDataAnchor(m), inAnchorPtr),
                              VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."),
                              return FAILED);
          }
        }
      } else {
        if (last_node_num_split != 1) {
          ge::OpDescPtr LastSplitVDDesc = AttrUtils::CopyOpDesc(fusedDesc);
          LastSplitVDDesc->SetName(LastSplitVDDesc->GetName() + "/SplitVD" + to_string(nodes_num - 1));
          LastSplitVDDesc->SetType("SplitVD");
          for (size_t c = num_split - 1; c >= last_node_num_split; c--) {
            OpDescUtils::ClearOutputDesc(LastSplitVDDesc, c);
          }
          ge::NodePtr last_splitvd_node = graph.AddNode(LastSplitVDDesc);
          FUSION_PASS_CHECK(last_splitvd_node == nullptr,
                            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                           "last_splitvd_nodeis null, fusion failed."),
                            return PARAM_INVALID);
          fusionNodes.push_back(last_splitvd_node);

          vector<int64_t> size_splits_new3;
          for (size_t n = 0; n < last_node_num_split; n++) {
            size_splits_new3.push_back(size_splits[SPLIT_MAX_NUM * i + n]);
          }

          ge::AttrUtils::SetInt(last_splitvd_node->GetOpDesc(), "split_dim", split_dim_num);
          ge::AttrUtils::SetInt(last_splitvd_node->GetOpDesc(), "num_split", last_node_num_split);
          ge::AttrUtils::SetListInt(last_splitvd_node->GetOpDesc(), "size_splits", size_splits_new3);

          LastSplitVDDesc->UpdateInputDesc(0, outputDesc[nodes_num - 1]);
          for (size_t h = 0; h < last_node_num_split; h++) {
            ge::GeTensorDesc SplitVDOutputTensor_3 = LastSplitVDDesc->GetOutputDesc(h);
            ge::GeShape SplitVDOutputShape_3 = SplitVDOutputTensor_3.GetShape();
            SplitVDOutputShape_3.SetDim(split_dim_num, size_splits_new3[h]);
            SplitVDOutputTensor_3.SetShape(SplitVDOutputShape_3);
            LastSplitVDDesc->UpdateOutputDesc(h, SplitVDOutputTensor_3);
          }

          FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitvd_base_node->GetOutDataAnchor(i),
                                                               last_splitvd_node->GetInDataAnchor(0)),
                            VECTOR_FUSION_INNER_ERR_REPORT(
                                FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's index[%lu] to fusion node:%s's index[%lu] failed.",
                                splitvd_base_node->GetName().c_str(), i, last_splitvd_node->GetName().c_str(), i),
                            return FAILED);

          for (size_t m = 0; m < last_node_num_split; m++) {
            for (InDataAnchorPtr inAnchorPtr :
                 fused_node->GetOutDataAnchor(SPLIT_MAX_NUM * i + m)->GetPeerInDataAnchors()) {
              FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(
                                               fused_node->GetOutDataAnchor(SPLIT_MAX_NUM * i + m), inAnchorPtr),
                                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."),
                                return FAILED);
              FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(last_splitvd_node->GetOutDataAnchor(m), inAnchorPtr),
                                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."),
                                return FAILED);
            }
          }
        } else {
          for (InDataAnchorPtr inAnchorPtr : fused_node->GetOutDataAnchor(SPLIT_MAX_NUM * i)->GetPeerInDataAnchors()) {
            FUSION_PASS_CHECK(
                SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(SPLIT_MAX_NUM * i), inAnchorPtr),
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitvd_base_node->GetOutDataAnchor(i), inAnchorPtr),
                              VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."),
                              return FAILED);
          }
        }
      }
    }
  }

  for (auto inAnchor : fused_node->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  for (auto outAnchor : fused_node->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }

  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(fused_node),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove Node [%s] failed", fused_node->GetName().c_str()),
      return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "SplitV --> SplitVD fusion SUCCESSS!!!!!");
  return SUCCESS;
}

REGISTER_PASS("ZSplitVFusionPass", BUILT_IN_GRAPH_PASS, SplitVFusionPass);
}  // namespace fe
