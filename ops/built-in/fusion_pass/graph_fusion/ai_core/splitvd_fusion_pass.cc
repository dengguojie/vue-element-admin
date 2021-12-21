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
 * \file splitvd_fusion_pass.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "splitvd_fusion_pass.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {

static const char* FUSED_NODE = "SplitVD";
static const std::string PATTERN_FUSEDNODE = "FusedNodeSplitVD";
/*
          input                                                             input
            |                                                                 |
       S p l i t V D            ---  -fusion-  --->                    S  p  l  i  t  V  D
      / ...   |        \                                             /         |        \
     /  ...   |         \                                           /          |         \
output_1 ... output_m .. output_n                             SplitVD_1 ... SplitVD_M ... SplitVD_N
                                                             /    |   \    /    |     \    /    |    \
                                                      output_1  output_2 ... output_m  ...           output_n
*/
vector<FusionPattern*> SplitVDFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SplitVDFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

Status SplitVDFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  ge::OpDescPtr fusedDesc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fused_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // A maximum of 63 tensors are supported in mini mode.
  int64_t num_split;
  ge::AttrUtils::GetInt(fusedDesc, "num_split", num_split);
  FUSION_PASS_CHECK(num_split <= 63,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The amount of num_split of SplitVD node is less than 63."),
                    return SUCCESS);

  if (num_split > 63) {
    size_t nodes_num;
    size_t nodes_num1 = num_split % 63;
    if (nodes_num1 == 0) {
      nodes_num = num_split / 63;
    } else {
      nodes_num = num_split / 63 + 1;
    }
    size_t last_node_num_split = num_split - (63 * (nodes_num - 1));

    vector<int64_t> size_splits;
    ge::AttrUtils::GetListInt(fusedDesc, "size_splits", size_splits);

    int64_t split_dim;
    ge::AttrUtils::GetInt(fusedDesc, "split_dim", split_dim);

    vector<int64_t> size_splits_new;
    for (size_t i = 0; i < nodes_num; i++) {
      if (i < nodes_num - 1) {
        int64_t size = 0;
        for (size_t m = 0; m < 63; m++) {
          size += size_splits[63 * i + m];
        }
        size_splits_new.push_back(size);
      } else {
        int64_t size = 0;
        for (size_t m = 0; m < last_node_num_split; m++) {
          size += size_splits[63 * i + m];
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
    fusionNodes.push_back(splitvd_base_node);
    ge::AttrUtils::SetListInt(splitvd_base_node->GetOpDesc(), "size_splits", size_splits_new);
    ge::AttrUtils::SetInt(splitvd_base_node->GetOpDesc(), "split_dim", split_dim);
    ge::AttrUtils::SetInt(splitvd_base_node->GetOpDesc(), "num_split", nodes_num);

    ge::GeTensorDesc SplitVDInputTensor_1 = SplitVDBaseDesc->GetInputDesc(0);
    ge::GeShape SplitVDInputShape_1 = SplitVDInputTensor_1.GetShape();
    int64_t dimnum = SplitVDInputShape_1.GetDimNum();
    if (split_dim < 0) {
      split_dim += dimnum;
    }
    for (size_t h = 0; h < nodes_num; h++) {
      ge::GeTensorDesc SplitVDOutputTensor_1 = SplitVDBaseDesc->GetOutputDesc(h);
      ge::GeShape SplitVDOutputShape_1 = SplitVDOutputTensor_1.GetShape();

      SplitVDOutputShape_1.SetDim(split_dim, size_splits_new[h]);
      SplitVDOutputTensor_1.SetShape(SplitVDOutputShape_1);
      SplitVDOutputTensor_1.SetOriginShape(SplitVDOutputShape_1);

      SplitVDBaseDesc->UpdateOutputDesc(h, SplitVDOutputTensor_1);
      outputDesc.push_back(SplitVDOutputTensor_1);
    }
    FUSION_PASS_CHECK(splitvd_base_node == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitvd_base_node:%s is null, fusion failed.",
                              splitvd_base_node->GetName().c_str()),
                      return PARAM_INVALID);

    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                           splitvd_base_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                fused_node->GetName().c_str(), (0), splitvd_base_node->GetName().c_str(), 0),
        return FAILED);

    for (size_t i = 0; i < nodes_num; i++) {
      if (i < nodes_num - 1) {
        ge::OpDescPtr SplitVDDesc = AttrUtils::CopyOpDesc(fusedDesc);
        SplitVDDesc->SetName(SplitVDDesc->GetName() + "/SplitVD_" + to_string(i));
        SplitVDDesc->SetType("SplitVD");
        for (size_t c = num_split - 1; c >= 63; c--) {
          OpDescUtils::ClearOutputDesc(SplitVDDesc, c);
        }
        ge::NodePtr splitvd_node = graph.AddNode(SplitVDDesc);
        fusionNodes.push_back(splitvd_node);

        vector<int64_t> size_splits_new2;
        for (size_t n = 0; n < 63; n++) {
          size_splits_new2.push_back(size_splits[63 * i + n]);
        }

        ge::AttrUtils::SetInt(splitvd_node->GetOpDesc(), "split_dim", split_dim);
        ge::AttrUtils::SetInt(splitvd_node->GetOpDesc(), "num_split", 63);

        ge::AttrUtils::SetListInt(splitvd_node->GetOpDesc(), "size_splits", size_splits_new2);

        SplitVDDesc->UpdateInputDesc(0, outputDesc[i]);
        for (int64_t h = 0; h < 63; h++) {
          ge::GeTensorDesc SplitVDOutputTensor_2 = SplitVDDesc->GetOutputDesc(h);
          ge::GeShape SplitVDOutputShape_2 = SplitVDOutputTensor_2.GetShape();
          SplitVDOutputShape_2.SetDim(split_dim, size_splits_new2[h]);
          SplitVDOutputTensor_2.SetShape(SplitVDOutputShape_2);
          SplitVDDesc->UpdateOutputDesc(h, SplitVDOutputTensor_2);
        }

        FUSION_PASS_CHECK(
            splitvd_node == nullptr,
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitvd_node:%s is null, fusion failed.", splitvd_node->GetName().c_str()),
            return PARAM_INVALID);

        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitvd_base_node->GetOutDataAnchor(i),
                                                             splitvd_node->GetInDataAnchor(0)),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                  "Add edge from fused node:%s's index[%lu] to fusion node:%s's index[%lu] failed.",
                                  splitvd_base_node->GetName().c_str(), i, splitvd_node->GetName().c_str(), i),
                          return FAILED);

        for (size_t m = 0; m < 63; m++) {
          for (InDataAnchorPtr inAnchorPtr : fused_node->GetOutDataAnchor(63 * i + m)->GetPeerInDataAnchors()) {
            FUSION_PASS_CHECK(
                SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(63 * i + m), inAnchorPtr),
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitvd_node->GetOutDataAnchor(m), inAnchorPtr),
                              VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
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
          fusionNodes.push_back(last_splitvd_node);

          vector<int64_t> size_splits_new3;
          for (size_t n = 0; n < last_node_num_split; n++) {
            size_splits_new3.push_back(size_splits[63 * i + n]);
          }

          ge::AttrUtils::SetInt(last_splitvd_node->GetOpDesc(), "split_dim", split_dim);
          ge::AttrUtils::SetInt(last_splitvd_node->GetOpDesc(), "num_split", last_node_num_split);
          ge::AttrUtils::SetListInt(last_splitvd_node->GetOpDesc(), "size_splits", size_splits_new3);

          LastSplitVDDesc->UpdateInputDesc(0, outputDesc[nodes_num - 1]);
          for (size_t h = 0; h < last_node_num_split; h++) {
            ge::GeTensorDesc SplitVDOutputTensor_3 = LastSplitVDDesc->GetOutputDesc(h);
            ge::GeShape SplitVDOutputShape_3 = SplitVDOutputTensor_3.GetShape();
            SplitVDOutputShape_3.SetDim(split_dim, size_splits_new3[h]);
            SplitVDOutputTensor_3.SetShape(SplitVDOutputShape_3);
            LastSplitVDDesc->UpdateOutputDesc(h, SplitVDOutputTensor_3);
          }

          FUSION_PASS_CHECK(last_splitvd_node == nullptr,
                            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "last_splitvd_node:%s is null, fusion failed.",
                                    last_splitvd_node->GetName().c_str()),
                            return PARAM_INVALID);
          FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitvd_base_node->GetOutDataAnchor(i),
                                                               last_splitvd_node->GetInDataAnchor(0)),
                            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                    "Add edge from fused node:%s's index[%lu] to fusion node:%s's index[%lu] failed.",
                                    splitvd_base_node->GetName().c_str(), i, last_splitvd_node->GetName().c_str(), i),
                            return FAILED);

          for (size_t m = 0; m < last_node_num_split; m++) {
            for (InDataAnchorPtr inAnchorPtr : fused_node->GetOutDataAnchor(63 * i + m)->GetPeerInDataAnchors()) {
              FUSION_PASS_CHECK(
                  SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(63 * i + m), inAnchorPtr),
                  VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
              FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(last_splitvd_node->GetOutDataAnchor(m), inAnchorPtr),
                                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
            }
          }
        } else {
          for (InDataAnchorPtr inAnchorPtr : fused_node->GetOutDataAnchor(63 * i)->GetPeerInDataAnchors()) {
            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(63 * i), inAnchorPtr),
                              VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitvd_base_node->GetOutDataAnchor(i), inAnchorPtr),
                              VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
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

  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fused_node),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove Node [%s] failed", fused_node->GetName().c_str()),
                    return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "SplitVD --> SplitVD fusion SUCCESSS!!!!!");
  return SUCCESS;
}

REGISTER_PASS("ZSplitVDFusionPass", BUILT_IN_GRAPH_PASS, SplitVDFusionPass);
}  // namespace fe
