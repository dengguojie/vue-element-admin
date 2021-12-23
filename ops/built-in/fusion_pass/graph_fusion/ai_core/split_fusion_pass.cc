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
 * \file split_fusion_pass.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "split_fusion_pass.h"
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

static const char* FUSED_NODE = "Split";
static const std::string PATTERN_FUSEDNODE = "FusedNodeSplit";
/*
1:
Split ---> SplitD

2:
          input                                                             input
            |                                                                 |
       S p l i t D            ---  -fusion-  --->                    S  p  l  i  t  V  D
      / ...   |        \                                             /         |        \
     /  ...   |         \                                           /          |         \
output_1 ... output_m .. output_n                             SplitVD_1 ... SplitVD_M ... SplitVD_N
                                                             /    |   \    /    |     \    /    |    \
                                                      output_1  output_2 ... output_m  ...           output_n
*/
vector<FusionPattern*> SplitFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SplitFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

Status SplitFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  // build attr infos
  std::vector<PassAttrInfo> splitAttrInfo;
  std::string fusionOpType = "SplitD";
  PassAttrInfo split_dim = {0, "split_dim", "SetInt"};
  splitAttrInfo.push_back(split_dim);

  // get node
  ge::NodePtr fused_node1 = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed"),
                    return PARAM_INVALID);

  // get desc
  ge::OpDescPtr fuse_desc1 = fused_node1->GetOpDesc();
  FUSION_PASS_CHECK(fuse_desc1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fused_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  // split dynamic shape check supported
  if (HasUnKnowShape(fused_node1)) {
    FUSION_PASS_CHECK(CheckOpSupported(fuse_desc1), OP_LOGI(FUSED_NODE, "split dynamic shape supported"),
                      return NOT_CHANGED);
    OP_LOGI(FUSED_NODE, "CheckOpSupported fail, split dynamic");
  }
  // build a fusion node op desc
  OpDescPtr fusion_desc = PatternFusionUtil::GetFusionOpDesc(fused_node1, fusionOpType, splitAttrInfo);
  FUSION_PASS_CHECK(fusion_desc == nullptr, OP_LOGI(FUSED_OP_TYPE, "fusion op desc is nullptr."),
                    return NOT_CHANGED);

  // check op support
  FUSION_PASS_CHECK(!CheckOpSupported(fusion_desc), OP_LOGI(FUSED_OP_TYPE.c_str(), "Split not supported."),
                    return NOT_CHANGED);

  int64_t num_split;
  ge::AttrUtils::GetInt(fuse_desc1, "num_split", num_split);
  ge::GeTensorDesc SplitInputTensor = fuse_desc1->GetInputDesc("x");
  ge::GeShape input_shape = SplitInputTensor.GetShape();

  if (IsUnknownShape(input_shape.GetDims()) && num_split > 63) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "ZSplitFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }

  ge::NodePtr fused_node = nullptr;
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, fused_node1, fusionOpType, splitAttrInfo, fused_node);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Split has input which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }

  ClearOpInferDepends(fused_node1);
  ge::OpDescPtr fusedDesc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fused_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // A maximum of 63 tensors are supported in mini mode.
  ge::AttrUtils::GetInt(fusedDesc, "num_split", num_split);
  FUSION_PASS_CHECK(num_split <= 63,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The amount of num_split of SplitD node is less than 63."),
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

    ge::AttrUtils::GetInt(fusedDesc, "num_split", num_split);

    int64_t split_dim;
    ge::AttrUtils::GetInt(fusedDesc, "split_dim", split_dim);
    vector<int64_t> size_splits;
    int64_t small_split_size;

    ge::GeTensorDesc SplitDInputTensor = fusedDesc->GetInputDesc(0);
    ge::GeShape SplitDInputShape = SplitDInputTensor.GetShape();
    int64_t dimnum = SplitDInputShape.GetDimNum();
    if (split_dim < 0) {
      split_dim += dimnum;
    }

    small_split_size = SplitDInputShape.GetDim(split_dim) / num_split;
    for (int64_t x = 0; x < num_split; x++) {
      size_splits.push_back(small_split_size);
    }

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

    ge::OpDescPtr SplitDBaseDesc = AttrUtils::CopyOpDesc(fusedDesc);
    SplitDBaseDesc->SetName(SplitDBaseDesc->GetName() + "/SplitVD" + "Base_node");
    SplitDBaseDesc->SetType("SplitVD");

    std::vector<ge::GeTensorDesc> outputDesc;
    for (size_t c = num_split - 1; c >= nodes_num; c--) {
      OpDescUtils::ClearOutputDesc(SplitDBaseDesc, c);
    }

    ge::NodePtr splitd_base_node = graph.AddNode(SplitDBaseDesc);
    newNodes.push_back(splitd_base_node);
    ge::AttrUtils::SetListInt(splitd_base_node->GetOpDesc(), "size_splits", size_splits_new);
    ge::AttrUtils::SetInt(splitd_base_node->GetOpDesc(), "split_dim", split_dim);
    ge::AttrUtils::SetInt(splitd_base_node->GetOpDesc(), "num_split", nodes_num);

    ge::GeTensorDesc SplitDInputTensor_1 = SplitDBaseDesc->GetInputDesc(0);
    ge::GeShape SplitDInputShape_1 = SplitDInputTensor_1.GetShape();

    for (size_t h = 0; h < nodes_num; h++) {
      ge::GeTensorDesc SplitDOutputTensor_1 = SplitDBaseDesc->GetOutputDesc(h);
      ge::GeShape SplitDOutputShape_1 = SplitDOutputTensor_1.GetShape();
      SplitDOutputShape_1.SetDim(split_dim, size_splits_new[h]);
      SplitDOutputTensor_1.SetShape(SplitDOutputShape_1);
      SplitDOutputTensor_1.SetOriginShape(SplitDOutputShape_1);
      SplitDBaseDesc->UpdateOutputDesc(h, SplitDOutputTensor_1);
      outputDesc.push_back(SplitDOutputTensor_1);
    }
    FUSION_PASS_CHECK(splitd_base_node == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitd_base_node:%s is null, fusion failed.",
                              splitd_base_node->GetName().c_str()),
                      return PARAM_INVALID);

    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                           splitd_base_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                fused_node->GetName().c_str(), (0), splitd_base_node->GetName().c_str(), 0),
        return FAILED);

    for (size_t i = 0; i < nodes_num; i++) {
      if (i < nodes_num - 1) {
        ge::OpDescPtr SplitDDesc = AttrUtils::CopyOpDesc(fusedDesc);
        SplitDDesc->SetName(SplitDDesc->GetName() + "/SplitVD" + to_string(i));
        SplitDDesc->SetType("SplitVD");
        for (size_t c = num_split - 1; c >= 63; c--) {
          OpDescUtils::ClearOutputDesc(SplitDDesc, c);
        }
        ge::NodePtr splitd_node = graph.AddNode(SplitDDesc);
        newNodes.push_back(splitd_node);

        vector<int64_t> size_splits_new2;
        for (size_t n = 0; n < 63; n++) {
          size_splits_new2.push_back(size_splits[63 * i + n]);
        }

        ge::AttrUtils::SetInt(splitd_node->GetOpDesc(), "split_dim", split_dim);
        ge::AttrUtils::SetInt(splitd_node->GetOpDesc(), "num_split", 63);
        ge::AttrUtils::SetListInt(splitd_node->GetOpDesc(), "size_splits", size_splits_new2);

        SplitDDesc->UpdateInputDesc(0, outputDesc[i]);
        for (int64_t h = 0; h < 63; h++) {
          ge::GeTensorDesc SplitDOutputTensor_2 = SplitDDesc->GetOutputDesc(h);
          ge::GeShape SplitDOutputShape_2 = SplitDOutputTensor_2.GetShape();
          SplitDOutputShape_2.SetDim(split_dim, size_splits_new2[h]);
          SplitDOutputTensor_2.SetShape(SplitDOutputShape_2);
          SplitDDesc->UpdateOutputDesc(h, SplitDOutputTensor_2);
        }

        FUSION_PASS_CHECK(
            splitd_node == nullptr,
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "splitd_node:%s is null, fusion failed.", splitd_node->GetName().c_str()),
            return PARAM_INVALID);

        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(splitd_base_node->GetOutDataAnchor(i), splitd_node->GetInDataAnchor(0)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Add edge from fused node:%s's index[%lu] to fusion node:%s's index[%lu] failed.",
                    splitd_base_node->GetName().c_str(), i, splitd_node->GetName().c_str(), i),
            return FAILED);

        for (size_t m = 0; m < 63; m++) {
          for (InDataAnchorPtr inAnchorPtr : fused_node->GetOutDataAnchor(63 * i + m)->GetPeerInDataAnchors()) {
            FUSION_PASS_CHECK(
                SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(63 * i + m), inAnchorPtr),
                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitd_node->GetOutDataAnchor(m), inAnchorPtr),
                              VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
          }
        }
      } else {
        if (last_node_num_split != 1) {
          ge::OpDescPtr LastSplitDDesc = AttrUtils::CopyOpDesc(fusedDesc);
          LastSplitDDesc->SetName(LastSplitDDesc->GetName() + "/SplitVD" + to_string(nodes_num - 1));
          LastSplitDDesc->SetType("SplitVD");
          for (size_t c = num_split - 1; c >= last_node_num_split; c--) {
            OpDescUtils::ClearOutputDesc(LastSplitDDesc, c);
          }
          ge::NodePtr last_splitd_node = graph.AddNode(LastSplitDDesc);
          newNodes.push_back(last_splitd_node);

          vector<int64_t> size_splits_new3;
          for (size_t n = 0; n < last_node_num_split; n++) {
            size_splits_new3.push_back(size_splits[63 * i + n]);
          }

          ge::AttrUtils::SetInt(last_splitd_node->GetOpDesc(), "split_dim", split_dim);
          ge::AttrUtils::SetInt(last_splitd_node->GetOpDesc(), "num_split", last_node_num_split);
          ge::AttrUtils::SetListInt(last_splitd_node->GetOpDesc(), "size_splits", size_splits_new3);

          LastSplitDDesc->UpdateInputDesc(0, outputDesc[nodes_num - 1]);
          for (size_t h = 0; h < last_node_num_split; h++) {
            ge::GeTensorDesc SplitDOutputTensor_3 = LastSplitDDesc->GetOutputDesc(h);
            ge::GeShape SplitDOutputShape_3 = SplitDOutputTensor_3.GetShape();
            SplitDOutputShape_3.SetDim(split_dim, size_splits_new3[h]);
            SplitDOutputTensor_3.SetShape(SplitDOutputShape_3);
            LastSplitDDesc->UpdateOutputDesc(h, SplitDOutputTensor_3);
          }

          FUSION_PASS_CHECK(last_splitd_node == nullptr,
                            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "last_splitd_node:%s is null, fusion failed.",
                                    last_splitd_node->GetName().c_str()),
                            return PARAM_INVALID);

          FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitd_base_node->GetOutDataAnchor(i),
                                                               last_splitd_node->GetInDataAnchor(0)),
                            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                    "Add edge from fused node:%s's index[%lu] to fusion node:%s's index[%lu] failed.",
                                    splitd_base_node->GetName().c_str(), i, last_splitd_node->GetName().c_str(), i),
                            return FAILED);

          for (size_t m = 0; m < last_node_num_split; m++) {
            for (InDataAnchorPtr inAnchorPtr : fused_node->GetOutDataAnchor(63 * i + m)->GetPeerInDataAnchors()) {
              FUSION_PASS_CHECK(
                  SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(63 * i + m), inAnchorPtr),
                  VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
              FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(last_splitd_node->GetOutDataAnchor(m), inAnchorPtr),
                                VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
            }
          }
        } else {
          for (InDataAnchorPtr inAnchorPtr : fused_node->GetOutDataAnchor(63 * i)->GetPeerInDataAnchors()) {
            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(63 * i), inAnchorPtr),
                              VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitd_base_node->GetOutDataAnchor(i), inAnchorPtr),
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

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Split --> SplitD or SplitVD fusion SUCCESSS!!!!!");
  return SUCCESS;
}

REGISTER_PASS("ZSplitFusionPass", BUILT_IN_GRAPH_PASS, SplitFusionPass);
}  // namespace fe
