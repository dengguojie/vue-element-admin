/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file concat_fusion_pass.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "concat_fusion_pass.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "Concat";
static const std::string PATTERN_FUSEDNODE = "FusedNodeConcat";

void ConcatFusionPass::UpdateInputName(ge::OpDescPtr& input_desc_ptr) {
  auto input_count = input_desc_ptr->GetAllInputsSize();
  map<string, uint32_t> name_index_map;
  string name = "x";
  for (size_t i = 0; i < input_count; ++i) {
    name_index_map.insert({name + std::to_string(i), i});
  }
  input_desc_ptr->UpdateInputName(name_index_map);
}

vector<FusionPattern*> ConcatFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConcatFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

Status ConcatFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  std::string fusionOpType = "ConcatD";
  std::vector<PassAttrInfo> concatAttrInfo;
  PassAttrInfo concat_dim = {0, "concat_dim", "SetInt"};
  concatAttrInfo.push_back(concat_dim);
  ge::NodePtr fused_node = nullptr;
  ge::NodePtr fused_node1 = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);

  FUSION_PASS_CHECK(fused_node1 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed"),
                    return PARAM_INVALID);

  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, fused_node1, fusionOpType, concatAttrInfo, fused_node);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Concat has input which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }

  ClearOpInferDepends(fused_node1);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Concat-->ConcatD fusion SUCCESSS!!!!!");

  ge::OpDescPtr fusedDesc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fused_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  int64_t num_N;
  int64_t num_N_del = 0;
  int64_t num_N_new;
  vector<int64_t> whereI;
  ge::AttrUtils::GetInt(fusedDesc, "N", num_N);
  for (int i=0;i<num_N;i++) {
     int64_t Repeatnum = 0;
     ge::GeTensorDesc selectInputDesc = fused_node->GetOpDesc()->GetInputDesc(i);
     vector<int64_t> selectInputShape = selectInputDesc.GetShape().GetDims();
     for (int j=0; j<selectInputShape.size(); j++){
         if (selectInputShape[j] == 0){
             Repeatnum +=1;
         }
     }
     if (Repeatnum > 0){
        num_N_del += 1;
        whereI.push_back(i);
     }
  }
  int64_t zero_num = whereI.size();
  if (zero_num > 0){
      for (int i=0; i<whereI.size(); i++){
          FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fused_node->GetInDataAnchor(whereI[i] - i)->GetPeerOutAnchor(),
                                                       fused_node->GetInDataAnchor(whereI[i] - i)) != SUCCESS,

                           OP_LOGE(FUSED_OP_TYPE.c_str(),"Remove edge failed."), return FAILED);
          RemoveInputDesc(fusedDesc, whereI[i] - i);
          ge::NodeUtils::ClearInDataAnchor(fused_node,fused_node->GetInDataAnchor(whereI[i] - i));
      }
  }
  num_N_new = num_N - num_N_del;
  ge::AttrUtils::SetInt(fusedDesc, "N", num_N_new);

  // A maximum of 63 tensors are supported in mini mode.
  int64_t inputs_num = fusedDesc->GetInputsSize();
  int64_t NeedTangent = 63;
  FUSION_PASS_CHECK(inputs_num <= NeedTangent,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The amount of input of ConcatD node is less than 63."),
                    return NOT_CHANGED);

  if (inputs_num > NeedTangent) {
    int64_t nodes_num, nodes_num1;
    nodes_num1 = inputs_num % NeedTangent;
    if (nodes_num1 == 0) {
      nodes_num = inputs_num / NeedTangent;
    } else {
      nodes_num = inputs_num / NeedTangent + 1;
    }
    int64_t last_node_inputs_num = inputs_num - (NeedTangent * (nodes_num - 1));

    ge::OpDescPtr ConcatdBaseDesc = AttrUtils::CopyOpDesc(fusedDesc);
    ConcatdBaseDesc->SetName(ConcatdBaseDesc->GetName() + "/ConcatD" + "Base_node");
    ConcatdBaseDesc->SetType("ConcatD");
    int64_t concat_dim;
    ge::AttrUtils::GetInt(fusedDesc, "concat_dim", concat_dim);
    ge::AttrUtils::SetInt(ConcatdBaseDesc, "concat_dim", concat_dim);

    for (int64_t c = inputs_num - 1; c >= nodes_num; c--) {
      RemoveInputDesc(ConcatdBaseDesc, c);
    }

    ge::NodePtr concatd_base_node = graph.AddNode(ConcatdBaseDesc);
    FUSION_PASS_CHECK(concatd_base_node == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "concatd_base_node:%s is null, fusion failed.",
                              concatd_base_node->GetName().c_str()),
                      return PARAM_INVALID);
    newNodes.push_back(concatd_base_node);
    ge::AttrUtils::SetInt(concatd_base_node->GetOpDesc(), "N", nodes_num);
    for (InDataAnchorPtr inAnchorPtr : fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(0), inAnchorPtr),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatd_base_node->GetOutDataAnchor(0), inAnchorPtr),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
    }

    for (int64_t i = 0; i < nodes_num; i++) {
      if (i < nodes_num - 1) {
        int64_t inputs_num = fusedDesc->GetInputsSize();
        ge::OpDescPtr ConcatdDesc = AttrUtils::CopyOpDesc(fusedDesc);
        ConcatdDesc->SetName(ConcatdDesc->GetName() + "/ConcatD" + to_string(i));
        ConcatdDesc->SetType("ConcatD");
        ge::AttrUtils::GetInt(fusedDesc, "concat_dim", concat_dim);
        ge::AttrUtils::SetInt(ConcatdDesc, "concat_dim", concat_dim);

        if (i == 0) {
          for (int64_t a = inputs_num - 1; a >= NeedTangent; a--) {
            RemoveInputDesc(ConcatdDesc, a);
          }
        } else {
          for (int64_t a = i * NeedTangent - 1; a >= 0; a--) {
            RemoveInputDesc(ConcatdDesc, a);
          }
          for (int64_t a = inputs_num - (NeedTangent * i + 1); a >= NeedTangent; a--) {
            RemoveInputDesc(ConcatdDesc, a);
          }
        }
        ge::NodePtr concatd_node = graph.AddNode(ConcatdDesc);
        newNodes.push_back(concatd_node);
        ge::AttrUtils::SetInt(concatd_node->GetOpDesc(), "N", NeedTangent);
        // infershape begin
        int64_t size = 0;
        int64_t num_concat = 63;
        int64_t concat_dim = 0;
        ge::AttrUtils::GetInt(concatd_node->GetOpDesc(), "concat_dim", concat_dim);
        ge::GeTensorDesc ConcatDInputTensor_0 = ConcatdDesc->GetInputDesc(0);
        ge::GeShape ConcatDInputShape_0 = ConcatDInputTensor_0.GetShape();
        int64_t dimnum = ConcatDInputShape_0.GetDimNum();
        auto axis = concat_dim;
        if (axis < 0) {
          axis += (dimnum);
        }
        for (int64_t n = 0; n < num_concat; n++) {
          //            ge::GeTensorDesc ConcatDInputTensor_1 = ConcatdDesc->GetDynamicInputDesc("x",i);
          Operator op1 = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
          ge::TensorDesc ConcatDInputTensor_1 = op1.GetDynamicInputDesc("x", 63 * i + n);
          ge::Shape ConcatDInputShape_1 = ConcatDInputTensor_1.GetShape();
          int64_t dim_axis_value = ConcatDInputShape_1.GetDim(axis);
          if (PatternFusionUtil::IsUnknownShape(size)) {
            continue;
          }

          if (PatternFusionUtil::IsUnknownShape(dim_axis_value)) {
            size = dim_axis_value;
          } else {
            size += dim_axis_value;
          }
        }
        ge::GeTensorDesc ConcatDOutputTensor_1 = ConcatdDesc->GetOutputDesc(0);
        ge::GeShape ConcatDOutputShape_1 = ConcatDOutputTensor_1.GetShape();
        ConcatDOutputShape_1.SetDim(axis, size);
        ConcatDOutputTensor_1.SetShape(ConcatDOutputShape_1);
        ConcatDOutputTensor_1.SetOriginShape(ConcatDOutputShape_1);
        ConcatdDesc->UpdateOutputDesc(0, ConcatDOutputTensor_1);
        UpdateInputName(ConcatdDesc);
        ConcatdBaseDesc->UpdateInputDesc(i, ConcatDOutputTensor_1);
        // infershape end
        FUSION_PASS_CHECK(
            concatd_node == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "concatd_node:%s is null, fusion failed.", concatd_node->GetName().c_str()),
            return PARAM_INVALID);

        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatd_node->GetOutDataAnchor(0),
                                                             concatd_base_node->GetInDataAnchor(i)),
                          OP_LOGE(FUSED_OP_TYPE.c_str(),
                                  "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                                  concatd_base_node->GetName().c_str(), i, concatd_node->GetName().c_str(), i),
                          return FAILED);

        for (int64_t m = 0; m < 63; m++) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(m + i * 63)->GetPeerOutAnchor(),
                                                 concatd_node->GetInDataAnchor(m)),
              OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                      fused_node->GetName().c_str(), (m + i * 63), concatd_node->GetName().c_str(), m),
              return FAILED);
        }
      } else {
        int64_t inputs_num = fusedDesc->GetInputsSize();
        ge::OpDescPtr LastConcatDDesc = AttrUtils::CopyOpDesc(fusedDesc);
        LastConcatDDesc->SetName(LastConcatDDesc->GetName() + "/ConcatD" + to_string(nodes_num - 1));
        LastConcatDDesc->SetType("ConcatD");

        for (int64_t b = inputs_num - last_node_inputs_num - 1; b >= 0; b--) {
          RemoveInputDesc(LastConcatDDesc, b);
        }
        ge::NodePtr last_concatd_node = graph.AddNode(LastConcatDDesc);
        FUSION_PASS_CHECK(last_concatd_node == nullptr,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                  last_concatd_node->GetName().c_str()),
                          return PARAM_INVALID);
        newNodes.push_back(last_concatd_node);
        ge::AttrUtils::SetInt(last_concatd_node->GetOpDesc(), "N", last_node_inputs_num);
        // the last_node infershape begin
        int32_t size = 0;
        int64_t num_concat = last_node_inputs_num;
        int64_t concat_dim;
        ge::AttrUtils::GetInt(last_concatd_node->GetOpDesc(), "concat_dim", concat_dim);
        ge::GeTensorDesc ConcatDInputTensor_3 = LastConcatDDesc->GetInputDesc(0);
        ge::GeShape ConcatDInputShape_3 = ConcatDInputTensor_3.GetShape();
        int64_t dimnum = ConcatDInputShape_3.GetDimNum();
        auto axis = concat_dim;
        if (axis < 0) {
          axis += (dimnum);
        }
        for (int32_t n = 0; n < num_concat; n++) {
          //            ge::GeTensorDesc ConcatDInputTensor_2 = LastConcatDDesc->GetDynamicInputDesc("x",i);
          Operator op2 = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
          ge::TensorDesc ConcatDInputTensor_2 = op2.GetDynamicInputDesc("x", n + 63 * i);
          ge::Shape ConcatDInputShape_2 = ConcatDInputTensor_2.GetShape();
          int64_t dim_axis_value = ConcatDInputShape_2.GetDim(axis);
          if (PatternFusionUtil::IsUnknownShape(size)) {
            continue;
          }

          if (PatternFusionUtil::IsUnknownShape(dim_axis_value)) {
            size = dim_axis_value;
          } else {
            size += dim_axis_value;
          }
        }
        ge::GeTensorDesc ConcatDOutputTensor_2 = LastConcatDDesc->GetOutputDesc(0);
        ge::GeShape ConcatDOutputShape_2 = ConcatDOutputTensor_2.GetShape();
        ConcatDOutputShape_2.SetDim(axis, size);
        ConcatDOutputTensor_2.SetShape(ConcatDOutputShape_2);
        ConcatDOutputTensor_2.SetOriginShape(ConcatDOutputShape_2);
        LastConcatDDesc->UpdateOutputDesc(0, ConcatDOutputTensor_2);
        UpdateInputName(LastConcatDDesc);
        ConcatdBaseDesc->UpdateInputDesc(i, ConcatDOutputTensor_2);
        // the last_node infershape end
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(last_concatd_node->GetOutDataAnchor(0),
                                                             concatd_base_node->GetInDataAnchor(i)),
                          OP_LOGE(FUSED_OP_TYPE.c_str(),
                                  "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                                  concatd_base_node->GetName().c_str(), i, last_concatd_node->GetName().c_str(), i),
                          return FAILED);

        for (int64_t n = 0; n < last_node_inputs_num; n++) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(n + i * 63)->GetPeerOutAnchor(),
                                                 last_concatd_node->GetInDataAnchor(n)),
              OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                      fused_node->GetName().c_str(), (n + i * 63), last_concatd_node->GetName().c_str(), n),
              return FAILED);
        }
      }
    }
    UpdateInputName(ConcatdBaseDesc);
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
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove Node [%s] failed", fused_node->GetName().c_str()),
                    return FAILED);

  return SUCCESS;
}

REGISTER_PASS("ZConcatFusionPass", BUILT_IN_GRAPH_PASS, ConcatFusionPass);
}  // namespace fe
