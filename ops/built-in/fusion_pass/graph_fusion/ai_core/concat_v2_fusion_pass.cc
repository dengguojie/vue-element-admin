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
 * \file concat_v2_fusion_pass.cpp
 * \brief ConcatExt2 fusion pass(ConcatExt2 --> ConcatExt2)
 */
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "concat_v2_fusion_pass.h"
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

static const char* FUSED_NODE = "ConcatV2";
static const std::string PATTERN_FUSEDNODE = "FusedNodeConcatV2";

void ConcatExt2FusionPass::UpdateInputName(ge::OpDescPtr& input_desc_ptr) {
  auto input_count = input_desc_ptr->GetAllInputsSize();
  map<string, uint32_t> name_index_map;
  string name = "x";
  for (size_t i = 0; i < input_count; ++i) {
    name_index_map.insert({name + std::to_string(i), i});
  }
  input_desc_ptr->UpdateInputName(name_index_map);
}

vector<FusionPattern*> ConcatExt2FusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConcatExt2FusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//including newly added nodes and fused but not deleted nodes
Status ConcatExt2FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  std::string fusionOpType = "ConcatV2D";
  std::vector<PassAttrInfo> concatv2AttrInfo;
  ge::NodePtr fused_node = nullptr;
  ge::NodePtr fused_node1 = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  int32_t inputsize = fused_node1->GetAllInDataAnchors().size();
  PassAttrInfo axis = {inputsize - 1, "concat_dim", "SetInt"};
  concatv2AttrInfo.push_back(axis);
  FUSION_PASS_CHECK(fused_node1 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed"),
                    return PARAM_INVALID);

  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, fused_node1, fusionOpType, concatv2AttrInfo, fused_node);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Concatv2 has input which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }

  ClearOpInferDepends(fused_node1);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Concatv2 fusion SUCCESSS!!!!!");

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
          OpDescUtils::ClearInputDesc(fusedDesc, whereI[i] - i);
          ge::NodeUtils::ClearInDataAnchor(fused_node,fused_node->GetInDataAnchor(whereI[i] - i));
      }
  }
  num_N_new = num_N - num_N_del;
  ge::AttrUtils::SetInt(fusedDesc, "N", num_N_new);
  // A maximum of 63 tensors are supported in mini mode.
  int64_t inputs_num = fusedDesc->GetInputsSize();
  OP_LOGI(FUSED_OP_TYPE.c_str(), "All of concat_v2 nums is:%d", inputs_num);
  int64_t NeedTangent = 63;
  FUSION_PASS_CHECK(inputs_num <= NeedTangent,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The amount of input of ConcatV2D node is less than 63."),
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

    ge::OpDescPtr ConcatExt2BaseDesc = AttrUtils::CopyOpDesc(fusedDesc);
    ConcatExt2BaseDesc->SetName(ConcatExt2BaseDesc->GetName() + "/ConcatV2D" + "Base_node");
    ConcatExt2BaseDesc->SetType("ConcatV2D");
    int64_t concat_dim;
    ge::AttrUtils::GetInt(fusedDesc, "concat_dim", concat_dim);
    ge::AttrUtils::SetInt(ConcatExt2BaseDesc, "concat_dim", concat_dim);

    for (int64_t c = inputs_num - 1; c >= nodes_num; c--) {
      OpDescUtils::ClearInputDesc(ConcatExt2BaseDesc, c);
    }

    ge::NodePtr concatext2_base_node = graph.AddNode(ConcatExt2BaseDesc);
    FUSION_PASS_CHECK(concatext2_base_node == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "concatext2_base_node:%s is null, fusion failed.",
                              concatext2_base_node->GetName().c_str()),
                      return PARAM_INVALID);
    fusionNodes.push_back(concatext2_base_node);
    ge::AttrUtils::SetInt(concatext2_base_node->GetOpDesc(), "N", nodes_num);
    for (InDataAnchorPtr inAnchorPtr : fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(0), inAnchorPtr),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatext2_base_node->GetOutDataAnchor(0), inAnchorPtr),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
    }
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Split nodes_num is:%d", nodes_num);
    for (int64_t i = 0; i < nodes_num; i++) {
      if (i < nodes_num - 1) {
        int64_t inputs_num = fusedDesc->GetInputsSize();
        ge::OpDescPtr ConcatExt2Desc = AttrUtils::CopyOpDesc(fusedDesc);
        ConcatExt2Desc->SetName(ConcatExt2Desc->GetName() + "/ConcatV2D" + to_string(i));
        ConcatExt2Desc->SetType("ConcatV2D");
        ge::AttrUtils::GetInt(fusedDesc, "concat_dim", concat_dim);
        ge::AttrUtils::SetInt(ConcatExt2Desc, "concat_dim", concat_dim);

        if (i == 0) {
          for (int64_t a = inputs_num - 1; a >= NeedTangent; a--) {
            RemoveInputDesc(ConcatExt2Desc, a);
          }
        } else {
          for (int64_t a = i * NeedTangent - 1; a >= 0; a--) {
            RemoveInputDesc(ConcatExt2Desc, a);
          }
          for (int64_t a = inputs_num - (NeedTangent * i + 1); a >= NeedTangent; a--) {
            RemoveInputDesc(ConcatExt2Desc, a);
          }
        }
        ge::NodePtr concatext2_node = graph.AddNode(ConcatExt2Desc);
        fusionNodes.push_back(concatext2_node);
        ge::AttrUtils::SetInt(concatext2_node->GetOpDesc(), "N", NeedTangent);
        // infershape begin
        int64_t size = 0;
        int64_t num_concat = 63;
        int64_t concat_dim = 0;
        ge::AttrUtils::GetInt(concatext2_node->GetOpDesc(), "concat_dim", concat_dim);
        ge::GeTensorDesc ConcatExt2InputTensor_0 = ConcatExt2Desc->GetInputDesc(0);
        ge::GeShape ConcatExt2InputShape_0 = ConcatExt2InputTensor_0.GetShape();
        int64_t dimnum = ConcatExt2InputShape_0.GetDimNum();
        auto axis = concat_dim;
        if (axis < 0) {
          axis += (dimnum);
        }

        for (int64_t n = 0; n < num_concat; n++) {
          if (zero_num > 0) {
            ge::GeTensorDesc ConcatExt2InputTensor_1 = ConcatExt2Desc->GetInputDesc(63 * i + n);
            ge::GeShape ConcatExt2InputShape_1 = ConcatExt2InputTensor_1.GetShape();
            int64_t dim_axis_value = ConcatExt2InputShape_1.GetDim(axis);
            if (PatternFusionUtil::IsUnknownShape(size)) {
              continue;
            }

            if (PatternFusionUtil::IsUnknownShape(dim_axis_value)) {
              size = dim_axis_value;
            } else {
              size += dim_axis_value;
            }
          } else {
            Operator op1 = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
            ge::TensorDesc ConcatExt2InputTensor_1 = op1.GetDynamicInputDesc("x", (inputs_num - 1) - (63 * i + n));
            ge::Shape ConcatExt2InputShape_1 = ConcatExt2InputTensor_1.GetShape();
            int64_t dim_axis_value = ConcatExt2InputShape_1.GetDim(axis);
            if (PatternFusionUtil::IsUnknownShape(size)) {
              continue;
            }

            if (PatternFusionUtil::IsUnknownShape(dim_axis_value)) {
              size = dim_axis_value;
            } else {
              size += dim_axis_value;
            }
          }
        }
        ge::GeTensorDesc ConcatExt2OutputTensor_1 = ConcatExt2Desc->GetOutputDesc(0);
        ge::GeShape ConcatExt2OutputShape_1 = ConcatExt2OutputTensor_1.GetShape();
        ConcatExt2OutputShape_1.SetDim(axis, size);
        ConcatExt2OutputTensor_1.SetShape(ConcatExt2OutputShape_1);
        ConcatExt2OutputTensor_1.SetOriginShape(ConcatExt2OutputShape_1);
        UpdateInputName(ConcatExt2Desc);
        ConcatExt2Desc->UpdateOutputDesc(0, ConcatExt2OutputTensor_1);
        ConcatExt2BaseDesc->UpdateInputDesc(i, ConcatExt2OutputTensor_1);
        // infershape end
        FUSION_PASS_CHECK(
            concatext2_node == nullptr,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "concatext2_node:%s is null, fusion failed.", concatext2_node->GetName().c_str()),
            return PARAM_INVALID);

        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatext2_node->GetOutDataAnchor(0),
                                                             concatext2_base_node->GetInDataAnchor(i)),
                          OP_LOGE(FUSED_OP_TYPE.c_str(),
                                  "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                                  concatext2_base_node->GetName().c_str(), i, concatext2_node->GetName().c_str(), i),
                          return FAILED);

        for (int64_t m = 0; m < 63; m++) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(m + i * 63)->GetPeerOutAnchor(),
                                                 concatext2_node->GetInDataAnchor(m)),
              OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                      fused_node->GetName().c_str(), (m + i * 63), concatext2_node->GetName().c_str(), m),
              return FAILED);
        }
      } else {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Begin split last concat node");
        int64_t inputs_num = fusedDesc->GetInputsSize();
        ge::OpDescPtr LastConcatExt2Desc = AttrUtils::CopyOpDesc(fusedDesc);
        LastConcatExt2Desc->SetName(LastConcatExt2Desc->GetName() + "/ConcatV2D" + to_string(nodes_num - 1));
        LastConcatExt2Desc->SetType("ConcatV2D");

        for (int64_t b = inputs_num - last_node_inputs_num - 1; b >= 0; b--) {
          RemoveInputDesc(LastConcatExt2Desc, b);
        }
        ge::NodePtr last_concatext2_node = graph.AddNode(LastConcatExt2Desc);
        FUSION_PASS_CHECK(last_concatext2_node == nullptr,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                  last_concatext2_node->GetName().c_str()),
                          return PARAM_INVALID);
        fusionNodes.push_back(last_concatext2_node);
        ge::AttrUtils::SetInt(last_concatext2_node->GetOpDesc(), "N", last_node_inputs_num);
        // the last_node infershape begin
        int32_t size = 0;
        int64_t num_concat = last_node_inputs_num;
        int64_t concat_dim;
        ge::AttrUtils::GetInt(last_concatext2_node->GetOpDesc(), "concat_dim", concat_dim);
        ge::GeTensorDesc ConcatExt2InputTensor_3 = LastConcatExt2Desc->GetInputDesc(0);
        ge::GeShape ConcatExt2InputShape_3 = ConcatExt2InputTensor_3.GetShape();
        int64_t dimnum = ConcatExt2InputShape_3.GetDimNum();
        auto axis = concat_dim;
        if (axis < 0) {
          axis += (dimnum);
        }
        for (int32_t n = 0; n < num_concat; n++) {
          if (zero_num > 0) {
            ge::GeTensorDesc ConcatExt2InputTensor_2 = LastConcatExt2Desc->GetInputDesc(n);
            ge::GeShape ConcatExt2InputShape_2 = ConcatExt2InputTensor_2.GetShape();
            int64_t dim_axis_value = ConcatExt2InputShape_2.GetDim(axis);
            if (PatternFusionUtil::IsUnknownShape(size)) {
              continue;
            }

            if (PatternFusionUtil::IsUnknownShape(dim_axis_value)) {
              size = dim_axis_value;
            } else {
              size += dim_axis_value;
            }
          } else {
            Operator op2 = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
            ge::TensorDesc ConcatExt2InputTensor_2 = op2.GetDynamicInputDesc("x", (inputs_num - 1) - (n + 63 * i));
            ge::Shape ConcatExt2InputShape_2 = ConcatExt2InputTensor_2.GetShape();
            int64_t dim_axis_value = ConcatExt2InputShape_2.GetDim(axis);
            if (PatternFusionUtil::IsUnknownShape(size)) {
              continue;
            }

            if (PatternFusionUtil::IsUnknownShape(dim_axis_value)) {
              size = dim_axis_value;
            } else {
              size += dim_axis_value;
            }
          }
        }
        ge::GeTensorDesc ConcatExt2OutputTensor_2 = LastConcatExt2Desc->GetOutputDesc(0);
        ge::GeShape ConcatExt2OutputShape_2 = ConcatExt2OutputTensor_2.GetShape();
        ConcatExt2OutputShape_2.SetDim(axis, size);
        ConcatExt2OutputTensor_2.SetShape(ConcatExt2OutputShape_2);
        ConcatExt2OutputTensor_2.SetOriginShape(ConcatExt2OutputShape_2);
        UpdateInputName(LastConcatExt2Desc);
        LastConcatExt2Desc->UpdateOutputDesc(0, ConcatExt2OutputTensor_2);
        ConcatExt2BaseDesc->UpdateInputDesc(i, ConcatExt2OutputTensor_2);
        // the last_node infershape end
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(last_concatext2_node->GetOutDataAnchor(0),
                                                             concatext2_base_node->GetInDataAnchor(i)),
                          OP_LOGE(FUSED_OP_TYPE.c_str(),
                                  "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                                  concatext2_base_node->GetName().c_str(), i, last_concatext2_node->GetName().c_str(), i),
                          return FAILED);

        for (int64_t n = 0; n < last_node_inputs_num; n++) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(n + i * 63)->GetPeerOutAnchor(),
                                                 last_concatext2_node->GetInDataAnchor(n)),
              OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                      fused_node->GetName().c_str(), (n + i * 63), last_concatext2_node->GetName().c_str(), n),
              return FAILED);
        }
      }
    }
    UpdateInputName(ConcatExt2BaseDesc);
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

REGISTER_PASS("ZConcatExt2FusionPass", BUILT_IN_GRAPH_PASS, ConcatExt2FusionPass);
}  // namespace fe

