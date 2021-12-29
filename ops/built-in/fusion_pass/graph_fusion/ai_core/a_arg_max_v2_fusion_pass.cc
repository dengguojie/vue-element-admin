/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file a_arg_max_v2_fusion_pass.cpp
 * \brief a_arg_max_v2_fusion_pass (argmaxv2+argmaxv2-->equal)
 */
#include "a_arg_max_v2_fusion_pass.h"
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "tbe_fusion_pass_util.h"

using namespace std;
using namespace ge;
namespace fe {
static const string PATTERN_ARGMAX1 = "ArgMax1";
static const string PATTERN_ARGMAX2 = "ArgMax2";
static const string PATTERN_EQUAL = "Equal";
static const string DTYPE = "dtype";
static const string ARGMAXV2 = "ArgMaxV2";
static const string EQUAL = "Equal";
static const int64_t COMP = 2147483648;

vector<FusionPattern*> AArgMaxV2FusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("AArgMaxV2FusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_ARGMAX1, {ARGMAXV2})
      .AddOpDesc(PATTERN_ARGMAX2, {ARGMAXV2})
      .AddOpDesc(PATTERN_EQUAL, {EQUAL})
      .SetInputs(PATTERN_EQUAL, {PATTERN_ARGMAX1, PATTERN_ARGMAX2})
      .SetOutput(PATTERN_EQUAL);
  patterns.push_back(pattern);
  return patterns;
}

Status AArgMaxV2FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr argmaxv2_nodea = GetNodeFromMapping(PATTERN_ARGMAX1, mapping);
  ge::NodePtr argmaxv2_nodeb = GetNodeFromMapping(PATTERN_ARGMAX2, mapping);
  ge::NodePtr equal_node = GetNodeFromMapping(PATTERN_EQUAL, mapping);
  // get opdesc
  ge::OpDescPtr in_op_a_ptr = argmaxv2_nodea->GetOpDesc();
  ge::OpDescPtr in_op_b_ptr = argmaxv2_nodeb->GetOpDesc();
  ge::OpDescPtr equal_ptr = equal_node->GetOpDesc();
  FUSION_PASS_CHECK(argmaxv2_nodea == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Argmaxv2 nodea is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(argmaxv2_nodeb == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Argmaxv2 nodeb is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(equal_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "equal node is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(in_op_a_ptr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ArgMaxv2a OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(in_op_b_ptr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ArgMaxv2b OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  DataType outputa_type = in_op_a_ptr->GetOutputDesc(0).GetDataType();
  FUSION_PASS_CHECK(outputa_type != ge::DT_INT64,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), " when %s outputdata's dtype is not int64 does not need fusion.",
                            in_op_a_ptr->GetName().c_str()),
                    return NOT_CHANGED);
  DataType outputb_type = in_op_b_ptr->GetOutputDesc(0).GetDataType();
  FUSION_PASS_CHECK(outputb_type != ge::DT_INT64,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), " when %s outputdata's dtype is not int64 does not need fusion.",
                            in_op_b_ptr->GetName().c_str()),
                    return NOT_CHANGED);

  // get operator
  Operator op_argmaxv2a = ge::OpDescUtils::CreateOperatorFromNode(argmaxv2_nodea);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node %s begin fusion.", in_op_a_ptr->GetName().c_str());
  Operator op_argmaxv2b = ge::OpDescUtils::CreateOperatorFromNode(argmaxv2_nodeb);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node %s begin fusion.", in_op_b_ptr->GetName().c_str());
  // get inputs tensordesc
  ge::GeTensorDesc x_input_desca = in_op_a_ptr->GetInputDesc(0);
  ge::GeTensorDesc x_input_descb = in_op_b_ptr->GetInputDesc(0);
  auto x_input_shapea = x_input_desca.GetShape().GetDims();
  auto x_input_shapeb = x_input_descb.GetShape().GetDims();
  auto x_dima = x_input_shapea.size();
  auto x_dimb = x_input_shapeb.size();
  std::vector<std::pair<int64_t, int64_t>> range;
  int64_t aixs_const_vala = 0;
  int64_t aixs_const_valb = 0;
  Tensor const_dataa;
  Tensor const_datab;
  if (op_argmaxv2a.GetInputConstData("dimension", const_dataa) == GRAPH_SUCCESS &&
      op_argmaxv2b.GetInputConstData("dimension", const_datab) == GRAPH_SUCCESS) {
    auto aixs_tensor_desca = op_argmaxv2a.GetInputDescByName("dimension");
    auto aixs_tensor_descb = op_argmaxv2b.GetInputDescByName("dimension");
    uint8_t* const_dataa_ptr = const_dataa.GetData();
    uint8_t* const_datab_ptr = const_datab.GetData();
    DataType input_axis_dtypea = aixs_tensor_desca.GetDataType();
    DataType input_axis_dtypeb = aixs_tensor_descb.GetDataType();
    bool flag = true;
    FUSION_PASS_CHECK(const_dataa_ptr == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Get dimension const data failed."),
                      return NOT_CHANGED);
    FUSION_PASS_CHECK(const_datab_ptr == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Get dimension const data failed."),
                      return NOT_CHANGED);
    if (input_axis_dtypea == DT_INT32) {
      aixs_const_vala = static_cast<int64_t>(*(reinterpret_cast<int32_t*>(const_dataa_ptr)));
    } else if (input_axis_dtypea == DT_INT64) {
      aixs_const_vala = *(reinterpret_cast<int64_t*>(const_dataa_ptr));
    } else {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "aixs only support int32 and int64 in AICORE");
      return NOT_CHANGED;
    }
    if (input_axis_dtypeb == DT_INT32) {
      aixs_const_valb = static_cast<int64_t>(*(reinterpret_cast<int32_t*>(const_datab_ptr)));
    } else if (input_axis_dtypeb == DT_INT64) {
      aixs_const_valb = *(reinterpret_cast<int64_t*>(const_datab_ptr));
    } else {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "aixs only support int32 and int64 in AICORE");
      return NOT_CHANGED;
    }
    if (aixs_const_vala < 0) {
      aixs_const_vala += x_dima;
    }
    if (aixs_const_valb < 0) {
      aixs_const_valb += x_dimb;
    }
    FUSION_PASS_CHECK((aixs_const_vala > static_cast<int64_t>(x_dima) - 1 || aixs_const_vala < 0),
                       OP_LOGW(FUSED_OP_TYPE.c_str(),
                               "Node:%s's dimension is invalid.shape dim is [%ld], axis is [%ld]",
                               x_dima, aixs_const_vala), return NOT_CHANGED);
    FUSION_PASS_CHECK((aixs_const_valb > static_cast<int64_t>(x_dimb) - 1 || aixs_const_valb < 0),
                       OP_LOGW(FUSED_OP_TYPE.c_str(),
                               "Node:%s's dimension is invalid.shape dim is [%ld], axis is [%ld]",
                               x_dimb, aixs_const_valb), return NOT_CHANGED);
    if (x_input_shapea.at(aixs_const_vala) != -1 || x_input_shapeb.at(aixs_const_valb) != -1) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "input[0]'s shape is static shape.");
      if (x_input_shapea.at(aixs_const_vala) < COMP || x_input_shapeb.at(aixs_const_valb) < COMP) {
        flag = false;
      }
    } else {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "input[0]'s shape is dynamic shape.");
      FUSION_PASS_CHECK(
          x_input_desca.GetShapeRange(range) != GRAPH_SUCCESS,
          OP_LOGW(FUSED_OP_TYPE.c_str(), "Node:%s get shape range failed.", in_op_a_ptr->GetName().c_str()),
          return NOT_CHANGED);
      FUSION_PASS_CHECK(
          x_input_descb.GetShapeRange(range) != GRAPH_SUCCESS,
          OP_LOGW(FUSED_OP_TYPE.c_str(), "Node:%s get shape range failed.", in_op_b_ptr->GetName().c_str()),
          return NOT_CHANGED);
      auto range_size = range.size();
      FUSION_PASS_CHECK(
          static_cast<int64_t>(range_size) <= aixs_const_vala,
          OP_LOGW(FUSED_OP_TYPE.c_str(), "Node:%s shape range is invalid.", in_op_a_ptr->GetName().c_str()),
          return NOT_CHANGED);
      FUSION_PASS_CHECK(
          static_cast<int64_t>(range_size) <= aixs_const_valb,
          OP_LOGW(FUSED_OP_TYPE.c_str(), "Node:%s shape range is invalid.", in_op_b_ptr->GetName().c_str()),
          return NOT_CHANGED);
      if (range[aixs_const_vala].second != -1 && range[aixs_const_vala].second < COMP) {
        flag = false;
      }
      if (range[aixs_const_valb].second != -1 && range[aixs_const_valb].second < COMP) {
        flag = false;
      }
    }
    FUSION_PASS_CHECK(flag,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node:%s does not need fusion.", in_op_a_ptr->GetName().c_str()),
                      return NOT_CHANGED);
  } else {
    bool flag = false;
    for (auto iter = x_input_shapea.begin(); iter != x_input_shapea.end(); ++iter) {
      if (*iter > COMP) {
        flag = true;
      }
    }
    for (auto iter = x_input_shapeb.begin(); iter != x_input_shapeb.end(); ++iter) {
      if (*iter > COMP) {
        flag = true;
      }
    }
    FUSION_PASS_CHECK(flag,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node:%s does not need fusion.", in_op_a_ptr->GetName().c_str()),
                      return NOT_CHANGED);
  }

  if ((argmaxv2_nodea->GetOutDataAnchor(0) == nullptr) || (argmaxv2_nodeb->GetOutDataAnchor(0) == nullptr)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "argmaxv2a cannot get node from equal.");
    return NOT_CHANGED;
  }

  if ((argmaxv2_nodea->GetOutDataAnchor(0)->GetPeerInDataAnchors().empty()) ||
      (argmaxv2_nodeb->GetOutDataAnchor(0)->GetPeerInDataAnchors().empty())) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "argmaxv2 peerInDataAnchors is empty.");
    return NOT_CHANGED;
  }

  ge::OpDescPtr equal_op_desc_ptr = equal_node->GetOpDesc();
  DataType types1 = ge::DT_INT32;
  op_argmaxv2a.SetAttr(DTYPE, types1);
  op_argmaxv2b.SetAttr(DTYPE, types1);
  equal_op_desc_ptr->MutableInputDesc(0)->SetDataType(ge::DT_INT32);
  equal_op_desc_ptr->MutableInputDesc(1)->SetDataType(ge::DT_INT32);
  equal_op_desc_ptr->MutableInputDesc(0)->SetOriginDataType(ge::DT_INT32);
  equal_op_desc_ptr->MutableInputDesc(1)->SetOriginDataType(ge::DT_INT32);
  in_op_a_ptr->MutableOutputDesc(0)->SetDataType(ge::DT_INT32);
  in_op_b_ptr->MutableOutputDesc(0)->SetDataType(ge::DT_INT32);
  in_op_a_ptr->MutableOutputDesc(0)->SetOriginDataType(ge::DT_INT32);
  in_op_b_ptr->MutableOutputDesc(0)->SetOriginDataType(ge::DT_INT32);

  return SUCCESS;
}
REGISTER_PASS("AArgMaxV2FusionPass", BUILT_IN_GRAPH_PASS, AArgMaxV2FusionPass);
}  // namespace fe
