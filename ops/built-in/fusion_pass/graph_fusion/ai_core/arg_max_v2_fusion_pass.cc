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
 * \file arg_max_v2_fusion_pass.cpp
 * \brief arg_max_v2_fusion_pass (argmaxv2-->cast)
 */
#include "arg_max_v2_fusion_pass.h"
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
static const string PATTERN_ARGMAXV2 = "ArgMaxV2";
static const char* DTYPE = "dtype";
static const string ARGMAXV2 = "ArgMaxV2";
static const int64_t COMP = 2147483648;

vector<FusionPattern*> ArgMaxV2FusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ArgMaxV2FusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_ARGMAXV2, {ARGMAXV2}).SetOutput(PATTERN_ARGMAXV2);
  patterns.push_back(pattern);
  return patterns;
}

Status ArgMaxV2FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr argmaxv2_node = GetNodeFromMapping(PATTERN_ARGMAXV2, mapping);
  string argmaxv2_node_type = argmaxv2_node->GetType();
  // get opdesc
  ge::OpDescPtr in_op_desc_ptr = argmaxv2_node->GetOpDesc();
  FUSION_PASS_CHECK(argmaxv2_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Argmaxv2 node is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(in_op_desc_ptr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ArgMaxv2 OpDesc is null, fusion failed."), return PARAM_INVALID);
  // get operator
  Operator op_argmaxv2 = ge::OpDescUtils::CreateOperatorFromNode(argmaxv2_node);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node %s begin fusion.", in_op_desc_ptr->GetName().c_str());
  // get inputs tensordesc
  ge::GeTensorDesc x_input_desc = in_op_desc_ptr->GetInputDesc(0);
  auto x_input_shape = x_input_desc.GetShape().GetDims();
  auto x_dim = x_input_shape.size();
  FUSION_PASS_CHECK((x_dim == 1 && x_input_shape[0] == -2),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Node:%s does not need fusion.", in_op_desc_ptr->GetName().c_str()),
                    return NOT_CHANGED);
  std::vector<std::pair<int64_t, int64_t>> range;
  Tensor const_data;
  if (op_argmaxv2.GetInputConstData("dimension", const_data) == GRAPH_SUCCESS) {
    int64_t aixs_const_val = 0;
    auto aixs_tensor_desc = op_argmaxv2.GetInputDescByName("dimension");
    DataType input_axis_dtype = aixs_tensor_desc.GetDataType();
    uint8_t* const_data_ptr = const_data.GetData();
    bool flag = true;
    FUSION_PASS_CHECK(const_data_ptr == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Get dimension const data failed."),
                      return NOT_CHANGED);
    if (input_axis_dtype == DT_INT32) {
      aixs_const_val = static_cast<int64_t>(*(reinterpret_cast<int32_t*>(const_data_ptr)));
    } else if (input_axis_dtype == DT_INT64) {
      aixs_const_val = *(reinterpret_cast<int64_t*>(const_data_ptr));
    } else {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "aixs only support int32 and int64 in AICORE");
      return NOT_CHANGED;
    }
    if (aixs_const_val < 0) {
      aixs_const_val += x_dim;
    }
    FUSION_PASS_CHECK((aixs_const_val > static_cast<int64_t>(x_dim) - 1 || aixs_const_val < 0),
                       VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's dimension is invalid.",
                       in_op_desc_ptr->GetName().c_str()), return PARAM_INVALID);
    if (x_input_shape.at(aixs_const_val) != -1) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "input[0]'s shape is static shape.");
      if (x_input_shape.at(aixs_const_val) < COMP) {
        flag = false;
      }
    } else {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "input[0]'s shape is dynamic shape.");
      FUSION_PASS_CHECK(x_input_desc.GetShapeRange(range) != GRAPH_SUCCESS,
                        OP_LOGW(FUSED_OP_TYPE.c_str(), "Node:%s get shape range failed.",
                        in_op_desc_ptr->GetName().c_str()),
                        return NOT_CHANGED);
      auto range_size = range.size();
      FUSION_PASS_CHECK(static_cast<int64_t>(range_size) <= aixs_const_val,
                        OP_LOGW(FUSED_OP_TYPE.c_str(), "Node:%s shape range is invalid.",
                        in_op_desc_ptr->GetName().c_str()),
                        return NOT_CHANGED);
      if (range[aixs_const_val].second != -1 && range[aixs_const_val].second < COMP) {
        flag = false;
      }
    }
    FUSION_PASS_CHECK(flag,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node:%s does not need fusion.",
                      in_op_desc_ptr->GetName().c_str()),
                      return NOT_CHANGED);
  } else {
    bool flag = false;
    for (auto iter = x_input_shape.begin(); iter != x_input_shape.end(); iter++) {
      if (*iter > COMP) {
        flag = true;
      }
    }
    FUSION_PASS_CHECK(flag,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node:%s does not need fusion.",
                      in_op_desc_ptr->GetName().c_str()),
                      return NOT_CHANGED);
  }
  /* when argmaxv2 outputdata's dtype is int32 does not need fusion, otherwise,
  modify output's tensor dtype int64->int32 */
  DataType outputdata_type = in_op_desc_ptr->GetOutputDesc(0).GetDataType();
  FUSION_PASS_CHECK(outputdata_type == ge::DT_INT32,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), " when %s outputdata's dtype is int32 does not need fusion.",
                    in_op_desc_ptr->GetName().c_str()),
                    return NOT_CHANGED);
  // set op_argmaxv2's attr
  DataType types1 = ge::DT_INT32;
  op_argmaxv2.SetAttr(DTYPE, types1);
  FUSION_PASS_CHECK(op_argmaxv2.GetAttr(DTYPE, types1) == ge::GRAPH_FAILED,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Get attr %s to node %s failed.",
                    DTYPE, in_op_desc_ptr->GetName().c_str()),
                    return NOT_CHANGED);
  // MutableInputDesc Support modification
  in_op_desc_ptr->MutableOutputDesc(0)->SetDataType(ge::DT_INT32);

  FUSION_PASS_CHECK(AddCastAfterNode(argmaxv2_node, 0, ge::DT_INT64, graph) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Add cast node faild"), return NOT_CHANGED);
  OP_LOGI(argmaxv2_node_type.c_str(), "End to insert Cast after %s.", argmaxv2_node->GetName().c_str());

  return SUCCESS;
}
REGISTER_PASS("ArgMaxV2FusionPass", BUILT_IN_GRAPH_PASS, ArgMaxV2FusionPass);
}  // namespace fe
