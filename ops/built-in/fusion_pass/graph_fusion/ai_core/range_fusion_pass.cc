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
 * \file range_fusion_pass.cpp
 * \brief Range fusion pass(Range --> RangeD)
 */
#include "range_fusion_pass.h"
#include <cmath>
#include <memory>
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
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "securec.h"

using namespace ge;
namespace fe {
static float EPSILON = 0.0000001;
static const char* FUSED_NODE = "Range";
static const string PATTERN_FUSED_NODE = "Range";

static void CalcData(const Tensor& data, const DataType& dtype, vector<float>& constVec) {
  const uint8_t* const_data = data.GetData();
  if (const_data == nullptr) {
    return;
  }
  size_t size = data.GetSize() / sizeof(float);
  for (size_t i = 0; i < size; ++i) {
    constVec.push_back(*((float*)const_data + i));
  }
}

static void CalcData(const Tensor& data, const DataType& dtype, vector<int32_t>& constVec) {
  const uint8_t* const_data = data.GetData();
  if (const_data == nullptr) {
    return;
  }
  size_t size = data.GetSize() / sizeof(int32_t);
  for (size_t i = 0; i < size; ++i) {
    constVec.push_back(*((int32_t*)const_data + i));
  }
}

static void AssistFloatHelp(const int32_t n, float* output) {
  for (int32_t i = 0; i < n; i++) {
    output[i] = static_cast<float>(i);
  }
}

static void AssistIntHelp(const int32_t n, int32_t* output) {
  for (int32_t i = 0; i < n; i++) {
    output[i] = i;
  }
}

vector<FusionPattern*> RangeFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (nothrow) FusionPattern("RangeFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSED_NODE, {FUSED_NODE}).SetOutput(PATTERN_FUSED_NODE);
  patterns.push_back(pattern);

  return patterns;
}

Status RangeFusionPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Range fusion in!");

  // get node
  NodePtr range_node = GetNodeFromMapping(PATTERN_FUSED_NODE, mapping);
  FUSION_PASS_CHECK(range_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "range_node is null, fusion failed."),
                    return PARAM_INVALID);
  // get desc
  OpDescPtr range_desc = range_node->GetOpDesc();
  FUSION_PASS_CHECK(range_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "range_desc is null, fusion failed."),
                    return PARAM_INVALID);

  // get op
  Operator range_op = OpDescUtils::CreateOperatorFromNode(range_node);

  // check input dtype
  DataType start_type = range_op.GetInputDesc("start").GetDataType();
  DataType limit_type = range_op.GetInputDesc("limit").GetDataType();
  DataType delta_type = range_op.GetInputDesc("delta").GetDataType();
  if ((start_type != limit_type) && (limit_type != delta_type)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the dtype of input tensor is not same, graph not changed");
    return NOT_CHANGED;
  }
  if ((start_type != DT_INT32) && (limit_type != DT_FLOAT)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the dtype of input tensor is not int32 or float, graph not changed");
    return NOT_CHANGED;
  }

  // get const data
  Tensor start_tensor;
  if (range_op.GetInputConstData("start", start_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Range has input of start which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }
  Tensor limit_tensor;
  if (range_op.GetInputConstData("limit", limit_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Range has input of limit which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }
  Tensor delta_tensor;
  if (range_op.GetInputConstData("delta", delta_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Range has input of delta which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }

  float start_fp = 0;
  float limit_fp = 0;
  float delta_fp = 0;
  if (start_type == DT_FLOAT) {
    vector<float> start_vec;
    vector<float> limit_vec;
    vector<float> delta_vec;
    CalcData(start_tensor, start_type, start_vec);
    CalcData(limit_tensor, limit_type, limit_vec);
    CalcData(delta_tensor, delta_type, delta_vec);
    start_fp = start_vec[0];
    limit_fp = limit_vec[0];
    delta_fp = delta_vec[0];
  } else {
    vector<int32_t> start_vec;
    vector<int32_t> limit_vec;
    vector<int32_t> delta_vec;
    CalcData(start_tensor, start_type, start_vec);
    CalcData(limit_tensor, limit_type, limit_vec);
    CalcData(delta_tensor, delta_type, delta_vec);
    start_fp = float(start_vec[0]);
    limit_fp = float(limit_vec[0]);
    delta_fp = float(delta_vec[0]);
  }

  // get input
  InDataAnchorPtr start_anchor_in = range_node->GetInDataAnchor(0);
  InDataAnchorPtr limit_anchor_in = range_node->GetInDataAnchor(1);
  InDataAnchorPtr delta_anchor_in = range_node->GetInDataAnchor(2);
  OutDataAnchorPtr start_anchor_out = start_anchor_in->GetPeerOutAnchor();
  OutDataAnchorPtr limit_anchor_out = limit_anchor_in->GetPeerOutAnchor();
  OutDataAnchorPtr delta_anchor_out = delta_anchor_in->GetPeerOutAnchor();
  Format const_format = range_op.GetInputDesc("start").GetFormat();
  FUSION_PASS_CHECK((fabs(delta_fp) < EPSILON), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Devide by 0 exception."),
                    return PARAM_INVALID);
  int dim_num = int(ceil(abs(limit_fp - start_fp) / abs(delta_fp)));

  // generate assist
  GeShape assist_shape({dim_num});
  GeTensorDesc assist_desc;
  assist_desc.SetDataType(start_type);
  assist_desc.SetFormat(const_format);
  assist_desc.SetShape(assist_shape);
  GeTensorPtr assist_ptr = nullptr;
  if (start_type == DT_INT32) {
    unique_ptr<int32_t[]> input_assist(new (nothrow) int32_t[dim_num]());
    FUSION_PASS_CHECK(input_assist.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    AssistIntHelp(dim_num, input_assist.get());
    FUSION_PASS_MAKE_SHARED(
        (assist_ptr = make_shared<GeTensor>(assist_desc, reinterpret_cast<uint8_t*>(input_assist.get()),
                                            dim_num * sizeof(int32_t))),
        assist_ptr = nullptr;
        return PARAM_INVALID);
  } else {
    unique_ptr<float[]> input_assist(new (nothrow) float[dim_num]());
    FUSION_PASS_CHECK(input_assist.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    AssistFloatHelp(dim_num, input_assist.get());
    FUSION_PASS_MAKE_SHARED((assist_ptr = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist.get()), dim_num * sizeof(float))),
                            assist_ptr = nullptr;
                            return PARAM_INVALID);
  }

  // const to attr
  range_op.SetAttr("start", start_fp);
  range_op.SetAttr("limit", limit_fp);
  range_op.SetAttr("delta", delta_fp);
  GraphUtils::RemoveEdge(start_anchor_out, start_anchor_in);
  GraphUtils::RemoveEdge(limit_anchor_out, limit_anchor_in);
  GraphUtils::RemoveEdge(delta_anchor_out, delta_anchor_in);
  NodeUtils::ClearInDataAnchor(range_node, start_anchor_in);
  NodeUtils::ClearInDataAnchor(range_node, limit_anchor_in);
  NodeUtils::ClearInDataAnchor(range_node, delta_anchor_in);
  OpDescUtils::ClearInputDesc(range_desc, 2);
  OpDescUtils::ClearInputDesc(range_desc, 1);
  OpDescUtils::ClearInputDesc(range_desc, 0);

  // set const input and modify node
  vector<GeTensorPtr> weights = {assist_ptr};
  OpDescUtils::SetWeights(range_node, weights);
  auto const_nodes = OpDescUtils::GetConstInputs(range_node);
  NodePtr const_node = const_nodes[0];
  const_node->GetOpDesc()->SetType("Constant");
  vector<bool> is_input_const = {true};
  range_desc->SetIsInputConst(is_input_const);
  range_desc->SetType("RangeD");
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Range fusion success!");
  return SUCCESS;
}

REGISTER_PASS("RangeFusionPass", BUILT_IN_GRAPH_PASS, RangeFusionPass);
}  // namespace fe
