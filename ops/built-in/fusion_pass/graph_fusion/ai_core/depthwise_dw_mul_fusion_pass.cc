/*
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
 * \file depthwise_dw_mul_fusion_pass.cc
 * \brief depthwise_dw_mul_fusion_pass
 */
#include "depthwise_dw_mul_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "conv_fusion_pass_utils.h"
#include "quant_host_cpu_op_common.h"
#include "op_log.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
static const float UINT_NUM_ZERO = 0;
static const float FLOAT_NUM_ONE = 1;
static const int8_t INT8_NUM_ZERO = 0;
static const int8_t INT8_NUM_ONE = 1;
static const std::string PATTERN_DEPTHWISEDW = "DepthwiseConv2DBackpropFilterD";
static const std::string CONSTANTOP = "Const";
static const char* DEPTHWISEDW = "DepthwiseConv2DBackpropFilterD";
static const char* DEPTHWISEDW_DYN = "DepthwiseConv2DBackpropFilter";
const int32_t UNKNOW_RANK_SHAPE = -2;
const int32_t UNKNOW_RANK_DIM = 1;
static const fp16_t FP16_NUM_ZERO = 0;
const int32_t N_DIM = 0;
const int32_t C_DIM = 1;
const int32_t H_DIM = 2;
const int32_t W_DIM = 3;
const int32_t SHAPE_LENTH = 4;
const size_t kOriShapeDim = 4;
const std::vector<ge::Format> kFormatList = {FORMAT_NCHW, FORMAT_HWCN};
const std::vector<int32_t> kTransposePerm = {1, 0, 2, 3};


ge::OpDescPtr DepthwiseDwMulFusionPass::CreateTranspose(const string& node_name, const ge::GeTensorDesc& output_desc) {
  vector<int64_t> output_shape_vec = output_desc.GetOriginShape().GetDims();
  std::swap(output_shape_vec[N_DIM], output_shape_vec[C_DIM]);
  ge::GeTensorDesc input_desc = output_desc;
  input_desc.SetShape(ge::GeShape(output_shape_vec));
  input_desc.SetOriginShape(ge::GeShape(output_shape_vec));

  ge::OpDescPtr transpose_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(transpose_desc = std::make_shared<ge::OpDesc>(node_name + "/TransposeD", "TransposeD"),
                          return nullptr);
  FUSION_PASS_CHECK(transpose_desc->AddInputDesc("x", input_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add input desc x to transposed."), return nullptr);
  FUSION_PASS_CHECK(transpose_desc->AddOutputDesc("y", output_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add input desc y to transposed."), return nullptr);
  ge::AttrUtils::SetListInt(transpose_desc, "perm", kTransposePerm);
  return transpose_desc;
}

vector<FusionPattern*> DepthwiseDwMulFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  OP_LOGI("Enter DepthwiseDwMulFusionPass Patterns");
  // define AvgPoolFusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DepthwiseDwMulFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_DEPTHWISEDW, {DEPTHWISEDW_DYN, DEPTHWISEDW}).SetOutput(PATTERN_DEPTHWISEDW);

  patterns.push_back(pattern);
  OP_LOGI("Leave DepthwiseDwMulFusionPass Patterns");
  return patterns;
}

Status DepthwiseDwMulFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                        vector<ge::NodePtr>& /* fusion_nodes */) {
  OP_LOGI("Enter DepthwiseDwMulFusionPass Fusion");
  ge::NodePtr depthwise_dw_node = GetNodeFromMapping(PATTERN_DEPTHWISEDW, mapping);
  FUSION_PASS_CHECK(depthwise_dw_node == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "depthwise_dw_node is null, fusion failed."), return PARAM_INVALID);
  ge::OpDescPtr depthwise_dw_desc = depthwise_dw_node->GetOpDesc();
  FUSION_PASS_CHECK(depthwise_dw_desc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "depthwise_dw_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  ge::GeTensorDesc dedy_input_tensor = depthwise_dw_node->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc dedw_output_tensor = depthwise_dw_node->GetOpDesc()->GetOutputDesc(0);
  ge::GeTensorDesc dedw_output_old_tensor = dedw_output_tensor;
  // get shape
  ge::GeShape dedy_shape = dedy_input_tensor.GetShape();
  ge::GeShape dedw_shape = dedw_output_tensor.GetShape();
  vector<int64_t> dedy_shape_vec = dedy_shape.GetDims();
  vector<int64_t> dedw_shape_vec = dedw_shape.GetDims();

  bool is_fuzz_build = false;
  ge::AttrUtils::GetBool(depthwise_dw_desc, ge::ATTR_NAME_FUZZ_BUILD, is_fuzz_build);
  bool is_dynamic = (dedy_shape_vec.size() == UNKNOW_RANK_DIM && dedy_shape_vec[0] == UNKNOW_RANK_SHAPE) ||
                    std::find(dedy_shape_vec.begin(), dedy_shape_vec.end(), -1) != dedy_shape_vec.end() ||
                    is_fuzz_build;

  std::vector<int64_t> filter_size;
  if (is_dynamic || !ge::AttrUtils::GetListInt(depthwise_dw_desc, "filter_size", filter_size)) {
    filter_size = dedw_shape_vec;
    OP_LOGD(FUSED_OP_TYPE.c_str(), "get filter_size from output_shape.");
  }

  int groups = 0;
  ge::AttrUtils::GetInt(depthwise_dw_node->GetOpDesc(), "groups", groups);
  FUSION_PASS_CHECK(groups <= 0, OP_LOGE(FUSED_OP_TYPE.c_str(), "groups should not less or equal with 0."),
                    return PARAM_INVALID);

  vector<int64_t> filter_size_reset;
  vector<int64_t> fractal_shape_reset;
  const ge::Format filter_ori_format = dedw_output_tensor.GetOriginFormat();
  FUSION_PASS_CHECK(!ConvFusionPassUtils::GetResizeDepthwiseFilter(filter_size, filter_ori_format, groups,
                                                                   filter_size_reset, fractal_shape_reset),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get filter resize shape failed."), return PARAM_INVALID);
  dedw_output_tensor.SetShape(ge::GeShape(fractal_shape_reset));
  dedw_output_tensor.SetOriginShape(ge::GeShape(filter_size_reset));

  ge::NodePtr reshape_node = nullptr;
  ge::NodePtr back_node = nullptr;
  if (filter_ori_format == ge::FORMAT_NCHW) {
    ge::OpDescPtr transpose_desc = CreateTranspose(depthwise_dw_node->GetName(), dedw_output_old_tensor);
    back_node = graph.AddNode(transpose_desc);
    FUSION_PASS_CHECK(back_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add reshape node into graph."),
                      return PARAM_INVALID);

    ge::OpDescPtr reshape_desc_ptr = ConvFusionPassUtils::CreateReshape(
        depthwise_dw_node->GetName(), dedw_output_tensor, transpose_desc->GetInputDesc(0));
    reshape_node = graph.AddNode(reshape_desc_ptr);
    FUSION_PASS_CHECK(reshape_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "failed to add reshape node into graph."),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(reshape_node->GetOutDataAnchor(0), back_node->GetInDataAnchor(0)) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s and node %s failed.", reshape_node->GetName().c_str(),
                back_node->GetName().c_str()),
                return PARAM_INVALID);
  } else {
    ge::OpDescPtr reshape_desc_ptr = ConvFusionPassUtils::CreateReshape(depthwise_dw_node->GetName(),
                                                                        dedw_output_tensor, dedw_output_old_tensor);
    reshape_node = graph.AddNode(reshape_desc_ptr);
    FUSION_PASS_CHECK(reshape_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "failed to add reshape node into graph."),
                      return PARAM_INVALID);
    back_node = reshape_node;
  }
  FUSION_PASS_CHECK(!ConvFusionPassUtils::ReplaceOutputAnchor(depthwise_dw_node, 0, back_node, 0),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to replace dw output anchor to reshape."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(depthwise_dw_node->GetOutDataAnchor(0), reshape_node->GetInDataAnchor(0)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s and node %s failed.",
              depthwise_dw_node->GetName().c_str(), reshape_node->GetName().c_str()),
      return PARAM_INVALID);

  // when static op or dynamic op phase_running, is_dynamic = false
  ge::AttrUtils::SetListInt(depthwise_dw_desc, "filter_size", filter_size_reset);
  depthwise_dw_desc->UpdateOutputDesc(0, dedw_output_tensor);
  OP_LOGI("Leave DepthwiseDwMulFusionPass Fusion");
  return SUCCESS;
}
REGISTER_PASS("DepthwiseDwMulFusionPass", BUILT_IN_GRAPH_PASS, DepthwiseDwMulFusionPass);
}  // namespace fe
