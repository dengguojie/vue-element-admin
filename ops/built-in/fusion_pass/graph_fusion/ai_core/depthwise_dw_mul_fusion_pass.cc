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
static const int64_t COUT = 16;
static const int64_t CIN = 16;
const int32_t UNKNOW_RANK_SHAPE = -2;
const int32_t UNKNOW_RANK_DIM = 1;
static const fp16_t FP16_NUM_ZERO = 0;
const int32_t N_DIM = 0;
const int32_t C_DIM = 1;
const int32_t H_DIM = 2;
const int32_t W_DIM = 3;
const int32_t SHAPE_LENTH = 4;
const std::vector<ge::Format> kFormatList = {FORMAT_NHWC, FORMAT_NCHW, FORMAT_HWCN};
const std::map<ge::Format, std::vector<int32_t>> kFormatInNchwDimMap = {
  {FORMAT_NCHW, {0, 1, 2, 3}},
  {FORMAT_NHWC, {0, 3, 1, 2}},
  {FORMAT_HWCN, {3, 2, 0, 1}}
};

Status GenerateConstFP16Dynamic(const vector<int64_t>& shape, float area_factor, float& output1) {
  float* output = &output1;
  for (int64_t i = 0; i < shape[N_DIM]; i++) {
    for (int64_t k = 0; (k < shape[H_DIM] && k < shape[W_DIM]); k++) {
      output[i * (shape[H_DIM] * shape[W_DIM]) + k * shape[W_DIM] + k] = area_factor;
    }
  }
  return SUCCESS;
}

NodePtr DepthwiseDwMulFusionPass::AddMul(ge::ComputeGraph& graph, ge::NodePtr& depthwise_dw_node,
                                         ge::Format& input_origin_format, const bool& is_dynamic) {
  OP_LOGI("Enter DepthwiseDwMulFusionPass::AddMul");
  ge::OutDataAnchorPtr depthwise_dw_anchor_ptr1 = depthwise_dw_node->GetOutDataAnchor(0);
  ge::NodePtr post_node = nullptr;
  ge::NodePtr mul_node = nullptr;
  ge::NodePtr reshape_node = nullptr;
  int64_t mul_n = 0;
  int64_t mul_h = 0;
  int64_t mul_w = 0;
  int64_t mul_c = 0;
  int64_t mul_c1 = 0;
  int64_t groups = 0;
  int64_t multiplier = 0;
  OP_LOGI("in AddMul After get variable");
  // creat a antiquant node
  std::shared_ptr<ge::OpDesc> mul_desc = nullptr;
  mul_desc = std::make_shared<ge::OpDesc>(depthwise_dw_node->GetName() + "_mul_layer", "Mul");
  FUSION_PASS_CHECK(mul_desc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "mul_desc is null, mul failed."),
                    return nullptr);

  // add input
  ge::GeTensorDesc input_desc = depthwise_dw_node->GetOpDesc()->GetOutputDesc(0);
  ge::GeShape mul_shape = input_desc.GetShape();
  vector<int64_t> dim_mul = mul_shape.GetDims();

  vector<int32_t> dim_map;
  if (dim_mul.size() != 0) {
    if (std::find(kFormatList.begin(), kFormatList.end(), input_origin_format) !=  kFormatList.end()) {
      dim_map = kFormatInNchwDimMap.at(input_origin_format);
      mul_n = dim_mul[dim_map[N_DIM]];
      mul_c = dim_mul[dim_map[C_DIM]];
      mul_h = dim_mul[dim_map[H_DIM]];
      mul_w = dim_mul[dim_map[W_DIM]];
    } else {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "input_origin_format only support NHWC, NCHW and HWCN");
      return nullptr;
    }
  } else {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "dim_mul is null, please check!");
    return nullptr;
  }
  OP_LOGI("in AddMul After get mul_n, H, W, C");

  ge::AttrUtils::GetInt(depthwise_dw_node->GetOpDesc(), "groups", groups);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "groups is %d", groups);
  multiplier = mul_c * mul_n / groups;

  mul_c1 = (groups + COUT - 1) / COUT;
  OP_LOGI("in AddMul After calculate mul_dim_info");
  ge::OpDescPtr reshape_desc;
  ge::GeTensorDesc output_desc;
  if (!is_dynamic) {
    vector<int64_t> mul_dim_info = {mul_c1 * mul_h * mul_w, multiplier, COUT, COUT};
    ge::GeShape mulInputShape(mul_dim_info);
    input_desc.SetShape(mulInputShape);
    input_desc.SetOriginShape(ge::GeShape({mul_h, mul_w, 1, mul_n * mul_c}));
    input_desc.SetOriginFormat(input_origin_format);
    input_desc.SetDataType(ge::DT_FLOAT);
    input_desc.SetOriginDataType(ge::DT_FLOAT);
    ge::GeShape mulOutputShape(mul_dim_info);
    output_desc.SetShape(mul_shape);
    output_desc.SetOriginShape(ge::GeShape({mul_h, mul_w, 1, mul_n * mul_c}));
    output_desc.SetOriginFormat(input_origin_format);
    output_desc.SetDataType(ge::DT_FLOAT);
    FUSION_PASS_CHECK(mul_desc->AddInputDesc(input_desc) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add mul_desc input failed."),
                      return nullptr);
    FUSION_PASS_CHECK(mul_desc->AddOutputDesc(output_desc) != SUCCESS,
                  OP_LOGE(FUSED_OP_TYPE.c_str(), "add mul_desc output failed."), return nullptr);

    ge::GeShape input_shape({mul_h, mul_w, 1, mul_n * mul_c});
    ge::GeTensorDesc reshape_input_desc(input_shape, FORMAT_HWCN, ge::DT_FLOAT);
    reshape_input_desc.SetShape(input_shape);
    reshape_input_desc.SetOriginShape(input_shape);

    ge::GeShape output_shape({mul_h, mul_w, mul_c, mul_n});
    ge::GeTensorDesc reshape_output_desc(output_shape, FORMAT_HWCN, ge::DT_FLOAT);
    reshape_output_desc.SetShape(output_shape);
    reshape_output_desc.SetOriginShape(output_shape);
    FUSION_PASS_MAKE_SHARED(reshape_desc = std::make_shared<ge::OpDesc>(
                              depthwise_dw_node->GetName() + "/Reshape", "Reshape"), return nullptr);
    FUSION_PASS_CHECK(reshape_desc->AddInputDesc("x", reshape_input_desc) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add input desc x to reshape."), return nullptr);
    FUSION_PASS_CHECK(reshape_desc->AddOutputDesc("y", reshape_output_desc) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add input desc y to reshape."), return nullptr);
    ge::AttrUtils::SetListInt(reshape_desc, "shape", {mul_h, mul_w, mul_c, mul_n});
    reshape_node = graph.AddNode(reshape_desc);
  } else {
    ge::GeTensorDesc input_desc_mul;
    vector<int64_t> mul_dim_info = {mul_c1 * mul_h * mul_w, 1, COUT, COUT};
    ge::GeShape mulInputShape(mul_dim_info);
    input_desc_mul.SetShape(mulInputShape);
    input_desc_mul.SetOriginShape(mul_shape);
    input_desc_mul.SetFormat(ge::FORMAT_FRACTAL_Z);
    input_desc_mul.SetOriginFormat(input_origin_format);
    input_desc_mul.SetDataType(ge::DT_FLOAT);
    input_desc_mul.SetOriginDataType(ge::DT_FLOAT);
    output_desc.SetShape(mulInputShape);
    output_desc.SetOriginShape(mul_shape);
    output_desc.SetFormat(ge::FORMAT_FRACTAL_Z);
    output_desc.SetOriginFormat(input_origin_format);
    output_desc.SetDataType(ge::DT_FLOAT);
    output_desc.SetOriginDataType(ge::DT_FLOAT);
    FUSION_PASS_CHECK(mul_desc->AddInputDesc("x1", input_desc_mul) != SUCCESS,
                  OP_LOGE(FUSED_OP_TYPE.c_str(), "add mul_desc input failed."), return nullptr);
    FUSION_PASS_CHECK(mul_desc->AddOutputDesc("y", output_desc) != SUCCESS,
                  OP_LOGE(FUSED_OP_TYPE.c_str(), "add mul_desc output failed."), return nullptr);
  }

  mul_node = graph.AddNode(mul_desc);
  if (!is_dynamic) {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0),
                        reshape_node->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge between pooling and next node failed!"),
                      return nullptr);
  }
  auto end_node = is_dynamic ? mul_node : reshape_node;

  for (auto postAnchorPtr0 : depthwise_dw_anchor_ptr1->GetPeerInDataAnchors()) {
    post_node = postAnchorPtr0->GetOwnerNode();

    // remove edge between depthwiseDw and next node
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(postAnchorPtr0, depthwise_dw_anchor_ptr1) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge between pooling and next node failed!"),
                      return nullptr);

    // add edge between mul and next_node
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(end_node->GetOutDataAnchor(0), postAnchorPtr0) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              end_node->GetName().c_str(), post_node->GetName().c_str()),
                      return nullptr);
  }
  // add edge between depthwiseDw and mul
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(depthwise_dw_anchor_ptr1, mul_node->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            depthwise_dw_node->GetName().c_str(), mul_node->GetName().c_str()),
                    return nullptr);
  OP_LOGI("Leave DepthwiseDwMulFusionPass::AddMul");
  return mul_node;
}

Status DepthwiseDwMulFusionPass::AddCoffe(ge::NodePtr& mul_node, const int64_t matrix_size,
                                          vector<int64_t>& dim_info, const bool& is_dynamic) {
  OP_LOGI("Enter DepthwiseDwMulFusionPass::AddCoffe");
  int64_t output_n = 0;
  int64_t output_h = 0;
  int64_t output_w = 0;
  int64_t output_c = 0;
  ge::GeTensorDesc input_desc0 = mul_node->GetOpDesc()->GetInputDesc(0);
  ge::Format input_desc0_origin_format = input_desc0.GetOriginFormat();
  vector<int64_t> out_dim_info = input_desc0.GetOriginShape().GetDims();
  OP_LOGI("in AddCoffe get variable");
  vector<int32_t> dim_map;
  if (out_dim_info.size() == SHAPE_LENTH) {
    if (std::find(kFormatList.begin(), kFormatList.end(), input_desc0_origin_format) !=  kFormatList.end()) {
      dim_map = kFormatInNchwDimMap.at(input_desc0_origin_format);
      output_n = out_dim_info[dim_map[N_DIM]];
      output_c = out_dim_info[dim_map[C_DIM]];
      output_h = out_dim_info[dim_map[H_DIM]];
      output_w = out_dim_info[dim_map[W_DIM]];
    }
  } else {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "dim_info is not right, please check!");
    return NOT_CHANGED;
  }
  OP_LOGI("in AddCoffe get output n, H, W, C");

  ge::GeTensorPtr coffe_ptr = nullptr;
  int64_t coffe_size = matrix_size;
  FUSION_PASS_CHECK(coffe_size <= 0, OP_LOGE(FUSED_OP_TYPE.c_str(), "coffe_size is Invalid"), return PARAM_INVALID);
  unique_ptr<float[]> inputAssit(new (std::nothrow) float[coffe_size]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                    return PARAM_INVALID);
  vector<int64_t> mul_dim_info;
  if (!is_dynamic) {
    Status ret = NnSet(coffe_size, FLOAT_NUM_ONE, *reinterpret_cast<float*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);
    OP_LOGI("in AddCoffe after fusion_pass_check");
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "CoffeFP16 is failed."), return ret);
  } else {
    int64_t output_c1 = (output_n + COUT - 1) / COUT;
    mul_dim_info = {output_c1 * output_h * output_w, 1, COUT, COUT};
    Status ret = GenerateConstFP16Dynamic(mul_dim_info, FLOAT_NUM_ONE, *reinterpret_cast<float*>(inputAssit.get()));
    OP_LOGI("in AddCoffe after fusion_pass_check");
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "CoffeFP16 is failed."), return ret);
  }

  vector<int64_t> coffe_dim_info_swap;
  if (input_desc0_origin_format == FORMAT_NHWC) {
    coffe_dim_info_swap = {output_n*output_c, output_h, output_w, 1};
  } else if (input_desc0_origin_format == FORMAT_NCHW) {
    coffe_dim_info_swap = {output_n*output_c, 1, output_h, output_w};
  } else if (input_desc0_origin_format == FORMAT_HWCN) {
    coffe_dim_info_swap = {output_h, output_w, 1, output_n*output_c};
  } else {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "format is wrong, please check!");
    return PARAM_INVALID;
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "in AddCoffe after coffe_dim_info_swap");
  // set const node shape
  ge::GeTensorDesc coffe_desc;
  if (!is_dynamic) {
    ge::GeShape coffeShape(dim_info);
    ge::GeShape coffeShapeOrigin(coffe_dim_info_swap);
    coffe_desc.SetShape(coffeShapeOrigin);
    coffe_desc.SetDataType(ge::DT_FLOAT);
    coffe_desc.SetOriginFormat(input_desc0_origin_format);
    coffe_desc.SetOriginShape(coffeShapeOrigin);
    coffe_desc.SetOriginDataType(ge::DT_FLOAT);
  } else {
    coffe_desc.SetShape(ge::GeShape(mul_dim_info));
    coffe_desc.SetDataType(ge::DT_FLOAT);
    coffe_desc.SetFormat(ge::FORMAT_FRACTAL_Z);
    coffe_desc.SetOriginFormat(input_desc0_origin_format);
    coffe_desc.SetOriginShape(ge::GeShape(out_dim_info));
    coffe_desc.SetOriginDataType(ge::DT_FLOAT);
  }
  FUSION_PASS_MAKE_SHARED((coffe_ptr = std::make_shared<ge::GeTensor>(
                            coffe_desc, reinterpret_cast<uint8_t*>(inputAssit.get()), coffe_size * sizeof(float))),
                          coffe_ptr = nullptr;
                          return PARAM_INVALID);
  ge::OpDescPtr mul_desc = mul_node->GetOpDesc();
  FUSION_PASS_CHECK(mul_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mul_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  OP_LOGI("in AddCoffe after SetShape, DataType, Format");
  vector<ge::GeTensorPtr> weights = {coffe_ptr};
  ge::OpDescUtils::SetWeights(mul_node, weights);
  mul_desc = mul_node->GetOpDesc();
  auto x2_desc = mul_desc->MutableInputDesc(1);
  x2_desc->SetName("x2");
  ge::AttrUtils::SetListStr(mul_desc, "_input_name_key", {"x1", "x2"});
  auto const_input_nodes = OpDescUtils::GetConstInputs(mul_node);
  NodePtr const_input = nullptr;
  if (const_input_nodes.size() != 0) {
    const_input = const_input_nodes[0];
  } else {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "const_input_nodes is null, please check!");
    return PARAM_INVALID;
  }
  const_input->GetOpDesc()->SetType(CONSTANTOP);

  OP_LOGI("Leave DepthwiseDwMulFusionPass::AddCoffe");
  return SUCCESS;
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
  // avgpool node
  ge::NodePtr depthwise_dw_node = GetNodeFromMapping(PATTERN_DEPTHWISEDW, mapping);
  FUSION_PASS_CHECK(depthwise_dw_node == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "depthwise_dw_node is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr depthwise_dw_desc = depthwise_dw_node->GetOpDesc();
  FUSION_PASS_CHECK(depthwise_dw_desc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "depthwise_dw_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  OP_LOGD("After get depthwise_dw_desc");
  ge::GeTensorDesc depthwise_dw_input_tensor = depthwise_dw_node->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc depthwise_dw_output_tensor = depthwise_dw_node->GetOpDesc()->GetOutputDesc(0);
  OP_LOGD("After get depthwise_dw_input_tensor and depthwise_dw_output_tensor");
  // get shape
  ge::GeShape depthwise_dw_input_shape = depthwise_dw_input_tensor.GetShape();
  ge::GeShape depthwise_dw_output_shape = depthwise_dw_output_tensor.GetShape();
  ge::Format output_origin_format = depthwise_dw_output_tensor.GetOriginFormat();
  // GESHAPE->vector
  OP_LOGD("After get depthwise_dw_input_shape, depthwise_dw_output_shape, input_origin_format, output_origin_format");
  vector<int64_t> dim_info = depthwise_dw_input_shape.GetDims();
  vector<int64_t> out_dim_info = depthwise_dw_output_shape.GetDims();
  bool is_dynamic = false;
  bool is_fuzz_build = false;
  std::vector<int64_t> filter_size;
  std::vector<int64_t> filter_size_reset(SHAPE_LENTH);
  ge::AttrUtils::GetBool(depthwise_dw_desc, ge::ATTR_NAME_FUZZ_BUILD, is_fuzz_build);
  is_dynamic = (dim_info.size() == UNKNOW_RANK_DIM && dim_info[0] == UNKNOW_RANK_SHAPE) ||
                std::find(dim_info.begin(), dim_info.end(), -1) != dim_info.end() || is_fuzz_build;
  if (is_dynamic) {
    filter_size = out_dim_info;
  } else if (!ge::AttrUtils::GetListInt(depthwise_dw_desc, "filter_size", filter_size)) {
    filter_size = depthwise_dw_output_shape.GetDims();
    OP_LOGD(FUSED_OP_TYPE.c_str(), "get filter_size from output_shape.");
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "DepthwiseDwMulFusionPass static shape");
  }

  int64_t output_c1 = 0;
  int64_t output_n = 0;
  int64_t output_c = 0;
  int64_t output_h = 0;
  int64_t output_w = 0;
  int64_t groups = 0;
  int64_t multiplier = 0;
  vector<int32_t> dim_map;
  if (out_dim_info.size() == SHAPE_LENTH && filter_size.size() == SHAPE_LENTH) {
    if (std::find(kFormatList.begin(), kFormatList.end(), output_origin_format) !=  kFormatList.end()) {
      dim_map = kFormatInNchwDimMap.at(output_origin_format);
      output_n = out_dim_info[dim_map[N_DIM]];
      output_c = out_dim_info[dim_map[C_DIM]];
      output_h = out_dim_info[dim_map[H_DIM]];
      output_w = out_dim_info[dim_map[W_DIM]];
      filter_size_reset[dim_map[N_DIM]] = filter_size[dim_map[N_DIM]] * filter_size[dim_map[C_DIM]];
      filter_size_reset[dim_map[C_DIM]] = 1;
      filter_size_reset[dim_map[H_DIM]] = filter_size[dim_map[H_DIM]];
      filter_size_reset[dim_map[W_DIM]] = filter_size[dim_map[W_DIM]];
    }
  } else {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "dim_info is not right, please check!");
    return NOT_CHANGED;
  }
  // when static op or dynamic op phase_running, is_dynamic = false
  OP_LOGD(FUSED_OP_TYPE.c_str(), "After get output N, H, W, C");
  vector<int64_t> dim_info2;
  ge::AttrUtils::SetListInt(depthwise_dw_desc, "filter_size", filter_size_reset);
  depthwise_dw_output_tensor.SetOriginShape(ge::GeShape(filter_size_reset));
  depthwise_dw_output_tensor.SetShape(ge::GeShape(filter_size_reset));
  depthwise_dw_desc->UpdateOutputDesc(0, depthwise_dw_output_tensor);
  dim_info2 = depthwise_dw_output_tensor.GetOriginShape().GetDims();

  OP_LOGD("After UpdateOutputDesc filter_grad, groups");
  ge::AttrUtils::GetInt(depthwise_dw_node->GetOpDesc(), "groups", groups);
  if (groups == 0) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "groups should not be 0.");
    return NOT_CHANGED;
  }
  multiplier = output_c * output_n / groups;
  output_c1 = (groups + COUT - 1) / COUT;
  int64_t matrix_size = output_n * output_h * output_w * output_c;
  if (is_dynamic) {
    matrix_size = output_c1 * output_h * output_w * COUT * multiplier * COUT;
  }
  FUSION_PASS_CHECK(matrix_size <= 0, OP_LOGE(FUSED_OP_TYPE.c_str(), "matrix_size is Invalid"), return PARAM_INVALID);
  vector<int64_t> assit_dim_info_origin = {output_c1 * output_h * output_w, 1, COUT*multiplier, COUT};

  ge::NodePtr mul_node = AddMul(graph, depthwise_dw_node, output_origin_format, is_dynamic);
  FUSION_PASS_CHECK(mul_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mul_node is null, AddMul failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(AddCoffe(mul_node, matrix_size, assit_dim_info_origin, is_dynamic) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "AddCoffe failed."), return PARAM_INVALID);
  OP_LOGI("Leave DepthwiseDwMulFusionPass Fusion");
  return SUCCESS;
}
REGISTER_PASS("DepthwiseDwMulFusionPass", BUILT_IN_GRAPH_PASS, DepthwiseDwMulFusionPass);
}  // namespace fe
