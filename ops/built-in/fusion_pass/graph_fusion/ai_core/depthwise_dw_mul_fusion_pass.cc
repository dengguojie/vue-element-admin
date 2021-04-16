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
static const int64_t COUT = 16;
static const int64_t CIN = 16;
static const int64_t COUT32 = 32;
static const int64_t CIN32 = 32;
const int32_t INDEX_CO_avg = 1;
const int32_t INDEX_CI_avg = 0;
static const fp16_t FP16_NUM_ZERO = 0;

NodePtr DepthwiseDwMulFusionPass::AddMul(ge::ComputeGraph& graph, ge::NodePtr& depthwise_dw_node, ge::Format& input_origin_format) {
  OP_LOGI("Enter DepthwiseDwMulFusionPass::AddMul");
  ge::OutDataAnchorPtr depthwise_dw_anchor_ptr1 = depthwise_dw_node->GetOutDataAnchor(0);
  ge::NodePtr post_node = nullptr;
  ge::NodePtr mul_node = nullptr;
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
  FUSION_PASS_CHECK(mul_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mul_desc is null, mul failed."), return nullptr);

  // add input
  ge::GeTensorDesc input_desc = depthwise_dw_node->GetOpDesc()->GetOutputDesc(0);
  ge::GeShape mul_shape = input_desc.GetShape();
  vector<int64_t> dim_mul = mul_shape.GetDims();

  if (dim_mul.size() != 0) {
    if (input_origin_format == FORMAT_NHWC) {
      mul_n = dim_mul[0];
      mul_h = dim_mul[1];
      mul_w = dim_mul[2];
      mul_c = dim_mul[3];
    } else if (input_origin_format == FORMAT_NCHW){
      mul_n = dim_mul[0];
      mul_h = dim_mul[2];
      mul_w = dim_mul[3];
      mul_c = dim_mul[1];
    } else if (input_origin_format == FORMAT_HWCN){
      mul_n = dim_mul[3];
      mul_h = dim_mul[0];
      mul_w = dim_mul[1];
      mul_c = dim_mul[2];
    } else {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "input_origin_format only support NHWC and NCHW");
      return nullptr;
    }
  } else {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "dim_mul is null, please check!");
    return nullptr;
  }
  OP_LOGI("in AddMul After get mul_n, H, W, C");

  ge::AttrUtils::GetInt(depthwise_dw_node->GetOpDesc(), "groups", groups);
  multiplier = mul_c * mul_n / groups;

  mul_c1 = (groups + COUT - 1) / COUT;
  vector<int64_t> mul_dim_info = {mul_c1 * mul_h * mul_w, 1, COUT*multiplier, COUT};
  OP_LOGI("in AddMul After calculate mul_dim_info");
  ge::GeShape mulInputShape(mul_dim_info);
  input_desc.SetShape(mulInputShape);
  input_desc.SetOriginShape(ge::GeShape({mul_h, mul_w, 1, mul_n*mul_c}));
  input_desc.SetOriginFormat(input_origin_format);
  input_desc.SetDataType(ge::DT_FLOAT);
  input_desc.SetOriginDataType(ge::DT_FLOAT);
  FUSION_PASS_CHECK(mul_desc->AddInputDesc(input_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add mul_desc input failed."), return nullptr);
  ge::GeTensorDesc output_desc;
  ge::GeShape mulOutputShape(mul_dim_info);
  output_desc.SetShape(mul_shape);
  output_desc.SetOriginShape(ge::GeShape({mul_h, mul_w, 1, mul_n*mul_c}));
  output_desc.SetOriginFormat(input_origin_format);
  output_desc.SetDataType(ge::DT_FLOAT);
  FUSION_PASS_CHECK(mul_desc->AddOutputDesc(output_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add mul_desc output failed."), return nullptr);

  mul_node = graph.AddNode(mul_desc);

  for (auto postAnchorPtr0 : depthwise_dw_anchor_ptr1->GetPeerInDataAnchors()) {
    post_node = postAnchorPtr0->GetOwnerNode();

    // remove edge between depthwiseDw and next node
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(postAnchorPtr0, depthwise_dw_anchor_ptr1) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge between pooling and next node failed!"),
                      return nullptr);

    // add edge between mul and next_node
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), postAnchorPtr0) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              mul_node->GetName().c_str(), post_node->GetName().c_str()),
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

Status DepthwiseDwMulFusionPass::AddCoffe(ge::ComputeGraph& graph, ge::NodePtr& mul_node, const int64_t matrix_size,
                                          vector<int64_t>& dim_info) {
  OP_LOGI("Enter DepthwiseDwMulFusionPass::AddCoffe");
  int64_t output_n = 0;
  int64_t output_h = 0;
  int64_t output_w = 0;
  int64_t output_c = 0;
  ge::GeTensorDesc input_desc0 = mul_node->GetOpDesc()->GetInputDesc(0);
  ge::Format input_desc0_origin_format = input_desc0.GetOriginFormat();
  vector<int64_t> out_dim_info = input_desc0.GetOriginShape().GetDims();
  OP_LOGI("in AddCoffe get variable");
  if (out_dim_info.size() == 4) {
    if (input_desc0_origin_format == FORMAT_NHWC) {
      output_n = out_dim_info[0];
      output_h = out_dim_info[1];
      output_w = out_dim_info[2];
      output_c = out_dim_info[3];
    } else if (input_desc0_origin_format == FORMAT_NCHW) {
      output_n = out_dim_info[0];
      output_h = out_dim_info[2];
      output_w = out_dim_info[3];
      output_c = out_dim_info[1];
    } else if (input_desc0_origin_format == FORMAT_HWCN) {
      output_n = out_dim_info[3];
      output_h = out_dim_info[0];
      output_w = out_dim_info[1];
      output_c = out_dim_info[2];
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
  Status ret = NnSet(coffe_size, FLOAT_NUM_ONE, *reinterpret_cast<float*>(inputAssit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);
  OP_LOGI("in AddCoffe after fusion_pass_check");
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "CoffeFP16 is failed."), return ret);

  vector<int64_t> coffe_dim_info_origin;
  vector<int64_t> coffe_dim_info_CN_swap;
  if (input_desc0_origin_format == FORMAT_NHWC) {
    coffe_dim_info_origin = {output_n, output_h, output_w, output_c};
    coffe_dim_info_CN_swap = {output_n*output_c, output_h, output_w, 1};
  } else if (input_desc0_origin_format == FORMAT_NCHW) {
    coffe_dim_info_origin = {output_n, output_c, output_h, output_w};
    coffe_dim_info_CN_swap = {output_n*output_c, 1, output_h, output_w};
  } else if (input_desc0_origin_format == FORMAT_HWCN) {
    coffe_dim_info_origin = {output_h, output_w, output_c, output_n};
    coffe_dim_info_CN_swap = {output_h, output_w, 1, output_n*output_c};
  } else {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "format is wrong, please check!");
    return PARAM_INVALID;
  }
  OP_LOGI("in AddCoffe after coffe_dim_info_CN_swap");
  // set const node shape
  ge::GeTensorDesc coffe_desc;
  ge::GeShape coffeShape(dim_info);
  ge::GeShape coffeShapeOrigin(coffe_dim_info_CN_swap);
  coffe_desc.SetShape(coffeShapeOrigin);
  coffe_desc.SetDataType(ge::DT_FLOAT);
  coffe_desc.SetOriginFormat(input_desc0_origin_format);
  coffe_desc.SetOriginShape(coffeShapeOrigin);
  coffe_desc.SetOriginDataType(ge::DT_FLOAT);
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
  auto const_input_nodes = OpDescUtils::GetConstInputs(mul_node);
  NodePtr const_input = nullptr;
  if (const_input_nodes.size() != 0) {
    const_input = const_input_nodes[0];
  } else {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "const_input_nodes is null, please check!");
    return PARAM_INVALID;
  }
  const_input->GetOpDesc()->SetType(CONSTANTOP);

  return SUCCESS;
  OP_LOGI("Leave DepthwiseDwMulFusionPass::AddCoffe");
}

vector<FusionPattern*> DepthwiseDwMulFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // define AvgPoolFusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DepthwiseDwMulFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_DEPTHWISEDW, {DEPTHWISEDW}).SetOutput(PATTERN_DEPTHWISEDW);

  patterns.push_back(pattern);

  return patterns;
}

Status DepthwiseDwMulFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGI("Enter DepthwiseDwMulFusionPass");
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
  ge::Format input_origin_format = depthwise_dw_input_tensor.GetOriginFormat();
  ge::Format output_origin_format = depthwise_dw_output_tensor.GetOriginFormat();
  // GESHAPE->vector
  OP_LOGD("After get depthwise_dw_input_shape, depthwise_dw_output_shape, input_origin_format, output_origin_format");
  vector<int64_t> dim_info = depthwise_dw_input_shape.GetDims();
  vector<int64_t> out_dim_info = depthwise_dw_output_shape.GetDims();
  int64_t output_c1 = 0;
  int64_t output_n = 0;
  int64_t output_c = 0;
  int64_t output_h = 0;
  int64_t output_w = 0;
  int64_t output_w = 0;
  int64_t groups = 0;
  int64_t multiplier = 0;
  if (out_dim_info.size() == 4) {
    if (output_origin_format == FORMAT_NHWC) {
      output_n = out_dim_info[0];
      output_h = out_dim_info[1];
      output_w = out_dim_info[2];
      output_c = out_dim_info[3];
    } else if (output_origin_format == FORMAT_NCHW) {
      output_n = out_dim_info[0];
      output_h = out_dim_info[2];
      output_w = out_dim_info[3];
      output_c = out_dim_info[1];
    } else if (output_origin_format == FORMAT_HWCN) {
      output_n = out_dim_info[3];
      output_h = out_dim_info[0];
      output_w = out_dim_info[1];
      output_c = out_dim_info[2];
    } 
  } else {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "dim_info is not right, please check!");
    return NOT_CHANGED;
  }
  OP_LOGD("After get output n, H, W, C");
  // output originshapeformat n,h,w,c--->n*c,h,w,1
  depthwise_dw_output_tensor.SetOriginShape(ge::GeShape({output_h, output_w, 1, output_n*output_c}));
  depthwise_dw_desc->UpdateOutputDesc("filter_grad", depthwise_dw_output_tensor);

  OP_LOGD("After UpdateOutputDesc filter_grad, groups");
  ge::AttrUtils::GetInt(depthwise_dw_node->GetOpDesc(), "groups", groups);
  if (groups == 0) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "groups should not be 0.");
    return nullptr;
  }
  multiplier = output_c * output_n / groups;
  output_c1 = (groups + COUT - 1) / COUT;
  int64_t matrix_size = output_n * output_h * output_w * output_c;
  FUSION_PASS_CHECK(matrix_size <= 0, OP_LOGE(FUSED_OP_TYPE.c_str(), "matrix_size is Invalid"), return PARAM_INVALID);
  vector<int64_t> assit_dim_info_origin = {output_c1 * output_h * output_w, 1, COUT*multiplier, COUT};

  vector<int64_t>::iterator iter = assit_dim_info_origin.begin();
  for (; iter != assit_dim_info_origin.end(); iter++) {
    cout << *iter;
  }
  cout << endl;
  ge::NodePtr mul_node = AddMul(graph, depthwise_dw_node, output_origin_format);
  FUSION_PASS_CHECK(mul_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mul_node is null, AddMul failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(AddCoffe(graph, mul_node, matrix_size, assit_dim_info_origin) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "AddCoffe failed."), return PARAM_INVALID);
  OP_LOGI("Leave DepthwiseDwMulFusionPass");
  return SUCCESS;
}
REGISTER_PASS("DepthwiseDwMulFusionPass", BUILT_IN_GRAPH_PASS, DepthwiseDwMulFusionPass);
}  // namespace fe