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
 * \file avg_pool_grad_fusion_pass.cpp
 * \brief avg_pool_grad fusion pass(avg_pool_grad --> avg_pool_grad_d)
 */
#include "avg_pool_grad_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "error_util.h"

using namespace ge;
namespace fe {
static const std::string kPatternAvgPoolGrad = "AvgPoolGrad";
const std::string kConstantOp = "Constant";
static const int64_t COUT = 16;
static const int64_t CIN = 16;

Status AvgPoolGradFusionPass::WindowedOutputSize(const int32_t input, const int32_t k_size, const int32_t stride,
                                                 const string padding, int32_t& output, int32_t& pad_befor,
                                                 int32_t& pad_after) {
  int32_t tmp_output = 0;
  int32_t tmp_padneed = 0;
  int32_t tmp_pad_befor = 0;
  int32_t tmp_pad_after = 0;
  FUSION_PASS_CHECK(stride <= 0,
    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Stride less or equal than zero"), return FAILED);

  if (padding == "VALID") {
    tmp_output = (input - k_size + stride) / stride;
    tmp_pad_befor = 0;
    tmp_pad_after = 0;
  } else if (padding == "SAME") {
    tmp_output = (input + stride - 1) / stride;
    tmp_padneed = max(0, ((tmp_output - 1) * stride + k_size - input));
    tmp_pad_befor = tmp_padneed / 2;
    tmp_pad_after = tmp_padneed - tmp_pad_befor;
  } else {
    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "AvgPoolGrad padding arg not surport padding model");
    return FAILED;
  }
  output = tmp_output;
  pad_befor = tmp_pad_befor;
  pad_after = tmp_pad_after;
  return SUCCESS;
}

Status AvgPoolGradFusionPass::TransposeNCHW2NHWC(const int32_t n_output, const int32_t h_output, const int32_t w_output,
                                                 const int32_t c_output, uint16_t* avgpoolout) {
  uint64_t len = static_cast<uint64_t>(n_output) * static_cast<uint64_t>(h_output) * static_cast<uint64_t>(w_output) *
                 static_cast<uint64_t>(c_output);
  FUSION_PASS_CHECK((len > INT_MAX) || (len <= 0),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Cannot malloc too large memory!"),
                    return FAILED);

  uint16_t* tmp = new (std::nothrow) uint16_t[len];
  FUSION_PASS_CHECK(tmp == nullptr,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Run malloc memory failed!"), return FAILED);
  for (int32_t n = 0; n < n_output; n++) {
    for (int32_t h = 0; h < h_output; h++) {
      for (int32_t w = 0; w < w_output; w++) {
        for (int32_t c = 0; c < c_output; c++) {
          tmp[n * h_output * w_output * c_output + h * w_output * c_output + w * c_output + c] =
              avgpoolout[n * c_output * h_output * w_output + c * h_output * w_output + h * w_output + w];
        }
      }
    }
  }
  errno_t ret = memcpy_s(avgpoolout, len * sizeof(uint16_t), tmp, len * sizeof(uint16_t));
  if (ret != EOK) {
    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Run memcpy_s fail!");
    delete[] tmp;
    return FAILED;
  }
  delete[] tmp;
  return SUCCESS;
}

Status AvgPoolGradFusionPass::AvgValueTableGen(const vector<int64_t> dim_info, const vector<int64_t> k_size,
                                               const vector<int64_t> strides, const string padding,
                                               const string data_format, vector<int64_t>& assit_dim_info,
                                               uint16_t* output) {
  // The caller can guarantee dim_info, k_size, strides have 4 elements
  int64_t n_input = dim_info[0];
  int64_t c_input = dim_info[3];
  int64_t h_input = dim_info[1];
  int64_t w_input = dim_info[2];
  int64_t h_ksize;
  int64_t w_ksize;
  int64_t h_stride;
  int64_t w_stride;

  if ((k_size[0] == 1) || (k_size[3] == 1)) {
    h_ksize = k_size[1];
    w_ksize = k_size[2];
  } else {
    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "AvgPoolGrad ksize error");
    return FAILED;
  }
  if ((strides[0] == 1) || (strides[3] == 1)) {
    h_stride = strides[1];
    w_stride = strides[2];
  } else {
    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "AvgPoolGrad strides arg error");
    return FAILED;
  }
  int32_t n_output = n_input;
  int32_t c_output = c_input;

  int32_t h_output = 0;
  int32_t pad_top = 0;
  int32_t pad_bottom = 0;
  FUSION_PASS_CHECK(WindowedOutputSize(h_input, h_ksize, h_stride, padding, h_output, pad_top, pad_bottom) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "WindowedOutputSize failed"), return FAILED);
  int32_t w_output = 0;
  int32_t pad_left = 0;
  int32_t pad_right = 0;
  int32_t add_flag_h = 0;
  int32_t add_flag_w = 0;
  FUSION_PASS_CHECK(WindowedOutputSize(w_input, w_ksize, w_stride, padding, w_output, pad_left, pad_right) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "WindowedOutputSize failed"), return FAILED);
  int64_t out_offset_point = 0;
  for (int n = 0; n < n_output; n++) {
    for (int c = 0; c < c_output; c++) {
      for (int h = 0; h < h_output; h++) {
        for (int w = 0; w < w_output; w++) {
          for (int hk = 0; hk < h_ksize; hk++) {
            for (int wk = 0; wk < w_ksize; wk++) {
              add_flag_h = 0;
              add_flag_w = 0;
              out_offset_point = n * c_output * h_output * w_output + c * h_output * w_output + h * w_output + w;
              if ((pad_top <= (h * h_stride + hk)) && ((h * h_stride + hk - pad_top) < h_input)) {
                add_flag_h = 1;
              }
              if ((pad_left <= (w * w_stride + wk)) && ((w * w_stride + wk - pad_left) < w_input)) {
                add_flag_w = 1;
              }
              if ((add_flag_h == 1) && (add_flag_w == 1)) {
                output[out_offset_point] += 1;
              }
            }
          }
          fp16_t tmp;
          tmp.val = output[out_offset_point];
          fp16_t tmp2;
          tmp2.val = 0;
          tmp2 = 1 / (float)tmp.val;
          output[out_offset_point] = tmp2.val;
        }
      }
    }
  }
  if (data_format == "NHWC") {
    FUSION_PASS_CHECK(TransposeNCHW2NHWC(n_output, h_output, w_output, c_output, output) != SUCCESS,
                      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "TransposeNCHW2NHWC failed"), return FAILED);
  }
  if (data_format == "NHWC") {
    assit_dim_info = {n_output, h_output, w_output, c_output};
  } else if (data_format == "NCHW") {
    assit_dim_info = {n_output, c_output, h_output, w_output};
  }
  return SUCCESS;
}

Status KernelGen(const int32_t h_ksize, const int32_t w_ksize, const int32_t c_input, uint16_t* kernel_table,
                 int64_t kernel_table_size) {
  // this 6D is not safe ,because gragh donot know this info
  // from depthwise, filter is HWNC, but ge get shape by NHWC, so, plugin set format HWNC.
  int64_t len = static_cast<int64_t>(h_ksize) * static_cast<int64_t>(w_ksize) * static_cast<int64_t>(c_input);
  FUSION_PASS_CHECK(len > kernel_table_size,
                    CUBE_INNER_ERR_REPORT(kPatternAvgPoolGrad.c_str(), "Access kernel_table_size overflow."),
                    return FAILED);
  fp16_t tmp;
  tmp.val = 0;
  for (int64_t i = 0; i < len; i++) {
    tmp.val = 1.0;
    fp16_t tmp2;
    tmp2.val = 0;
    tmp2 = (float)tmp.val;
    kernel_table[i] = tmp2.val;
  }
  return SUCCESS;
}

Status KernelGenDynamic(const vector<int64_t> shape, const float areaFactor, uint16_t& output1) {
  uint16_t* output = &output1;
  fp16_t area_factor;
  area_factor.val = 0;
  area_factor = static_cast<float>(areaFactor);
  for (int64_t i = 0; i < shape[0]; i++) {
    for (int64_t j = 0; j < shape[1]; j++) {
      for (int64_t k = 0; k < shape[2]; k++) {
        for (int64_t l = 0; l < shape[3]; l++) {
          if (k == l) {
            output[i * (shape[1] * shape[2] * shape[3]) + j * (shape[2] * shape[3]) + k * shape[3] + l] =
                                                                                                      area_factor.val;
          }
        }
      }
    }
  }
  return SUCCESS;
}

vector<FusionPattern*> AvgPoolGradFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("AvgPoolGradFusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    CUBE_CALL_ERR_REPORT(kFusedOpType.c_str(), "new a pattern object failed."), return patterns);
  pattern->AddOpDesc(kPatternAvgPoolGrad, {"AvgPoolGrad"}).SetOutput(kPatternAvgPoolGrad);
  patterns.push_back(pattern);
  return patterns;
}

// vector<NodePtr> &fusion_nodes: Store fusion nodes,
// including newly added nodes and fused but not deleted nodes
Status AvgPoolGradFusionPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& fusion_nodes) {
  std::string fusion_op_type = "AvgPoolGradD";
  std::map<int16_t, std::string> avg_pool_grad_attr_info;
  avg_pool_grad_attr_info[0] = "orig_input_shape";
  PatternFusionUtil pattern_fusion_util;
  // get node pointer
  NodePtr avg_pool_grad_fused_node = GetNodeFromMapping(kPatternAvgPoolGrad, mapping);
  FUSION_PASS_CHECK(avg_pool_grad_fused_node == nullptr,
                    CUBE_CALL_ERR_REPORT(kFusedOpType.c_str(),
                                          "Pointer avg_pool_grad_fused_node is null, fusion failed."),
                    return PARAM_INVALID);
  // get opdesc pointer
  OpDescPtr avg_pool_grad_desc = avg_pool_grad_fused_node->GetOpDesc();
  FUSION_PASS_CHECK(avg_pool_grad_desc == nullptr,
                    CUBE_CALL_ERR_REPORT(kFusedOpType.c_str(), "Pointer avg_pool_grad_desc is null, fusion failed."),
                    return PARAM_INVALID);
  string data_format;
  vector<int64_t> k_size;
  vector<int64_t> strides;
  string padding;

  // get ksize padding strides value data_format
  // data_format is optional
  AttrUtils::GetStr(avg_pool_grad_desc, "data_format", data_format);
  FUSION_PASS_CHECK(!AttrUtils::GetListInt(avg_pool_grad_desc, "ksize", k_size),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Get attr ksize failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(!AttrUtils::GetListInt(avg_pool_grad_desc, "strides", strides),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Get attr strides failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(!AttrUtils::GetStr(avg_pool_grad_desc, "padding", padding),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Get attr padding failed."), return PARAM_INVALID);

  // get const org_input_shape desc, dtype, format, dims
  InDataAnchorPtr avg_pool_grad_const_anchor_ptr0 = avg_pool_grad_fused_node->GetInDataAnchor(0);
  FUSION_PASS_CHECK(avg_pool_grad_const_anchor_ptr0 == nullptr,
                    CUBE_CALL_ERR_REPORT(kFusedOpType.c_str(),
                                         "Pointer avg_pool_grad_const_anchor_ptr0 is null, fusion failed."),
                    return PARAM_INVALID);
  OutDataAnchorPtr const_anchor_ptr = avg_pool_grad_const_anchor_ptr0->GetPeerOutAnchor();
  FUSION_PASS_CHECK(const_anchor_ptr == nullptr,
                    CUBE_CALL_ERR_REPORT(kFusedOpType.c_str(), "Pointer const_anchor_ptr is null, fusion failed."),
                    return PARAM_INVALID);
  NodePtr const_node = const_anchor_ptr->GetOwnerNode();
  auto const_node_desc = const_node->GetOpDesc();
  FUSION_PASS_CHECK(const_node_desc == nullptr,
                    CUBE_CALL_ERR_REPORT(kFusedOpType.c_str(), "Pointer const_node_desc is null, fusion failed."),
                    return PARAM_INVALID);
  GeTensorDesc org_input_shape_tensor = const_node_desc->GetOutputDesc(0);
  DataType data_type = org_input_shape_tensor.GetDataType();
  GeShape org_input_shape = org_input_shape_tensor.GetShape();
  int64_t dim_nums = org_input_shape.GetShapeSize();
  FUSION_PASS_CHECK(dim_nums != 4, OP_LOGW(kFusedOpType.c_str(), "The org_input_shape dim_nums must be 4."),
                    return NOT_CHANGED);
  vector<int64_t> dim_info = org_input_shape.GetDims();
  // get orig_input_shape value
  Operator op = OpDescUtils::CreateOperatorFromNode(avg_pool_grad_fused_node);
  Tensor orig_input_shape_const_tensor;
  op.GetInputConstData(avg_pool_grad_attr_info[0], orig_input_shape_const_tensor);
  FUSION_PASS_CHECK(data_type != DT_INT32,
                    OP_LOGW(kFusedOpType.c_str(), "The orig_input_shape dtype only surpport INT32."),
                    return NOT_CHANGED);
  int32_t* orig_input_shape_const_tensor_ptr = (int32_t*)orig_input_shape_const_tensor.GetData();
  FUSION_PASS_CHECK(PatternFusionUtil::IsUnknownShape(dim_info[0]),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                          "AvgPoolGradFusionPass cannot be applied for unknown shape."),
                    return GRAPH_FAILED);
  FUSION_PASS_CHECK(dim_info[0] != 4, OP_LOGW(kFusedOpType.c_str(), "The orig_input_shape must be list of 4."),
                    return NOT_CHANGED);

  // gen avgtable matrix
  auto avg_pool_grad_fused_node_desc = avg_pool_grad_fused_node->GetOpDesc();
  FUSION_PASS_CHECK(avg_pool_grad_fused_node_desc == nullptr,
                    CUBE_CALL_ERR_REPORT(kFusedOpType.c_str(), "Pointer const_node_desc is null, fusion failed."),
                    return PARAM_INVALID);
  GeTensorDesc avg_pool_input_shape_tensor = avg_pool_grad_fused_node_desc->GetInputDesc(1);
  GeShape avg_pool_shape = avg_pool_input_shape_tensor.GetShape();
  vector<int64_t> avg_pool_dim_info = avg_pool_shape.GetDims();
  if (avg_pool_dim_info.size() == 1 && avg_pool_dim_info[0] == -2) {
    avg_pool_dim_info.push_back(-1);
    avg_pool_dim_info.push_back(-1);
    avg_pool_dim_info.push_back(-1);
  }
  FUSION_PASS_CHECK(avg_pool_dim_info.size() < 4, OP_LOGW(kFusedOpType.c_str(), "Dims must great than 3"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(k_size.size() != 4, OP_LOGW(kFusedOpType.c_str(), "The k_size must list of 4 element."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(strides.size() != 4, OP_LOGW(kFusedOpType.c_str(), "The strides must list of 4 element."),
                    return NOT_CHANGED);

  if (data_format == "NHWC") {
    OP_LOGI(kFusedOpType.c_str(), "AvgPoolGrad data_format NHWC.");
    FUSION_PASS_CHECK((k_size[0] != 1) || (k_size[3] != 1),
                      OP_LOGW(kFusedOpType.c_str(), "AvgPoolGrad NHWC,ksize only surpport ksize[0]==ksize[3]==1."),
                      return NOT_CHANGED);
    FUSION_PASS_CHECK((strides[0] != 1) || (strides[3] != 1),
                      OP_LOGW(kFusedOpType.c_str(), "AvgPoolGrad NHWC stride only surpport strides[0]==strides[3]==1."),
                      return NOT_CHANGED);
  } else if (data_format == "NCHW") {
    OP_LOGI(kFusedOpType.c_str(), "AvgPoolGrad data_format NCHW.");
    int64_t k_size_h = 0;
    int64_t k_size_w = 0;
    int64_t stride_h = 0;
    int64_t stride_w = 0;
    int64_t orig_input_shape_h = 0;
    int64_t orig_input_shape_w = 0;
    int64_t orig_input_shape_c = 0;
    int64_t output_shape_h = 0;
    int64_t output_shape_w = 0;
    int64_t output_shape_c = 0;
    FUSION_PASS_CHECK((k_size[0] != 1) || (k_size[1] != 1),
                      OP_LOGW(kFusedOpType.c_str(), "AvgPoolGrad NCHW, stride only surpport ksize[0]==ksize[3]==1."),
                      return NOT_CHANGED);
    k_size_h = k_size[2];
    k_size_w = k_size[3];
    k_size[0] = 1;
    k_size[1] = k_size_h;
    k_size[2] = k_size_w;
    k_size[3] = 1;
    FUSION_PASS_CHECK(
        (strides[0] != 1) || (strides[1] != 1),
        OP_LOGW(kFusedOpType.c_str(), "AvgPoolGrad NCHW, stride only surpport strides[0]==strides[1]==1."),
        return NOT_CHANGED);
    stride_h = strides[2];
    stride_w = strides[3];
    strides[0] = 1;
    strides[1] = stride_h;
    strides[2] = stride_w;
    strides[3] = 1;
    orig_input_shape_h = orig_input_shape_const_tensor_ptr[2];
    orig_input_shape_w = orig_input_shape_const_tensor_ptr[3];
    orig_input_shape_c = orig_input_shape_const_tensor_ptr[1];
    orig_input_shape_const_tensor_ptr[1] = orig_input_shape_h;
    orig_input_shape_const_tensor_ptr[2] = orig_input_shape_w;
    orig_input_shape_const_tensor_ptr[3] = orig_input_shape_c;
    output_shape_h = avg_pool_dim_info[2];
    output_shape_w = avg_pool_dim_info[3];
    output_shape_c = avg_pool_dim_info[1];
    avg_pool_dim_info[1] = output_shape_h;
    avg_pool_dim_info[2] = output_shape_w;
    avg_pool_dim_info[3] = output_shape_c; 
  }
  vector<int64_t> orig_input_shape_v{orig_input_shape_const_tensor_ptr[0], orig_input_shape_const_tensor_ptr[1],
                                     orig_input_shape_const_tensor_ptr[2], orig_input_shape_const_tensor_ptr[3]};
  bool is_dynamic = false;
  for (size_t i = 0; i < orig_input_shape_v.size(); i++) {
    auto dim = orig_input_shape_const_tensor_ptr[i];
    if (dim <= 0) {
      is_dynamic = true;
    }
  }
  if (is_dynamic && avg_pool_dim_info[0] == -2) {
    OP_LOGW(kFusedOpType.c_str(), "AvgPoolGradFusionPass not support input_grad is [-2]");
    return NOT_CHANGED;
  }
  Status ret;
  if (!is_dynamic) {
    GeTensorPtr avg_table_assit_ptr = nullptr;
    bool is_dynamic_dim_info = false;
    for (size_t i = 0; i < avg_pool_dim_info.size(); i++) {
      if (PatternFusionUtil::IsUnknownShape(avg_pool_dim_info[i])) {
        is_dynamic_dim_info = true;
      }
    }
    if (is_dynamic_dim_info) {
      avg_pool_dim_info.resize(4);
      if (padding == "SAME") {
        avg_pool_dim_info[0] = orig_input_shape_const_tensor_ptr[0];
        avg_pool_dim_info[3] = orig_input_shape_const_tensor_ptr[3];
        avg_pool_dim_info[1] = (orig_input_shape_const_tensor_ptr[1] + strides[1] - 1) / strides[1];
        avg_pool_dim_info[2] = (orig_input_shape_const_tensor_ptr[2] + strides[2] - 1) / strides[2];

      } else {
        avg_pool_dim_info[0] = orig_input_shape_const_tensor_ptr[0];
        avg_pool_dim_info[1] =
          (orig_input_shape_const_tensor_ptr[1] - k_size[1] + 1 + (strides[1] - 1)) / strides[1];
        avg_pool_dim_info[2] =
          (orig_input_shape_const_tensor_ptr[2] - k_size[2] + 1 + (strides[2] - 1)) / strides[2];
        avg_pool_dim_info[3] = orig_input_shape_const_tensor_ptr[3];
      }
    }
    int64_t value_table_size = avg_pool_dim_info[0] * avg_pool_dim_info[1] * avg_pool_dim_info[2] * avg_pool_dim_info[3];

    FUSION_PASS_CHECK(
        (((avg_pool_dim_info[1] * avg_pool_dim_info[2] * avg_pool_dim_info[3]) == 0) || (value_table_size <= 0)),
        OP_LOGW(kFusedOpType.c_str(), "The value_table_size have 0 element"), return NOT_CHANGED);
    FUSION_PASS_CHECK(
        (avg_pool_dim_info[0] != value_table_size / (avg_pool_dim_info[1] * avg_pool_dim_info[2] * avg_pool_dim_info[3])),
        OP_LOGW(kFusedOpType.c_str(), "The value_table_size overlap , over int64"), return NOT_CHANGED);

    unique_ptr<uint16_t[]> input_assit(new (std::nothrow) uint16_t[value_table_size]());
    FUSION_PASS_CHECK(input_assit.get() == nullptr,
                      CUBE_CALL_ERR_REPORT(kFusedOpType.c_str(), "The input_assit is NULL"),
                      return PARAM_INVALID);
    vector<int64_t> avg_pool_assit_dim_info;
    ret = AvgValueTableGen(orig_input_shape_v, k_size, strides, padding, data_format, avg_pool_assit_dim_info,
                                  input_assit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "AssitHelp failed."), return ret);
    GeShape avg_pool_assit_shape(avg_pool_assit_dim_info);
    GeTensorDesc tensor_desc(GeShape(), FORMAT_NHWC, DT_FLOAT16);
    if (data_format == "NHWC") {
      tensor_desc.SetFormat(FORMAT_NHWC);
      tensor_desc.SetOriginFormat(FORMAT_NHWC);
    } else if (data_format == "NCHW") {
      tensor_desc.SetFormat(FORMAT_NCHW);
      tensor_desc.SetOriginFormat(FORMAT_NCHW);
    }
    tensor_desc.SetShape(avg_pool_assit_shape);
    tensor_desc.SetOriginShape(avg_pool_assit_shape);

    FUSION_PASS_MAKE_SHARED(
        (avg_table_assit_ptr = std::make_shared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t*>(input_assit.get()),
                                                          value_table_size * sizeof(uint16_t))),
        avg_table_assit_ptr = nullptr;
        return PARAM_INVALID);
    vector<GeTensorPtr> avg_pool_grad_weights = {avg_table_assit_ptr};
    ret = OpDescUtils::SetWeights(avg_pool_grad_fused_node, avg_pool_grad_weights);
    FUSION_PASS_CHECK(ret != GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add mean matrix failed."),
                      return ret);
    auto avg_pool_const_input_nodes = OpDescUtils::GetConstInputs(avg_pool_grad_fused_node);
    NodePtr avg_pool_const_input = avg_pool_const_input_nodes[0];
    auto avg_pool_const_input_desc = avg_pool_const_input->GetOpDesc();
    FUSION_PASS_CHECK(avg_pool_const_input_desc == nullptr,
                      CUBE_CALL_ERR_REPORT(kFusedOpType.c_str(), "The avg_pool_const_input_desc is NULL"),
                      return PARAM_INVALID);
    avg_pool_const_input_desc->SetType(kConstantOp);
  }

  // gen kernel matrix
  // orig_input_shape_v[1] must be channel
  int64_t kernel_table_size = orig_input_shape_v[3] * k_size[1] * k_size[2];
  int64_t inputC1 = 0;  
  if (is_dynamic) {
    inputC1 = (avg_pool_dim_info[3] + COUT - 1) / COUT;  
    kernel_table_size = COUT * inputC1 * CIN * k_size[1] * k_size[2];
    FUSION_PASS_CHECK((inputC1 != kernel_table_size / (COUT * CIN * k_size[1] * k_size[2])),
                  OP_LOGW(kFusedOpType.c_str(), "The kernel_table_size overlap , over int64"), return NOT_CHANGED);
    OP_LOGI(kFusedOpType.c_str(), "kernel_table_size is %d", kernel_table_size);
  } else {
    kernel_table_size = orig_input_shape_v[3] * k_size[1] * k_size[2];
    FUSION_PASS_CHECK((orig_input_shape_v[3] != kernel_table_size / (k_size[1] * k_size[2])),
                OP_LOGW(kFusedOpType.c_str(), "The kernel_table_size overlap , over int64"), return NOT_CHANGED);
    OP_LOGI(kFusedOpType.c_str(), "kernel_table_size is %d", kernel_table_size);
  }
  FUSION_PASS_CHECK((((k_size[1] * k_size[2]) == 0) || (kernel_table_size <= 0)),
                    OP_LOGW(kFusedOpType.c_str(), "The kernel_table_size have O element"), return NOT_CHANGED);

  unique_ptr<uint16_t[]> kernel_table_input_assit(new (std::nothrow) uint16_t[kernel_table_size]());
  FUSION_PASS_CHECK(kernel_table_input_assit.get() == nullptr,
                    CUBE_CALL_ERR_REPORT(kFusedOpType.c_str(), "The kernel_table_input_assit is NULL"),
                    return PARAM_INVALID);
  vector<int64_t> assit_dim_info_dynamic = {inputC1 * k_size[1] *k_size[2], 1, CIN, COUT};
  if (!is_dynamic) {
    ret = KernelGen(k_size[1], k_size[2], orig_input_shape_v[3], kernel_table_input_assit.get(), kernel_table_size);
  } else {
    float areaFactor = 1.0;
    ret = KernelGenDynamic(assit_dim_info_dynamic, areaFactor, *kernel_table_input_assit.get());
  }
  FUSION_PASS_CHECK(ret != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "The kernel_table matrix AssitHelp failed."),
                    return ret);
  vector<int64_t> kernel_table_assit_dim_info{k_size[1], k_size[2], orig_input_shape_v[3], 1LL};
  vector<int64_t> kernel_table_assit_dim_info_dynamic{k_size[1], k_size[2], 1LL, avg_pool_dim_info[3]};
  GeTensorDesc kernel_table_tensor_desc(GeShape(), FORMAT_HWCN, DT_FLOAT16);  
  if (!is_dynamic) {
    OP_LOGI(kFusedOpType.c_str(), "not dynamic");
    GeShape assit_shape(kernel_table_assit_dim_info);
    kernel_table_tensor_desc.SetShape(assit_shape);
    kernel_table_tensor_desc.SetOriginShape(assit_shape);
    kernel_table_tensor_desc.SetOriginFormat(FORMAT_HWCN);
    kernel_table_tensor_desc.SetFormat(FORMAT_HWCN);
  } else {
    OP_LOGI(kFusedOpType.c_str(), "dynamic");
    GeShape assit_ori_shape(kernel_table_assit_dim_info_dynamic);
    GeShape assit_shape(assit_dim_info_dynamic);
    kernel_table_tensor_desc.SetShape(assit_shape);
    kernel_table_tensor_desc.SetOriginShape(assit_ori_shape);
    kernel_table_tensor_desc.SetOriginFormat(FORMAT_HWCN);
    kernel_table_tensor_desc.SetFormat(FORMAT_FRACTAL_Z);
  }
  GeTensorPtr kernel_table_assit_ptr = nullptr;
  FUSION_PASS_MAKE_SHARED((kernel_table_assit_ptr = std::make_shared<GeTensor>(
                               kernel_table_tensor_desc, reinterpret_cast<uint8_t*>(kernel_table_input_assit.get()),
                               kernel_table_size * sizeof(uint16_t))),
                          kernel_table_assit_ptr = nullptr;
                          return PARAM_INVALID);
  vector<GeTensorPtr> kernel_weights = {kernel_table_assit_ptr};
  ret = OpDescUtils::SetWeights(avg_pool_grad_fused_node, kernel_weights);
  FUSION_PASS_CHECK(ret != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add kernel matrix failed."), return ret);
  auto kernel_const_input_nodes = OpDescUtils::GetConstInputs(avg_pool_grad_fused_node);
  NodePtr kernel_const_input = kernel_const_input_nodes[0];
  auto kernel_const_input_desc = kernel_const_input->GetOpDesc();
  FUSION_PASS_CHECK(kernel_const_input_desc == nullptr,
                    CUBE_CALL_ERR_REPORT(kFusedOpType.c_str(), "The kernel_const_input_desc is NULL"),
                    return PARAM_INVALID);
  kernel_const_input_desc->SetType(kConstantOp);

  if (!is_dynamic) {
    NodePtr fusion_node = nullptr;
    PassAttrInfo orig_input_shape{0, "orig_input_shape", "SetListInt"};
    std::vector<PassAttrInfo> avg_pool_grad_pass_info{orig_input_shape};
    // const org input shape change to attr
    ret = pattern_fusion_util.ConstToAttrWithNode(graph, avg_pool_grad_fused_node, fusion_op_type,
                                                  avg_pool_grad_pass_info, fusion_node);
    FUSION_PASS_CHECK(ret != SUCCESS,
                      OP_LOGW(kFusedOpType.c_str(), "AvgPoolGrad has input which is not a CONST, graph not changed."),
                      return NOT_CHANGED);
    fusion_nodes.push_back(fusion_node);
  }
  return SUCCESS;
}
REGISTER_PASS("AvgPoolGradFusionPass", BUILT_IN_GRAPH_PASS, AvgPoolGradFusionPass);
}  // namespace fe
