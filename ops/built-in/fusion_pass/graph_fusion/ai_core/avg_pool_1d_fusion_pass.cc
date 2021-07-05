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
 * \file avg_pool_1d_fusion_pass.cpp
 * \brief avg_pool_1d fusion pass(avg_pool_1d --> avg_pool_1dD)
 */
#include "avg_pool_1d_fusion_pass.h"

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
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
static const string kPatternAvgPool1D = "AvgPool1D";
static const string kConstantOp = "Constant";
constexpr int64_t kC0{16};

void SetAssitInfo(const int64_t n_input, const int64_t c1_input, const int64_t h_input, const int64_t w_in_input,
                  const int64_t c0_input, vector<int64_t>& assit_dim_info) {
  assit_dim_info = {n_input, c1_input, h_input, w_in_input, c0_input};
}

int64_t CalWoutput(const int64_t w_in_input, const int64_t padl, const int64_t padr, const int64_t k_size,
                   const int64_t stride, const bool ceil_mode) {
  int64_t res{0};
  if (ceil_mode) {
    res = (w_in_input + padl + padr - k_size + stride - 1) / stride + 1;
  } else {
    res = ((w_in_input + padl + padr) - k_size) / stride + 1;
  }
  if (padl) {
    // ensure that the last pooling starts inside the image needed to avoid problems in ceil mode existing bug in
    // pytorch code padl = 0 and stride is big, but kernel is small, return nan
    if (((res - 1) * stride) >= (w_in_input + padl)) {
      res--;
    }
  }
  return res;
}

template <typename T>
Status AvgPool1DFusionPass::AvgValueTableGen(const vector<int64_t>& dim_info, const int64_t kernel_size,
                                             const int64_t stride_size, const vector<int64_t>& padding,
                                             const bool ceil_mode, const bool count_include_pad,
                                             vector<int64_t>& assit_dim_info, T* output) {
  FUSION_PASS_CHECK(output == nullptr, OP_LOGE(kFusedOpType.c_str(), "The output pointer is null!"), return FAILED);
  FUSION_PASS_CHECK(dim_info.size() < 4, OP_LOGE(kFusedOpType.c_str(), "The dim_info at least has 4 elements!"),
                    return FAILED);
  FUSION_PASS_CHECK(padding.size() < 2, OP_LOGE(kFusedOpType.c_str(), "The padding at least has 2 elements!"),
                    return FAILED);

  int64_t n_input{1};
  int64_t c1_input{1};
  int64_t h_input{1};
  // dim_info must NCHW
  int64_t w_in_input = dim_info[3];
  int64_t c0_input = kC0;
  int64_t k_size = kernel_size;
  int64_t stride = stride_size;
  int64_t padl = padding[0];
  int64_t padr = padding[1];
  int64_t n_output = n_input;
  int64_t c1_output = c1_input;
  int64_t h_output = h_input;
  int64_t w_output = 0;
  int64_t c0_output = c0_input;

  FUSION_PASS_CHECK(stride == 0, OP_LOGE(kFusedOpType.c_str(), "The stride should not be 0, fusion failed."),
                    return FAILED);
  w_output = CalWoutput(w_in_input, padl, padr, k_size, stride, ceil_mode);
  SetAssitInfo(n_output, c1_output, h_output, w_output, c0_output, assit_dim_info);

  // set output data
  float data_num{0.0};
  int64_t start{0};
  int64_t end{0};
  int64_t out_offset_point{0};
  for (int64_t n = 0; n < n_output; n++) {
    for (int64_t c1 = 0; c1 < c1_output; c1++) {
      for (int64_t h = 0; h < h_output; h++) {
        for (int64_t w = 0; w < w_output; w++) {
          start = stride * w;
          end = stride * w + k_size;
          if (!count_include_pad) {
            start = max(start, padl);
            end = min(end, w_in_input + padl);
          } else {
            end = min(end, w_in_input + padl + padr);
          }
          data_num = end - start;
          if (data_num == 0) {
            OP_LOGE("divied by zero error.");
            return FAILED;
          }
          if (typeid(output) == typeid(float*)) {
            float tmp;
            tmp = 1.0 / data_num;
            for (int64_t c0 = 0; c0 < c0_output; c0++) {
              out_offset_point = n * c1_output * h_output * w_output * c0_output +
                                 c1 * h_output * w_output * c0_output + h * w_output * c0_output + w * c0_output + c0;
              output[out_offset_point] = tmp;
            }
          } else if (typeid(output) == typeid(uint16_t*)) {
            fp16_t tmp;
            tmp = 1.0 / data_num;
            for (int64_t c0 = 0; c0 < c0_output; c0++) {
              out_offset_point = n * c1_output * h_output * w_output * c0_output +
                                 c1 * h_output * w_output * c0_output + h * w_output * c0_output + w * c0_output + c0;
              output[out_offset_point] = tmp.val;
            }
          } else {
            OP_LOGE(kFusedOpType.c_str(), "Check data type, output is zero!");
            return FAILED;
          }
        }
      }
    }
  }
  return SUCCESS;
}

vector<FusionPattern*> AvgPool1DFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (nothrow) FusionPattern("AvgPool1DFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "New a pattern object failed."), return patterns);
  pattern->AddOpDesc(kPatternAvgPool1D, {"AvgPool1D"}).SetOutput(kPatternAvgPool1D);
  patterns.push_back(pattern);
  return patterns;
}

Status AvgPool1DFusionPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& fusion_nodes) {
  // avg_pool_1d node
  NodePtr avgpool1d_fussed_node = GetNodeFromMapping(kPatternAvgPool1D, mapping);
  FUSION_PASS_CHECK(avgpool1d_fussed_node == nullptr,
                    OP_LOGE(kFusedOpType.c_str(), "The avgpool1d_fussed_node is null, fusion failed."),
                    return PARAM_INVALID);

  // input of avg_pool_1d
  OpDescPtr avgpool1d_desc = avgpool1d_fussed_node->GetOpDesc();
  FUSION_PASS_CHECK(avgpool1d_desc == nullptr,
                    OP_LOGE(kFusedOpType.c_str(), "The avgpool1d_fussed_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  int64_t k_size;
  int64_t strides;
  vector<int64_t> pads;
  bool ceil_mode{false};
  bool count_include_pad{false};

  // get ksize pads strides value ceil_mode count_include_pad
  FUSION_PASS_CHECK(!AttrUtils::GetInt(avgpool1d_desc, "ksize", k_size),
                    OP_LOGE(kFusedOpType.c_str(), "Get attr ksize failed."), return FAILED);
  FUSION_PASS_CHECK(!AttrUtils::GetInt(avgpool1d_desc, "strides", strides),
                    OP_LOGE(kFusedOpType.c_str(), "Get attr strides failed."), return FAILED);
  FUSION_PASS_CHECK(!AttrUtils::GetListInt(avgpool1d_desc, "pads", pads),
                    OP_LOGE(kFusedOpType.c_str(), "Get attr pads failed."), return FAILED);
  // Attr ceil_mode and count_include_pad are optional
  AttrUtils::GetBool(avgpool1d_desc, "ceil_mode", ceil_mode);
  AttrUtils::GetBool(avgpool1d_desc, "count_include_pad", count_include_pad);
  // get const input_shape desc, dtype, format, dims
  Operator op = OpDescUtils::CreateOperatorFromNode(avgpool1d_fussed_node);

  // gen avgtable matrix
  auto avgpool1d_op_desc = avgpool1d_fussed_node->GetOpDesc();
  FUSION_PASS_CHECK(avgpool1d_op_desc == nullptr,
                    OP_LOGE(kFusedOpType.c_str(), "The opdesc of avgpool1d_fussed_node is nullptr."),
                    return PARAM_INVALID);
  GeTensorDesc avgpool1d_input_shape_tensor = avgpool1d_op_desc->GetInputDesc(0);

  DataType input_type = avgpool1d_input_shape_tensor.GetDataType();
  FUSION_PASS_CHECK((input_type != DT_FLOAT16 && input_type != DT_FLOAT),
                    OP_LOGW(kFusedOpType.c_str(), "matrix only support float16 and float32"), return NOT_CHANGED);
  GeShape avgpool1d_shape = avgpool1d_input_shape_tensor.GetShape();
  vector<int64_t> avgpool1d_dim_info = avgpool1d_shape.GetDims();
  FUSION_PASS_CHECK(avgpool1d_dim_info.size() < 4,
                    OP_LOGE(kFusedOpType.c_str(), "The size of avgpool1d_dim_info should great than 3, fusion failed."),
                    return FAILED);
  Format input_format = avgpool1d_input_shape_tensor.GetFormat();
  FUSION_PASS_CHECK(pads.size() != 2, OP_LOGW(kFusedOpType.c_str(), "Pads must list of 2 elements."),
                    return NOT_CHANGED);
  if (input_format == FORMAT_NCHW) {
    OP_LOGI(kFusedOpType.c_str(), "AvgPool1D input_format NCHW.");
  } else if (input_format == FORMAT_NHWC) {
    OP_LOGI(kFusedOpType.c_str(), "AvgPool1D input_format NHWC.");
    int64_t orig_input_shape_h = avgpool1d_dim_info[1];
    int64_t orig_input_shape_w = avgpool1d_dim_info[2];
    int64_t orig_input_shape_c = avgpool1d_dim_info[3];
    avgpool1d_dim_info[1] = orig_input_shape_c;
    avgpool1d_dim_info[2] = orig_input_shape_h;
    avgpool1d_dim_info[3] = orig_input_shape_w;
  }
  vector<int64_t> orig_input_shape_v{avgpool1d_dim_info[0], avgpool1d_dim_info[1], avgpool1d_dim_info[2],
                                     avgpool1d_dim_info[3]};
  // orig_input_shape_v must NCHW
  GeTensorPtr avg_table_assit_ptr{nullptr};
  FUSION_PASS_CHECK(strides == 0, OP_LOGE(kFusedOpType.c_str(), "The stride should not be 0, fusion failed."),
                    return PARAM_INVALID);
  for (size_t i = 1; i <= 3; i++) {
    auto dim = avgpool1d_dim_info[i];
    FUSION_PASS_CHECK(PatternFusionUtil::IsUnknownShape(dim),
                      OP_LOGE(kFusedOpType.c_str(), "AvgPool1DFusionPass cannot be applied for unknown shape."),
                      return GRAPH_FAILED);
  }
  int64_t w_output = CalWoutput(avgpool1d_dim_info[3], pads[0], pads[1], k_size, strides, ceil_mode);
  FUSION_PASS_CHECK(w_output <= 0, OP_LOGE(kFusedOpType.c_str(), "Should keep w_output > 0!"), return GRAPH_FAILED);
  int64_t value_table_size = w_output * kC0;

  FUSION_PASS_CHECK(
      (((avgpool1d_dim_info[1] * avgpool1d_dim_info[2] * avgpool1d_dim_info[3]) == 0) || (value_table_size <= 0)),
      OP_LOGE(kFusedOpType.c_str(), "The value_table_size have 0 element"), return FAILED);
  vector<int64_t> avgpool1d_assit_dim_info;
  Status ret;
  if (input_type == DT_FLOAT16) {
    unique_ptr<uint16_t[]> input_assit(new (nothrow) uint16_t[value_table_size]());
    FUSION_PASS_CHECK(input_assit.get() == nullptr, OP_LOGE(kFusedOpType.c_str(), "The input_assit is NULL"),
                      return PARAM_INVALID);
    ret = AvgValueTableGen(orig_input_shape_v, k_size, strides, pads, ceil_mode, count_include_pad,
                           avgpool1d_assit_dim_info, input_assit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(kFusedOpType.c_str(), "AssitHelp failed."), return ret);
    GeShape avppool1d_assit_shape(avgpool1d_assit_dim_info);
    // set tensor_desc
    GeTensorDesc tensor_desc(GeShape(), FORMAT_NC1HWC0, input_type);
    tensor_desc.SetFormat(FORMAT_NC1HWC0);
    tensor_desc.SetOriginFormat(input_format);
    tensor_desc.SetShape(avppool1d_assit_shape);
    tensor_desc.SetOriginShape(avppool1d_assit_shape);
    FUSION_PASS_MAKE_SHARED(
        (avg_table_assit_ptr = make_shared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t*>(input_assit.get()),
                                                     value_table_size * sizeof(uint16_t))),
        avg_table_assit_ptr = nullptr;
        return PARAM_INVALID);
  } else if (input_type == DT_FLOAT) {
    unique_ptr<float[]> input_assit(new (nothrow) float[value_table_size]());
    FUSION_PASS_CHECK(input_assit.get() == nullptr, OP_LOGE(kFusedOpType.c_str(), "The input_assit is NULL"),
                      return PARAM_INVALID);
    ret = AvgValueTableGen(orig_input_shape_v, k_size, strides, pads, ceil_mode, count_include_pad,
                           avgpool1d_assit_dim_info, input_assit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(kFusedOpType.c_str(), "AssitHelp failed."), return ret);
    GeShape avppool1d_assit_shape(avgpool1d_assit_dim_info);
    // set tensor_desc
    GeTensorDesc tensor_desc(GeShape(), FORMAT_NC1HWC0, input_type);
    tensor_desc.SetFormat(FORMAT_NC1HWC0);
    tensor_desc.SetOriginFormat(input_format);
    tensor_desc.SetShape(avppool1d_assit_shape);
    tensor_desc.SetOriginShape(avppool1d_assit_shape);
    FUSION_PASS_MAKE_SHARED(
        (avg_table_assit_ptr = make_shared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t*>(input_assit.get()),
                                                     value_table_size * sizeof(float))),
        avg_table_assit_ptr = nullptr;
        return PARAM_INVALID);
  }
  vector<GeTensorPtr> avgpool1d_weights = {avg_table_assit_ptr};
  FUSION_PASS_CHECK(OpDescUtils::SetWeights(avgpool1d_fussed_node, avgpool1d_weights) != GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "SetWeights failed."), return FAILED);

  auto avgpool1d_const_input_nodes = OpDescUtils::GetConstInputs(avgpool1d_fussed_node);
  NodePtr avgpool1d_const_input = avgpool1d_const_input_nodes[0];
  auto avgpool1d_const_input_desc = avgpool1d_const_input->GetOpDesc();
  FUSION_PASS_CHECK(avgpool1d_const_input_desc == nullptr,
                    OP_LOGE(kFusedOpType.c_str(), "The avgpool1d_const_input_desc is null, fusion failed."),
                    return PARAM_INVALID);
  avgpool1d_const_input_desc->SetType(kConstantOp);
  avgpool1d_desc->SetType("AvgPool1DD");
  return SUCCESS;
}
REGISTER_PASS("AvgPool1DFusionPass", BUILT_IN_GRAPH_PASS, AvgPool1DFusionPass);
}  // namespace fe
