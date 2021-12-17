/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file avg_pool3d_fusion_pass.cpp
 * \brief avg_pool3d fusion pass(AvgPool3D --> AvgPool3DD)
 */
#include "avg_pool3d_fusion_pass.h"

#include <string>
#include <vector>

#include "fp16_t.hpp"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
static const string kPatternAvgPool3D = "AvgPool3D";
static const string kConstantOp = "Constant";
constexpr int64_t kC0{16};
static const int kIndex2 = 2;
static const int kIndex3 = 3;
static const int kIndex4 = 4;
static const int kIndex5 = 5;
static const int kDimSize3 = 3;
static const int kDimSize5 = 5;

bool IsVectorImpl(int fmap_h, int fmap_w, int kh, int kw, const vector<int>& pads) {
  for (size_t i = 0; i < pads.size(); i++) {
    if (pads[i] != 0) {
      return false;
    }
  }
  if (fmap_h != kh || fmap_w != kw) {
    return false;
  }
  return true;
}

bool IsZeroPads(const vector<int>& pads) {
  bool all_zero = true;
  for (size_t i = 0; i < pads.size(); i++) {
    if (pads[i] != 0) {
      all_zero = false;
    }
  }
  return all_zero;
}

Status GenFilter(int32_t n, uint16_t* output, float val) {
  int cnt = n / (kC0 * kC0);
  for (int32_t i = 0; i < cnt; ++i) {
    for (int32_t j = 0; j < kC0; j++) {
      for (int32_t k = 0; k < kC0; k++) {
        fp16_t t;
        t.val = 0;
        t = j == k ? val : 0;
        output[i * kC0 * kC0 + j * kC0 + k] = t.val;
      }
    }
  }
  return SUCCESS;
}

int GetIntersection(int pos1_start, int pos1_end, int pos2_start, int pos2_end) {
  if (pos1_end <= pos2_start) {
    return 0;
  }
  if (pos1_start >= pos2_end) {
    return 0;
  }
  if (pos1_start < pos2_start) {
    pos1_start = pos2_start;
  }
  if (pos1_end > pos2_end) {
    pos1_end = pos2_end;
  }
  return pos1_end - pos1_start;
}

void GenMultiplier(int fmap_n, int fmap_c1, int fmap_d, int fmap_h, int fmap_w, int dout, int ho, int wo, int kd,
                   int kh, int kw, int stride_d, int stride_h, int stride_w, const vector<int>& pads, uint16_t* data,
                   int max_size, bool count_include_pad) {
  int pad_d = pads[0] + pads[1];
  int pad_h = pads[kIndex2] + pads[kIndex3];
  int pad_w = pads[kIndex4] + pads[kIndex5];
  int cnt = 0;
  int len_d = fmap_d + pad_d;
  int len_h = fmap_h + pad_h;
  int len_w = fmap_w + pad_w;
  for (int nn = 0; nn < fmap_n; nn++) {
    int start_d = 0;
    for (int di = 0; di < dout; di++) {
      int v_kd = start_d + kd <= len_d ? kd : len_d - start_d;
      for (int cc = 0; cc < fmap_c1; cc++) {
        int start_h = 0;
        for (int hi = 0; hi < ho; hi++) {
          int v_kh = start_h + kh <= len_h ? kh : len_h - start_h;
          int start_w = 0;
          for (int wi = 0; wi < wo; wi++) {
            int v_kw = start_w + kw <= len_w ? kw : len_w - start_w;
            int valid_d = GetIntersection(start_d, start_d + kd, pads[0], pads[0] + fmap_d);
            int valid_h = GetIntersection(start_h, start_h + kh, pads[kIndex2], pads[kIndex2] + fmap_h);
            int valid_w = GetIntersection(start_w, start_w + kw, pads[kIndex4], pads[kIndex4] + fmap_w);
            int valid_data = valid_d * valid_h * valid_w;
            int valid_kernel = v_kd * v_kh * v_kw;
            fp16_t t;
            t.val = 0;
            FUSION_PASS_CHECK(valid_data == 0,
                              VECTOR_FUSION_INNER_ERR_REPORT(kPatternAvgPool3D.c_str(),
                                                             "valid_data cannot be zero."), return);
            float val = count_include_pad ? 1.0 / valid_kernel : 1.0 / valid_data;
            t = val;
            for (int c = 0; c < kC0; c++) {
              if (cnt >= max_size) {
                VECTOR_FUSION_INNER_ERR_REPORT(kPatternAvgPool3D.c_str(),
                                              "Multiplier size error, max size is %d.", max_size);
                return;
              }
              data[cnt] = t.val;
              cnt++;
            }
            start_w += stride_w;
          }
          start_h += stride_h;
        }
      }
      start_d += stride_d;
    }
  }
}

bool GetStridesAndKSize(Operator& op, Format refer, int32_t& strd, int32_t& strh, int32_t& strw, int32_t& kd,
                        int32_t& kh, int32_t& kw) {
  std::vector<int32_t> stride_list;
  std::vector<int32_t> ksize_list;
  FUSION_PASS_CHECK(op.GetAttr("strides", stride_list) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kPatternAvgPool3D.c_str(), "Get attr strides failed."),
                    return false);
  FUSION_PASS_CHECK(op.GetAttr("ksize", ksize_list) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kPatternAvgPool3D.c_str(), "Get attr ksize failed."),
                    return false);
  if (ksize_list.size() == 1) {
    kd = ksize_list[0];
    kh = ksize_list[0];
    kw = ksize_list[0];
  } else if (ksize_list.size() == kDimSize3) {
    kd = ksize_list[0];
    kh = ksize_list[1];
    kw = ksize_list[kIndex2];
  } else if (ksize_list.size() == kDimSize5) {
    if (refer == FORMAT_NCDHW) {
      kd = ksize_list[kIndex2];
      kh = ksize_list[kIndex3];
      kw = ksize_list[kIndex4];
    } else if (refer == FORMAT_NDHWC) {
      kd = ksize_list[1];
      kh = ksize_list[kIndex2];
      kw = ksize_list[kIndex3];
    } else {
      // DHWCN
      kd = ksize_list[0];
      kh = ksize_list[1];
      kw = ksize_list[kIndex2];
    }
  }
  if (stride_list.size() == 1) {
    strd = stride_list[0];
    strh = stride_list[0];
    strw = stride_list[0];
  } else if (stride_list.size() == kDimSize3) {
    strd = stride_list[0];
    strh = stride_list[1];
    strw = stride_list[kIndex2];
  } else if (stride_list.size() == kDimSize5) {
    if (refer == FORMAT_NCDHW) {
      strd = stride_list[kIndex2];
      strh = stride_list[kIndex3];
      strw = stride_list[kIndex4];
    } else if (refer == FORMAT_NDHWC) {
      strd = stride_list[1];
      strh = stride_list[kIndex2];
      strw = stride_list[kIndex3];
    } else {
      // DHWCN
      strd = stride_list[0];
      strh = stride_list[1];
      strw = stride_list[kIndex2];
    }
  }
  return true;
}

vector<FusionPattern*> AvgPool3DFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (nothrow) FusionPattern("AvgPool3DFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "New a pattern obj failed."),
                    return patterns);
  pattern->AddOpDesc(kPatternAvgPool3D, {"AvgPool3D"}).SetOutput(kPatternAvgPool3D);
  patterns.push_back(pattern);
  return patterns;
}

Status AvgPool3DFusionPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& /* fusion_nodes */) {
  OP_LOGD(kFusedOpType.c_str(), "get into AvgPool3d fusion pass.");
  NodePtr op_node = GetNodeFromMapping("AvgPool3D", mapping);
  Operator op = OpDescUtils::CreateOperatorFromNode(op_node);
  OpDescPtr op_node_desc = op_node->GetOpDesc();
  GeTensorDesc input_tensor_desc = op_node_desc->GetInputDesc(0);
  GeTensorDesc out_tensor_desc = op_node_desc->GetOutputDesc(0);
  Format data_format = out_tensor_desc.GetFormat();
  vector<int64_t> dims_out = out_tensor_desc.GetShape().GetDims();
  vector<int64_t> dims_in = input_tensor_desc.GetShape().GetDims();
  int64_t dout;
  int64_t ho;
  int64_t wo;
  int64_t fmap_n;
  int64_t fmap_c;
  int64_t fmap_d;
  int64_t fmap_h;
  int64_t fmap_w;
  if (data_format == FORMAT_NDHWC) {
    FUSION_PASS_CHECK(dims_out.size() < 4,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Dims of output is not enough."),
                      return FAILED);
    FUSION_PASS_CHECK(dims_in.size() < 5,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Dims of input is not enough."),
                      return FAILED);
    dout = dims_out[1];
    ho = dims_out[kIndex2];
    wo = dims_out[kIndex3];
    fmap_n = dims_in[0];
    fmap_d = dims_in[1];
    fmap_h = dims_in[kIndex2];
    fmap_w = dims_in[kIndex3];
    fmap_c = dims_in[kIndex4];
  } else if (data_format == FORMAT_NCDHW) {
    FUSION_PASS_CHECK(dims_out.size() < 5,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Dims of output is not enough."),
                      return FAILED);
    FUSION_PASS_CHECK(dims_in.size() < 5,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Dims of input is not enough."),
                      return FAILED);
    dout = dims_out[kIndex2];
    ho = dims_out[kIndex3];
    wo = dims_out[kIndex4];
    fmap_n = dims_in[0];
    fmap_c = dims_in[1];
    fmap_d = dims_in[kIndex2];
    fmap_h = dims_in[kIndex3];
    fmap_w = dims_in[kIndex4];
  } else {
    // DHWCN
    FUSION_PASS_CHECK(dims_out.size() < 3,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Dims of output is not enough."),
                      return FAILED);
    FUSION_PASS_CHECK(dims_in.size() < 5,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Dims of input is not enough."),
                      return FAILED);
    dout = dims_out[0];
    ho = dims_out[1];
    wo = dims_out[kIndex2];
    fmap_d = dims_in[0];
    fmap_h = dims_in[1];
    fmap_w = dims_in[kIndex2];
    fmap_c = dims_in[kIndex3];
    fmap_n = dims_in[kIndex4];
  }
  bool is_dynamic = false;
  // when static op or dynamic op phase_running, is_dynamic = false
  if (std::find(dims_in.begin(), dims_in.end(), -1) != dims_in.end()) {
    is_dynamic = true;
    OP_LOGD(kFusedOpType.c_str(), "avg_pool3d fusion pass in dynamic mode.");
  }
  FUSION_PASS_CHECK((PatternFusionUtil::IsUnknownShape(fmap_c)),
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                                   "AvgPool3DFusionPass cannot be applied for unknown shape."),
                    return FAILED);
  int64_t fmap_c1 = (fmap_c + kC0 - 1) / kC0;

  int kd;
  int kh;
  int kw;
  int stride_d;
  int stride_h;
  int stride_w;
  FUSION_PASS_CHECK(!GetStridesAndKSize(op, data_format, stride_d, stride_h, stride_w, kd, kh, kw),
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "GetStridesAndKSize failed"), return FAILED);

  vector<int32_t> pads;
  FUSION_PASS_CHECK(op.GetAttr("pads", pads) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Get attr pads failed"),
                    return FAILED);
  // Attr count_include_pad, ceil_mode and divisor_override are optional
  bool count_include_pad = true;
  op.GetAttr("count_include_pad", count_include_pad);
  bool ceil_mode = false;
  op.GetAttr("ceil_mode", ceil_mode);
  int divisor_override{0};
  op.GetAttr("divisor_override", divisor_override);

  if (!is_dynamic && IsVectorImpl(fmap_h, fmap_w, kh, kw, pads)) {
    OP_LOGD("get into vector impl.");
    op_node_desc->SetType("AvgPool3DD");
    return SUCCESS;
  }

  int64_t filter_size = fmap_c1 * kd * kh * kw * kC0 * kC0;
  GeTensorPtr filter_ptr{nullptr};
  unique_ptr<uint16_t[]> filter_mem(new (nothrow) uint16_t[filter_size]());
  FUSION_PASS_CHECK(filter_mem.get() == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Filter is NULL"),
                    return PARAM_INVALID);

  float val = 1.0;
  if (divisor_override) {
    val = 1.0 / divisor_override;
  } else if (!is_dynamic) {
    val = (!IsZeroPads(pads)) ? 1.0 : (1.0 / (kd * kh * kw));
  } else {
    val = count_include_pad ? 1.0 / (kd * kh * kw) : 1.0;
  }

  FUSION_PASS_CHECK(GenFilter(filter_size, filter_mem.get(), val) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "GenFilter failed"), return FAILED);

  // define shape
  vector<int64_t> assit_dim_info{fmap_c1 * kd * kh * kw, 1, kC0, kC0};
  GeShape assit_shape(assit_dim_info);
  GeTensorDesc filter_tensor_desc(assit_shape, FORMAT_FRACTAL_Z_3D, DT_FLOAT16);

  vector<int64_t> assit_dim_info_ori{kd, kh, kw, fmap_c, 1};
  GeShape assit_shape_ori(assit_dim_info_ori);
  filter_tensor_desc.SetOriginShape(assit_shape_ori);
  filter_tensor_desc.SetOriginFormat(FORMAT_DHWCN);

  FUSION_PASS_MAKE_SHARED(
      (filter_ptr = make_shared<GeTensor>(filter_tensor_desc, reinterpret_cast<uint8_t*>(filter_mem.get()),
                                          filter_size * sizeof(uint16_t))),
      filter_ptr = nullptr;
      return PARAM_INVALID);

  vector<GeTensorPtr> weights = {filter_ptr};

  if (!is_dynamic && !IsZeroPads(pads) && !divisor_override) {
    OP_LOGD(kFusedOpType.c_str(), "create multiplier data.");
    GeTensorPtr multiplier_ptr{nullptr};
    int64_t multiplier_size = fmap_n * fmap_c1 * kC0 * dout * ho * wo;
    unique_ptr<uint16_t[]> multiplier_mem(new (nothrow) uint16_t[multiplier_size]());
    FUSION_PASS_CHECK(multiplier_mem.get() == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Multiplier is NULL"),
                      return PARAM_INVALID);

    GenMultiplier(fmap_n, fmap_c1, fmap_d, fmap_h, fmap_w, dout, ho, wo, kd, kh, kw, stride_d, stride_h, stride_w, pads,
                  multiplier_mem.get(), multiplier_size, count_include_pad);

    vector<int64_t> mul_dim_info{fmap_n, dout, fmap_c1, ho, wo, kC0};
    GeShape mul_shape(mul_dim_info);
    GeTensorDesc mul_tensor_desc(mul_shape, FORMAT_NDC1HWC0, DT_FLOAT16);
    mul_tensor_desc.SetOriginShape(mul_shape);
    mul_tensor_desc.SetOriginFormat(FORMAT_NDC1HWC0);
    FUSION_PASS_MAKE_SHARED(
        (multiplier_ptr = make_shared<GeTensor>(mul_tensor_desc, reinterpret_cast<uint8_t*>(multiplier_mem.get()),
                                                multiplier_size * sizeof(uint16_t))),
        multiplier_ptr = nullptr;
        return PARAM_INVALID);
    weights.push_back(multiplier_ptr);
  }

  FUSION_PASS_CHECK(OpDescUtils::SetWeights(op_node, weights) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "SetWeights failed"), return FAILED);
  auto const_input_nodes = OpDescUtils::GetConstInputs(op_node);
  FUSION_PASS_CHECK(const_input_nodes.empty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                                                  "GetConstInputs Error Size: %lu", const_input_nodes.size()),
                    return PARAM_INVALID);

  for (size_t i = 0; i < const_input_nodes.size(); i++) {
    NodePtr const_input = const_input_nodes[i];
    const_input->GetOpDesc()->SetType(kConstantOp);
  }
  if (!is_dynamic) {
    op_node_desc->SetType("AvgPool3DD");
  }

  GE_DUMP(make_shared<ComputeGraph>(graph), "avg_pool3d_fusion_pass_finish");
  return SUCCESS;
}
REGISTER_PASS("AvgPool3DFusionPass", BUILT_IN_GRAPH_PASS, AvgPool3DFusionPass);
}  // namespace fe
