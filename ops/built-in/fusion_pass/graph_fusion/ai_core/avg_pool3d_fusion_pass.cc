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
 * \file avg_pool3d_fusion_pass.cpp
 * \brief avg_pool3d fusion pass(AvgPool3D --> AvgPool3DD)
 */
#include "avg_pool3d_fusion_pass.h"

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
static const string kPatternAvgPool3D = "AvgPool3D";
static const string kConstantOp = "Constant";
constexpr int64_t kC0{16};

bool IsVectorImpl(int fmap_h, int fmap_w, int kh, int kw, vector<int> pads) {
  for (int i = 0; i < pads.size(); i++) {
    if (pads[i] != 0) {
      return false;
    }
  }
  if (fmap_h != kh || fmap_w != kw) {
    return false;
  }
  return true;
}

bool IsZeroPads(vector<int> pads) {
  bool all_zero=true;
  for (int i=0; i < pads.size(); i++) {
    if (pads[i] != 0) {
      all_zero = false;
    }
  }
  return all_zero;
}

Status GenFilter(int32_t n, uint16_t* output, float val) {
  int cnt = n / (kC0*kC0);
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
                   int kh, int kw, int stride_d, int stride_h, int stride_w, vector<int> pads, uint16_t *data,
                   int max_size, bool ceil_mode, bool count_include_pad) {
  int pad_d = pads[0] + pads[1];
  int pad_h = pads[2] + pads[3];
  int pad_w = pads[4] + pads[5];
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
            int valid_h = GetIntersection(start_h, start_h + kh, pads[2], pads[2] + fmap_h);
            int valid_w = GetIntersection(start_w, start_w + kw, pads[4], pads[4] + fmap_w);
            int valid_data = valid_d * valid_h * valid_w;
            int valid_kernel = v_kd * v_kh * v_kw;
            fp16_t t;
            t.val = 0;
            float val = count_include_pad ? 1.0 / valid_kernel : 1.0 / valid_data;
            t = val;
            for (int c = 0; c < kC0; c++) {
              if (cnt >= max_size){
                OP_LOGE("Multiplier size error, max size is %d.",max_size);
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


vector<FusionPattern*> AvgPool3DFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (nothrow) FusionPattern("AvgPool3DFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "New a pattern obj failed."), return patterns);
  pattern->AddOpDesc(kPatternAvgPool3D, {"AvgPool3D"}).SetOutput(kPatternAvgPool3D);
  patterns.push_back(pattern);
  return patterns;
}

Status AvgPool3DFusionPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& fusion_nodes) {
  NodePtr op_node = GetNodeFromMapping("AvgPool3D", mapping);
  Operator op = OpDescUtils::CreateOperatorFromNode(op_node);
  OpDescPtr op_node_desc = op_node->GetOpDesc();
  GeTensorDesc inputTensorDesc = op_node_desc->GetInputDesc(0);
  GeTensorDesc outTensorDesc = op_node_desc->GetOutputDesc(0);
  Format data_format = outTensorDesc.GetFormat();
  vector<int64_t> dims_out = outTensorDesc.GetShape().GetDims();
  vector<int64_t> dims_in = inputTensorDesc.GetShape().GetDims();
  int64_t dout;
  int64_t ho;
  int64_t wo;
  int64_t fmap_n;
  int64_t fmap_c;
  int64_t fmap_d;
  int64_t fmap_h;
  int64_t fmap_w;
  if (data_format == FORMAT_NDHWC) {
    dout = dims_out[1];
    ho = dims_out[2];
    wo = dims_out[3];
    fmap_n = dims_in[0];
    fmap_d = dims_in[1];
    fmap_h = dims_in[2];
    fmap_w = dims_in[3];
    fmap_c = dims_in[4];
  } else if (data_format == FORMAT_NCDHW) {
    dout = dims_out[2];
    ho = dims_out[3];
    wo = dims_out[4];
    fmap_n = dims_in[0];
    fmap_c = dims_in[1];
    fmap_d = dims_in[2];
    fmap_h = dims_in[3];
    fmap_w = dims_in[4];
  } else {
    // DHWCN
    dout = dims_out[0];
    ho = dims_out[1];
    wo = dims_out[2];
    fmap_d = dims_in[0];
    fmap_h = dims_in[1];
    fmap_w = dims_in[2];
    fmap_c = dims_in[3];
    fmap_n = dims_in[4];
  }

  if (PatternFusionUtil::IsUnknownShape(dout) ||
      PatternFusionUtil::IsUnknownShape(ho) ||
      PatternFusionUtil::IsUnknownShape(wo) ||
      PatternFusionUtil::IsUnknownShape(fmap_d) ||
      PatternFusionUtil::IsUnknownShape(fmap_h) ||
      PatternFusionUtil::IsUnknownShape(fmap_w) ||
      PatternFusionUtil::IsUnknownShape(fmap_n) ||
      PatternFusionUtil::IsUnknownShape(fmap_c)) {
    OP_LOGE(kFusedOpType.c_str(), "AvgPool3DFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }

  int64_t fmap_c1 = (fmap_c + kC0 - 1) / kC0;

  vector<int32_t> ksize;
  op.GetAttr("ksize",ksize);
  int kd=ksize[1];
  int kh=ksize[2];
  int kw=ksize[3];

  vector<int32_t> strides;
  op.GetAttr("strides",strides);
  int stride_d = strides[1];
  int stride_h = strides[2];
  int stride_w = strides[3];

  vector<int32_t> pads;
  op.GetAttr("pads",pads);

  bool count_include_pad = true;
  op.GetAttr("count_include_pad",count_include_pad);

  bool ceil_mode = false;
  op.GetAttr("ceil_mode", ceil_mode);

  int divisor_override{0};
  op.GetAttr("divisor_override",divisor_override);

  if (IsVectorImpl(fmap_h, fmap_w, kh, kw, pads)) {
    op_node_desc->SetType("AvgPool3DD");
    return SUCCESS;
  }

  int64_t filter_size = fmap_c1 * kd * kh * kw * kC0 * kC0;
  GeTensorPtr filter_ptr{nullptr};
  unique_ptr<uint16_t> filter_mem(new (nothrow) uint16_t[filter_size]());
  FUSION_PASS_CHECK(filter_mem.get() == nullptr, OP_LOGE(kFusedOpType.c_str(), "Filter is NULL"),
                    return PARAM_INVALID);

  float val = 1.0 / (kd * kh * kw);

  if (divisor_override) {
    val = 1.0 / divisor_override;
  } else if (!IsZeroPads(pads)) {
    // need multiplier and filter is all one in diagnoal
    val = 1.0;
  }

  GenFilter(filter_size, filter_mem.get(), val);

  // define shape
  vector<int64_t> assit_dim_info{fmap_c1 * kd * kh * kw, 1, kC0, kC0};
  GeShape assit_shape(assit_dim_info);
  GeTensorDesc filter_tensor_desc(assit_shape, FORMAT_FRACTAL_Z_3D, DT_FLOAT16);

  vector<int64_t> assit_dim_info_ori{kd, kh, kw, fmap_c, 1};
  GeShape assit_shape_ori(assit_dim_info_ori);
  filter_tensor_desc.SetOriginShape(assit_shape_ori);
  filter_tensor_desc.SetOriginFormat(FORMAT_DHWCN);

  FUSION_PASS_MAKE_SHARED((filter_ptr = make_shared<GeTensor>(filter_tensor_desc,
                                                              reinterpret_cast<uint8_t*>(filter_mem.get()),
                                                              filter_size * sizeof(uint16_t))),
                          filter_ptr = nullptr;
                          return PARAM_INVALID);

  vector<GeTensorPtr> weights = {filter_ptr};

  if (!IsZeroPads(pads) && !divisor_override) {
    GeTensorPtr multiplier_ptr{nullptr};
    int64_t multiplier_size = fmap_n * fmap_c1 * kC0 * dout * ho * wo;
    unique_ptr<uint16_t> multiplier_mem(new (nothrow) uint16_t[multiplier_size]());
    FUSION_PASS_CHECK(multiplier_mem.get() == nullptr, OP_LOGE(kFusedOpType.c_str(), "Multiplier is NULL"),
                      return PARAM_INVALID);

    GenMultiplier(fmap_n, fmap_c1, fmap_d, fmap_h, fmap_w, dout, ho, wo, kd, kh, kw, stride_d, stride_h, stride_w, pads,
                  multiplier_mem.get(), multiplier_size, ceil_mode, count_include_pad);

    vector<int64_t> mul_dim_info{fmap_n,  dout, fmap_c1, ho, wo, kC0};
    GeShape mul_shape(mul_dim_info);
    GeTensorDesc mul_tensor_desc(mul_shape, FORMAT_NDC1HWC0, DT_FLOAT16);
    mul_tensor_desc.SetOriginShape(mul_shape);
    mul_tensor_desc.SetOriginFormat(FORMAT_NDC1HWC0);
    FUSION_PASS_MAKE_SHARED((multiplier_ptr = make_shared<GeTensor>(mul_tensor_desc,
                                                                    reinterpret_cast<uint8_t*>(multiplier_mem.get()),
                                                                    multiplier_size * sizeof(uint16_t))),
                            multiplier_ptr = nullptr;
                            return PARAM_INVALID);
    weights.push_back(multiplier_ptr);
  }

  OpDescUtils::SetWeights(op_node, weights);
  auto const_input_nodes = OpDescUtils::GetConstInputs(op_node);
  if (const_input_nodes.size() <= 0) {
    OP_LOGE(kFusedOpType.c_str(), "GetConstInputs Error Size: %u",const_input_nodes.size());
    return PARAM_INVALID;
  }

  for (int i = 0; i < const_input_nodes.size(); i++) {
    NodePtr const_input = const_input_nodes[i];
    const_input->GetOpDesc()->SetType(kConstantOp);
  }

  op_node_desc->SetType("AvgPool3DD");

  return SUCCESS;
}
REGISTER_PASS("AvgPool3DFusionPass", BUILT_IN_GRAPH_PASS, AvgPool3DFusionPass);
}  // namespace fe
