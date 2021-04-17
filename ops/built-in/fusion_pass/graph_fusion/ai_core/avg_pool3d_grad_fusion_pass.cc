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
 * \file avg_pool3d_grad_fusion_pass.cc
 * \brief avg_pool3d grad fusion pass(AvgPool3DGrad --> AvgPool3DGradD)
 */
#include "avg_pool3d_grad_fusion_pass.h"

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
#include "error_util.h"

using namespace std;
using namespace ge;

namespace fe {
namespace {
  static const string kPatternAvgPool3DGrad = "AvgPool3DGrad";
  static const string kOpType = "AvgPool3DGrad";
  static const string kConstantOp = "Constant";
  constexpr int64_t kC0{16};

  constexpr int64_t kKsizeDim{3};
  constexpr int64_t kStridesDim{3};
  constexpr int64_t kOriShapeDim{5};
  constexpr int64_t kShapeDim{6};
  constexpr int64_t kPadsDim{6};
}

bool IsVectorImpl(const vector<int64_t> &fmap_shape, 
  const vector<int64_t> &ksize,
  const vector<int64_t> &pads)
{
  int64_t fd = fmap_shape[1];
  int64_t fh = fmap_shape[2];
  int64_t fw = fmap_shape[3];
  int64_t kh = ksize[0];
  int64_t kw = ksize[1];
  int64_t kd = ksize[2];

  if ((kd >= fd + pads[0] + pads[1]) && (kh >= fh + pads[2] + pads[3]) && (kw >= fw + pads[4] + pads[5])) {
    return true;
  }
  return false;
}

bool TransformFormat(const vector<int64_t> &ori_shape, const string &ori_format,
                     vector<int64_t> &tar_shape, const string &tar_format)
{
  if (ori_shape.size() != tar_shape.size()) {
    return false;
  }
  for (uint32_t i = 0; i < tar_shape.size(); ++i) {
    for (uint32_t j = 0; j < ori_shape.size(); ++j) {
      if (tar_format[i] == ori_format[j]) {
        tar_shape[i] = ori_shape[j];
        break;
      }
    }
  }
  return true;
}

vector<FusionPattern*> AvgPool3DGradFusionPass::DefinePatterns() {
  OP_LOGI("get into avgpool3DGrad define patterns...");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (nothrow) FusionPattern("AvgPool3DGradFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, CUBE_CALL_ERR_REPORT(kFusedOpType.c_str(), "New a pattern obj failed."),
    return patterns);
  pattern->AddOpDesc(kPatternAvgPool3DGrad, {"AvgPool3DGrad"}).SetOutput(kPatternAvgPool3DGrad);
  patterns.push_back(pattern);
  return patterns;
}

bool DeleteNodeFromGraph(ComputeGraph& graph, NodePtr& node_ptr)
{
  for (auto in_anchor: node_ptr->GetAllInDataAnchors()) {
    if (in_anchor != nullptr) {
      in_anchor->UnlinkAll();
    }
  }
  FUSION_PASS_CHECK(graph.RemoveNode(node_ptr) != SUCCESS,
                    CUBE_CALL_ERR_REPORT("avgpool3d_grad", "remove node %s failed.", node_ptr->GetName().c_str()),
                    return false);
  return true;
}

void GenFilter(int64_t filter_size, float val, uint16_t *data)
{
  uint64_t block_num = filter_size / (kC0 * kC0);
  fp16_t tmp_val;
  for (uint64_t i = 0, idx = 0; i < block_num; ++i) {
    for (uint64_t j = 0; j < kC0; ++j) {
      for (uint64_t k = 0; k < kC0 && idx < filter_size; ++k) {
        float tmp = j == k ? val: 0;
        tmp_val = tmp;
        data[idx++] = tmp_val.val;
      }
    }
  }
}

void GenMultiplies(const vector<int64_t> &input_shape, const vector<int64_t> &grads_shape,
                   const vector<int64_t> &ksize, const vector<int64_t> &strides,\
                   const vector<int64_t> &pads, bool ceil_mode, bool count_include_pad,
                   int64_t size, uint16_t *data)
{
  int64_t input_len_d = input_shape[1] + pads[0] + pads[1];
  int64_t input_len_h = input_shape[2] + pads[2] + pads[3];
  int64_t input_len_w = input_shape[3] + pads[4] + pads[5];
  fp16_t tmp16;
  for (int64_t nn = 0, cnt = 0; nn < grads_shape[0]; ++nn) {
    for (int64_t dd = 0, d_st = 0; dd < grads_shape[1]; ++dd, d_st += strides[2]) {
      for (int64_t c1 = 0; c1 < grads_shape[2]; ++c1) {
        for (int64_t hh = 0, h_st = 0; hh < grads_shape[3]; ++hh, h_st += strides[0]) {
          for (int64_t ww = 0, w_st = 0; ww < grads_shape[4]; ++ww, w_st += strides[1]) {
            float val = 1.0;
            int64_t valid_d = 0;
            int64_t valid_h = 0;
            int64_t valid_w = 0;
            if (count_include_pad) {
              valid_d = d_st + ksize[2] <= input_len_d ? ksize[2]: input_len_d - d_st;
              valid_h = h_st + ksize[0] <= input_len_h ? ksize[0]: input_len_h - h_st;
              valid_w = w_st + ksize[1] <= input_len_w ? ksize[1]: input_len_w - w_st;
            } else {
              valid_d = min(d_st + ksize[2], pads[0] + input_shape[1]) - max(pads[0], d_st);
              valid_h = min(h_st + ksize[0], pads[2] + input_shape[2]) - max(pads[2], h_st);
              valid_w = min(w_st + ksize[1], pads[4] + input_shape[3]) - max(pads[4], w_st);
            }

            float tmp = 1.0 / (valid_d * valid_h * valid_w);
            tmp16 = tmp;
            for (int c0 = 0; c0 < grads_shape[5]; ++c0) {
              FUSION_PASS_CHECK(cnt >= size,
                                CUBE_CALL_ERR_REPORT("avgpool3d_grad", "Multiplier size exceed"),
                                return);
              data[cnt ++] = tmp16.val;
            }
          }
        }
      }
    }
  }
}

bool IsPadsZero(const vector<int64_t> &pads)
{
  for (auto i: pads) {
    if (i != 0) {
      return false;
    }
  }
  return true;
}

Status AvgPool3DGradFusionPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& fusion_nodes) {
  OP_LOGI("get into avgpool_3d_grad fusion pass..");
  ge::NodePtr node_ptr = GetNodeFromMapping("AvgPool3DGrad", mapping);
  FUSION_PASS_CHECK(node_ptr == nullptr,
                    CUBE_CALL_ERR_REPORT(kOpType.c_str(), "AvgPool3DGrad node ptr is null, fusion failed."),
                    return PARAM_INVALID);

  std::string fusion_op_type = "AvgPool3DGradD";
  std::vector<PassAttrInfo> pass_info;
  ge::NodePtr fusion_node_ptr = nullptr;
  PassAttrInfo orig_input_shape_attr = {0, "orig_input_shape", "SetListInt"};
  pass_info.push_back(orig_input_shape_attr);
  // const org input shape change to attr
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, node_ptr, fusion_op_type, pass_info,
                                                      fusion_node_ptr);
  if (ret != SUCCESS) {
    OP_LOGI(kOpType.c_str(), "AvgPool3DGrad has input which is not a CONST, graph not changed.");
    return NOT_CHANGED;
  }
  // get opdesc pointer
  ge::OpDescPtr op_desc = fusion_node_ptr->GetOpDesc();
  FUSION_PASS_CHECK(op_desc == nullptr,
                    CUBE_CALL_ERR_REPORT(kOpType.c_str(), "AvgPool3DGrad is null, fusion failed"),
                    return PARAM_INVALID);

  vector<int64_t> ksize;
  vector<int64_t> strides;
  vector<int64_t> pads;
  vector<int64_t> orig_input_shape;
  bool ceil_mode = false;
  bool count_include_pad = true;
  int64_t divisor_override = 0;
  string format_str;

  AttrUtils::GetListInt(op_desc, "ksize", ksize); // default HWD
  AttrUtils::GetListInt(op_desc, "strides", strides);
  AttrUtils::GetListInt(op_desc, "pads", pads);
  AttrUtils::GetListInt(op_desc, "orig_input_shape", orig_input_shape);
  AttrUtils::GetBool(op_desc, "ceil_mode", ceil_mode);
  AttrUtils::GetBool(op_desc, "count_include_pad", count_include_pad);
  AttrUtils::GetInt(op_desc, "divisor_override", divisor_override);
  AttrUtils::GetStr(op_desc, "data_format", format_str);

  // check attributes' size
  FUSION_PASS_CHECK(orig_input_shape.size() != kOriShapeDim,
                    CUBE_INNER_ERR_REPORT(kOpType.c_str(), "orig_input_shape dim must be 5."), return PARAM_INVALID);
  FUSION_PASS_CHECK(format_str != "NDHWC" && format_str != "NCDHW",
                    CUBE_INNER_ERR_REPORT(kOpType.c_str(),
                                            "format should be NDHWC or NCDHW."), return PARAM_INVALID);
  FUSION_PASS_CHECK(ksize.size() != kKsizeDim,
                    CUBE_INNER_ERR_REPORT(kOpType.c_str(), "ksize len should be 3."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(strides.size() != kStridesDim,
                    CUBE_INNER_ERR_REPORT(kOpType.c_str(), "strides len should be 5."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(pads.size() != kPadsDim,
                    CUBE_INNER_ERR_REPORT(kOpType.c_str(), "pads len should be 6."),
                    return PARAM_INVALID);

  // format to NDHWC
  vector<int64_t> orig_input_shape_formated(orig_input_shape.size());
  TransformFormat(orig_input_shape, format_str, orig_input_shape_formated, "NDHWC");

  op_desc->SetType("AvgPool3DGradD");
  Format format = format_str == "NDHWC" ? FORMAT_NDHWC: FORMAT_NCDHW;

  GeTensorDesc grads_tensor_desc = fusion_node_ptr->GetOpDesc()->GetInputDesc("grads");
  vector<int64_t> grads_ori_shape_vec = grads_tensor_desc.GetShape().GetDims();
  vector<int64_t> grads_ori_shape_vec_formated(grads_ori_shape_vec.size());
  string grads_ori_format_str = grads_tensor_desc.GetOriginFormat() == FORMAT_NCDHW? "NCDHW": "NDHWC";
  TransformFormat(grads_ori_shape_vec, grads_ori_format_str, grads_ori_shape_vec_formated, "NDHWC");

  // determin whether to enter the vector impl.
  if (IsVectorImpl(orig_input_shape_formated, ksize, pads)) {
    FUSION_PASS_CHECK(grads_ori_shape_vec_formated[1] != 1 ||
                        grads_ori_shape_vec_formated[2] != 1 ||
                        grads_ori_shape_vec_formated[3] != 1,
                      CUBE_INNER_ERR_REPORT(kOpType.c_str(), "global mode, grads shape is incorrected."),
                      return PARAM_INVALID);
    return SUCCESS;
  }

  // create filter tensor
  float val = 1.0f;
  if (divisor_override != 0) {
    val = 1.0 / divisor_override;
  } else if (IsPadsZero(pads) && !ceil_mode) {
    val = 1.0f / (ksize[0] * ksize[1] * ksize[2]);
  }
  vector<int64_t> filter_shape_vec {(orig_input_shape_formated[4] + kC0 - 1) / kC0 * ksize[0] * ksize[1] * ksize[2],
                                    1, kC0, kC0};
  int64_t filter_size = 1;
  for (auto i: filter_shape_vec) {
    FUSION_PASS_CHECK(PatternFusionUtil::IsUnknownShape(i),
                      CUBE_INNER_ERR_REPORT(kOpType.c_str(), "input:ksize has an unknown shape."),
                      return NOT_CHANGED);
    FUSION_PASS_CHECK(filter_size == 0 || filter_size * i / filter_size != i,
                      CUBE_INNER_ERR_REPORT(kOpType.c_str(), "input:filter input size exceed INT64_MAX."),
                      return NOT_CHANGED);
    filter_size *= i;
  }
  GeTensorPtr filter_ptr{nullptr};
  shared_ptr<uint16_t> filter_mem(new (nothrow) uint16_t[filter_size],
                                    default_delete<uint16_t[]>());
  FUSION_PASS_CHECK(filter_mem.get() == nullptr, CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Filter is NULL."),
                    return PARAM_INVALID);

  GenFilter(filter_size, val, filter_mem.get());
  GeShape filter_shape(filter_shape_vec);
  GeTensorDesc filter_tensor_desc(filter_shape, FORMAT_FRACTAL_Z_3D, ge::DT_FLOAT16);
  vector<int64_t> filter_ori_shape_vec = {ksize[2], ksize[0], ksize[1], 1, orig_input_shape_formated[4]};
  GeShape filter_ori_shape(filter_ori_shape_vec);
  filter_tensor_desc.SetOriginShape(filter_ori_shape);
  filter_tensor_desc.SetOriginFormat(FORMAT_DHWCN);
  FUSION_PASS_MAKE_SHARED((filter_ptr = make_shared<GeTensor>(filter_tensor_desc,
                                                              reinterpret_cast<uint8_t *>(filter_mem.get()),
                                                              filter_size * sizeof(uint16_t))),
                          filter_ptr = nullptr;
                          return PARAM_INVALID);
  vector<GeTensorPtr> weights;
  weights.push_back(filter_ptr);
  // create mul tensor
  if (divisor_override == 0 && (!IsPadsZero(pads) || ceil_mode)) {
    OP_LOGI(kOpType.c_str(), "create mul matrix...");
    vector<int64_t> grads_shape_vec = {grads_ori_shape_vec_formated[0],
                                       grads_ori_shape_vec_formated[1],
                                       (grads_ori_shape_vec_formated[4] + kC0 - 1) / kC0,
                                       grads_ori_shape_vec_formated[2],
                                       grads_ori_shape_vec_formated[3],
                                       kC0};
    int64_t grads_size = 1;
    for (auto i: grads_shape_vec) {
      FUSION_PASS_CHECK(PatternFusionUtil::IsUnknownShape(i),
                        CUBE_INNER_ERR_REPORT(kOpType.c_str(), "input:grads has an unknown shape."),
                        return NOT_CHANGED);
      FUSION_PASS_CHECK(grads_size == 0 || grads_size * i / grads_size != i,
                        CUBE_INNER_ERR_REPORT(kOpType.c_str(), "input:grads input size exceed INT64_MAX."),
                        return NOT_CHANGED);
      grads_size *= i;
    }
    GeTensorPtr multiplier_ptr;
    shared_ptr<uint16_t> multiplies_mem(new (nothrow) uint16_t[grads_size],
                                      default_delete<uint16_t[]>());
    FUSION_PASS_CHECK(multiplies_mem.get() == nullptr,
                      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "multiplies is NULL."),
                      return PARAM_INVALID);
    GenMultiplies(orig_input_shape_formated, grads_shape_vec, ksize, strides, pads, ceil_mode, count_include_pad, grads_size, multiplies_mem.get());
    GeShape mul_shape(grads_shape_vec);
    GeTensorDesc mul_tensor_desc(mul_shape, FORMAT_NDC1HWC0, DT_FLOAT16);
    mul_tensor_desc.SetOriginShape(GeShape(grads_ori_shape_vec_formated));
    mul_tensor_desc.SetOriginFormat(FORMAT_NDHWC);
    FUSION_PASS_MAKE_SHARED(
            (multiplier_ptr = std::make_shared<ge::GeTensor>(mul_tensor_desc, reinterpret_cast<uint8_t*>(multiplies_mem.get()),
                                                             grads_size *sizeof(uint16_t))),
            multiplier_ptr = nullptr;
            return PARAM_INVALID);
    weights.push_back(multiplier_ptr);
  }
  ge::OpDescUtils::SetWeights(fusion_node_ptr, weights);

  // set filter and mul inputas constant input
  auto const_input_nodes = OpDescUtils::GetConstInputs(fusion_node_ptr);
  FUSION_PASS_CHECK(const_input_nodes.size() <= 0,
                    CUBE_INNER_ERR_REPORT(kOpType.c_str(), "GetConstInputs Error Size: %zu", const_input_nodes.size()),
                    return PARAM_INVALID);
  for (int i = 0; i < const_input_nodes.size(); i++) {
    NodePtr const_input = const_input_nodes[i];
    const_input->GetOpDesc()->SetType(kConstantOp);
  }

  return SUCCESS;
}

REGISTER_PASS("AvgPool3DGradFusionPass", BUILT_IN_GRAPH_PASS, AvgPool3DGradFusionPass);
} // namespace fe
