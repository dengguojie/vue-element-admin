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
 * \file a_a_matmul_nd_to_nd_fusion_pass.cc
 * \brief
 * support dtype: float16
 * unsupported scene: cube_vector_split; one of m,k,n is not divisible by 16(float16)
 * Data(ND)         Data(ND)   Data(ND,optional)                 Data(ND)         Data(ND)      Data(ND,optional)
 *   \                  /          /                               \                    /             /
 *    \                /          /                                 \                  /             /
 *     \              /          /                                   \                /             /
 *   TransData   TransData      /                                     \              /             /
 *       \          /          /                                       \            /             /
 * MatMul/MatMulV2/BatchMatMul/BatchMatMulV2(FRACTAL_NZ)  ->   MatMul/MatMulV2/BatchMatMul/BatchMatMulV2(ND)
 *             |                                                             |
 *         TransData                                                         |
 *             |                                                             |
 *         NetOutput                                                     NetOutput

 * The following scene(Split K) can also apply this graph fusion to enable ND mode
 * Data(ND)         Data(ND)   Data(ND,optional)                 Data(ND)         Data(ND)      Data(ND,optional)
 *   \                  /          /                               \                    /             /
 *    \                /          /                                 \                  /             /
 *     \              /          /                                   \                /             /
 *   TransData   TransData      /                                     \              /             /
 *       \          /          /                                       \            /             /
 *  MatMul/MatMulV2/BatchMatMul/BatchMatMulV2(FRACTAL_NZ)  ->  BatchMatMul/BatchMatMulV2/MatMul/MatMulV2/(FRACTAL_NZ)
 *             |                                                             |
 *          cast[Optional + FP16 --> FP32]                                   |
 *             |                                                             |
 *         NetOutput(FRACTAL_NZ + FP32)                                  NetOutput(FRACTAL_NZ + FP32)
 */
#include "a_a_matmul_nz_to_nd_fusion_pass.h"

#include <string>
#include <vector>

#include "common/util/platform_info.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

namespace fe {
namespace {
static const char kDescMatMul[] = "MatMul";
static const char kDescTransdata0[] = "Transdata0";
static const char kDescTransdata1[] = "Transdata1";
static const char kDescTransdataOut[] = "TransdataOut";
static const char kDescCast[] = "Cast";

static const char kOpTypeMatMul[] = "MatMul";
static const char kOpTypeMatMulV2[] = "MatMulV2";
static const char kOpTypeBatchMatMul[] = "BatchMatMul";
static const char kOpTypeBatchMatMulV2[] = "BatchMatMulV2";
static const char kOpTypeTransData[] = "TransData";
static const char kOpTypeCast[] = "Cast";
static const char kOpTypeNetOutput[] = "NetOutput";
static const char kOpTypeData[] = "Data";

static const char kFormatNd[] = "ND";
static const char kFormatFractalNz[] = "FRACTAL_NZ";

static const char kNameFusionPass[] = "AAMatMulNzToNdFusionPass";
static const char kSplitKdimNameFusionPass[] = "AAMatMulNzToNdSplitKFusionPass";
static const char kSplitKdimNameFP16FusionPass[] = "AAMatMulNzToNdSplitKFP16FusionPass";

static const int kNumDataNodes = 2;
static const int kNumAlignHalf = 16;
static const int kRIdxLast = -1;
static const int kRIdxLastSecond = -2;
}  // namespace

static const vector<string> whitelist_op_type = {kOpTypeMatMul, kOpTypeMatMulV2, kOpTypeBatchMatMul,
                                                 kOpTypeBatchMatMulV2};

vector<FusionPattern*> AAMatMulNzToNdFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern(kNameFusionPass);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGW(kNameFusionPass, "pattern is nullptr, Create pattern not success!"),
                    return patterns);

  OP_LOGD(kNameFusionPass, "Start to define pattern.");
  pattern->AddOpDesc(kDescMatMul, {kOpTypeMatMul, kOpTypeMatMulV2, kOpTypeBatchMatMul, kOpTypeBatchMatMulV2})
      .AddOpDesc(kDescTransdata0, {kOpTypeTransData})
      .AddOpDesc(kDescTransdata1, {kOpTypeTransData})
      .AddOpDesc(kDescTransdataOut, {kOpTypeTransData})
      .SetInputs(kDescMatMul, {kDescTransdata0, kDescTransdata1})
      .SetInputs(kDescTransdataOut, {kDescMatMul})
      .SetOutput(kDescTransdataOut);
  OP_LOGD(kNameFusionPass, "End to define pattern.");

  FusionPattern* split_k_pattern = new (std::nothrow) FusionPattern(kSplitKdimNameFusionPass);
  FUSION_PASS_CHECK(split_k_pattern == nullptr,
                    OP_LOGW(kSplitKdimNameFusionPass, "pattern is nullptr, Create pattern not success!"),
                    return patterns);

  OP_LOGD(kSplitKdimNameFusionPass, "Start to define pattern.");

  split_k_pattern->AddOpDesc(kDescMatMul, {kOpTypeMatMul, kOpTypeMatMulV2, kOpTypeBatchMatMul, kOpTypeBatchMatMulV2})
      .AddOpDesc(kDescTransdata0, {kOpTypeTransData})
      .AddOpDesc(kDescTransdata1, {kOpTypeTransData})
      .SetInputs(kDescMatMul, {kDescTransdata0, kDescTransdata1})
      .SetOutput(kDescMatMul);
  OP_LOGD(kSplitKdimNameFusionPass, "End to define pattern.");

  FusionPattern* force_fp16_split_k_pattern = new (std::nothrow) FusionPattern(kSplitKdimNameFP16FusionPass);
  FUSION_PASS_CHECK(force_fp16_split_k_pattern == nullptr,
                    OP_LOGW(kSplitKdimNameFP16FusionPass, "pattern is nullptr, Create pattern not success!"),
                    return patterns);
  // The following Case is another split kdim pattern.
  OP_LOGD(kSplitKdimNameFP16FusionPass, "Start to define pattern.");
  force_fp16_split_k_pattern
      ->AddOpDesc(kDescMatMul, {kOpTypeMatMul, kOpTypeMatMulV2, kOpTypeBatchMatMul, kOpTypeBatchMatMulV2})
      .AddOpDesc(kDescTransdata0, {kOpTypeTransData})
      .AddOpDesc(kDescTransdata1, {kOpTypeTransData})
      .AddOpDesc(kOpTypeCast, {kOpTypeCast})
      .SetInputs(kDescMatMul, {kDescTransdata0, kDescTransdata1})
      .SetInputs(kDescCast, {kDescMatMul})
      .SetOutput(kDescCast);
  // The mapping order strictly follow the belowing:
  patterns.push_back(pattern);
  patterns.push_back(force_fp16_split_k_pattern);
  patterns.push_back(split_k_pattern);
  OP_LOGD(kSplitKdimNameFP16FusionPass, "End to define pattern.");
  return patterns;
}

bool AAMatMulNzToNdFusionPass::CheckFormatOfTransData(const NodePtr node_ptr_transdata, const string& expect_src_format,
                                                      const string& expect_dst_format) {
  string src_format;
  string dst_format;
  FUSION_PASS_CHECK(!AttrUtils::GetStr(node_ptr_transdata->GetOpDesc(), "src_format", src_format),
                    OP_LOGW(kNameFusionPass, "Unable to get the attribute src_format from TransData."), return false);
  FUSION_PASS_CHECK(!AttrUtils::GetStr(node_ptr_transdata->GetOpDesc(), "dst_format", dst_format),
                    OP_LOGW(kNameFusionPass, "Unable to get the attribute dst_format from TransData."), return false);
  if (src_format != expect_src_format || dst_format != expect_dst_format) {
    return false;
  }
  return true;
}

bool AAMatMulNzToNdFusionPass::CheckOutputFormatOfMatMul(const NodePtr cur_node_ptr,
                                                         const ge::Format& expect_dst_format) const {
  auto node_output_ptr = cur_node_ptr->GetOpDesc()->GetOutputDescPtr(0);
  FUSION_PASS_CHECK(node_output_ptr == nullptr,
                    OP_LOGW(kNameFusionPass, "Unable to get the output pointer for MatMul"), return false);
  ge::Format node_dst_format = node_output_ptr->GetFormat();
  if (node_dst_format != expect_dst_format) {
    return false;
  }
  return true;
}

bool AAMatMulNzToNdFusionPass::IsNumOfNodesOutCorrect() {
  auto out_nodes_data_0 = node_ptr_data_0->GetOutDataNodes().size();
  FUSION_PASS_CHECK(out_nodes_data_0 != 1, OP_LOGI(node_ptr_matmul,
                    "Data (index: 0) before matmul node should only have 1 output, actual %zu.",
                    out_nodes_data_0), return false);

  auto out_nodes_data_1 = node_ptr_data_1->GetOutDataNodes().size();
  FUSION_PASS_CHECK(out_nodes_data_1 != 1, OP_LOGI(node_ptr_matmul,
                    "Data(index: 1) before matmul node should only have 1 output, actual %zu.",
                    out_nodes_data_1), return false);

  auto out_nodes_transdata_0 = node_ptr_transdata_0->GetOutDataNodes().size();
  FUSION_PASS_CHECK(out_nodes_transdata_0 != 1, OP_LOGI(node_ptr_matmul,
                    "Transdata(index: 0) before matmul node should only have 1 output, actual %zu.",
                    out_nodes_transdata_0), return false);

  auto out_nodes_transdata_1 = node_ptr_transdata_1->GetOutDataNodes().size();
  FUSION_PASS_CHECK(out_nodes_transdata_1 != 1, OP_LOGI(node_ptr_matmul,
                    "Transdata(index: 1) before matmul node should only have 1 output, actual %zu.",
                    out_nodes_transdata_1), return false);

  auto out_nodes_matmul = node_ptr_matmul->GetOutDataNodes().size();
  FUSION_PASS_CHECK(out_nodes_matmul != 1,
                    OP_LOGI(node_ptr_matmul, "Matmul node should only have 1 output, actual %zu.", out_nodes_matmul),
                    return false);

  if (!split_k_mode) {
    auto out_nodes_transdata = node_ptr_transdata_out->GetOutDataNodes().size();
    FUSION_PASS_CHECK(out_nodes_transdata != 1,
                      OP_LOGI(node_ptr_matmul,
                              "Transdata after MatMul should only have 1 output, actual %zu.", out_nodes_transdata),
                      return false);
  } else if (split_k_force_fp16_mode) {
    auto out_nodes_cast = node_ptr_cast->GetOutDataNodes().size();
    FUSION_PASS_CHECK(out_nodes_cast != 1,
                      OP_LOGI(node_ptr_matmul,
                              "Cast after MatMul should only have 1 output, actual %zu.", out_nodes_cast),
                      return false);
  }
  return true;
}

bool AAMatMulNzToNdFusionPass::IsStaticShape() {
  return !HasUnKnowShape(node_ptr_data_0) && !HasUnKnowShape(node_ptr_data_1);
}

bool AAMatMulNzToNdFusionPass::IsAligned() {
  auto shape_data_0 = node_ptr_data_0->GetOpDesc()->MutableOutputDesc(0)->GetShape();
  auto shape_data_1 = node_ptr_data_1->GetOpDesc()->MutableOutputDesc(0)->GetShape();

  auto len_shape_data_0 = shape_data_0.GetDimNum();
  auto len_shape_data_1 = shape_data_1.GetDimNum();
  if (shape_data_0.GetDim(len_shape_data_0 + kRIdxLast) % kNumAlignHalf == 0 &&
      shape_data_0.GetDim(len_shape_data_0 + kRIdxLastSecond) % kNumAlignHalf == 0 &&
      shape_data_1.GetDim(len_shape_data_1 + kRIdxLast) % kNumAlignHalf == 0 &&
      shape_data_1.GetDim(len_shape_data_1 + kRIdxLastSecond) % kNumAlignHalf == 0) {
    return true;
  }
  return false;
}

bool AAMatMulNzToNdFusionPass::GetNodePtr(const Mapping& mapping) {
  node_ptr_matmul = GetNodeFromMapping(kDescMatMul, mapping);
  FUSION_PASS_CHECK(node_ptr_matmul == nullptr, OP_LOGW(kNameFusionPass, "Get node matmul failed"), return false);
  node_ptr_transdata_0 = GetNodeFromMapping(kDescTransdata0, mapping);
  FUSION_PASS_CHECK(node_ptr_transdata_0 == nullptr,
                    OP_LOGW(kNameFusionPass, "Get node transdata(idx: 0) before matmul failed"), return false);
  node_ptr_transdata_1 = GetNodeFromMapping(kDescTransdata1, mapping);
  FUSION_PASS_CHECK(node_ptr_transdata_1 == nullptr,
                    OP_LOGW(kNameFusionPass, "Get node transdata(idx: 1) before matmul failed"), return false);
  node_ptr_transdata_out = GetNodeFromMapping(kDescTransdataOut, mapping);
  node_ptr_cast = GetNodeFromMapping(kDescCast, mapping);
  return true;
}

bool AAMatMulNzToNdFusionPass::IsLinkRelationshipCorrect(const Mapping& mapping) {
  if (!GetNodePtr(mapping)) {
    return false;
  }
  // Get the final output node
  auto out_nodes_pattern_out = node_ptr_matmul->GetOutDataNodes();
  if (node_ptr_transdata_out != nullptr) {
    // if transdata_out is detected, this pattern is not split k pattern
    out_nodes_pattern_out = node_ptr_transdata_out->GetOutDataNodes();
    FUSION_PASS_CHECK(!CheckFormatOfTransData(node_ptr_transdata_out, kFormatFractalNz, kFormatNd),
                      OP_LOGI(node_ptr_transdata_out, "TransData after Matmul node does not match."), return false);
  } else {
    FUSION_PASS_CHECK(!CheckOutputFormatOfMatMul(node_ptr_matmul, FORMAT_FRACTAL_NZ),
                      OP_LOGI(kSplitKdimNameFusionPass,
                              "Matmul/MatMulV2/BatchMatMul node[%s]'s format does not match.",
                              node_ptr_matmul->GetName().c_str()),
                      return false);

    split_k_mode = true;
    split_k_force_fp16_mode = node_ptr_cast != nullptr;
    if (split_k_force_fp16_mode) {
      out_nodes_pattern_out = node_ptr_cast->GetOutDataNodes();
      OP_LOGD(kSplitKdimNameFusionPass, "The current scenario is K_dim split and force fp16");
    } else {
      OP_LOGD(kSplitKdimNameFusionPass, "The current scenario is K_dim split and not force fp16");
    }
  }

  FUSION_PASS_CHECK(out_nodes_pattern_out.size() != 1, OP_LOGI(node_ptr_transdata_out,
                    "[Transdata after matmul]/[MatMul] node should only have 1 output, actual %zu.",
                    out_nodes_pattern_out.size()), return false);
  node_ptr_netoutput = out_nodes_pattern_out.at(0);
  const auto in_nodes_transdata0_in = node_ptr_transdata_0->GetInDataNodes();
  FUSION_PASS_CHECK(in_nodes_transdata0_in.size() != 1, OP_LOGI(node_ptr_transdata_0,
                    "Transdata(idx: 0) before matmul node should only have 1 input, actual %zu.",
                    in_nodes_transdata0_in.size()), return false);
  node_ptr_data_0 = in_nodes_transdata0_in.at(0);

  auto in_nodes_transdata1_in = node_ptr_transdata_1->GetInDataNodes();
  FUSION_PASS_CHECK(in_nodes_transdata1_in.size() != 1, OP_LOGI(node_ptr_transdata_1,
                    "Transdata(idx: 1) before matmul node should only have 1 input, actual %zu.",
                    in_nodes_transdata1_in.size()), return false);
  node_ptr_data_1 = in_nodes_transdata1_in.at(0);

  FUSION_PASS_CHECK(node_ptr_netoutput->GetType() != kOpTypeNetOutput,
                    OP_LOGI(node_ptr_netoutput, "The NetOutput operator does not match."), return false);
  FUSION_PASS_CHECK(node_ptr_data_0->GetType() != kOpTypeData,
                    OP_LOGI(node_ptr_data_0, "Data(idx:0) before Matmul node does not match."), return false);
  FUSION_PASS_CHECK(node_ptr_data_1->GetType() != kOpTypeData,
                    OP_LOGI(node_ptr_data_1, "Data(idx:1) before Matmul node does not match."), return false);

  FUSION_PASS_CHECK(!CheckFormatOfTransData(node_ptr_transdata_0, kFormatNd, kFormatFractalNz),
                    OP_LOGI(node_ptr_transdata_0, "TransData(idx:0) before MatMul node does not match."), return false);

  FUSION_PASS_CHECK(!CheckFormatOfTransData(node_ptr_transdata_1, kFormatNd, kFormatFractalNz),
                    OP_LOGI(kNameFusionPass, "TransData(idx:1) before Matmul node does not match."),
                    return false);

  return true;
}

bool AAMatMulNzToNdFusionPass::IsDtypeCorrect() const {
  // Check if the matmul's data type is correct
  FUSION_PASS_CHECK(node_ptr_matmul == nullptr, OP_LOGW(kNameFusionPass, "The node pointer of matmul is Null."),
                    return false);
  bool input_dtype_correct = (node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->GetDataType() == DT_FLOAT16 ||
                              node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->GetDataType() == DT_FLOAT16);
  ge::DataType matmul_output_dtype = (split_k_mode && !split_k_force_fp16_mode) ? DT_FLOAT : DT_FLOAT16;
  bool output_dtype_correct = node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->GetDataType() == matmul_output_dtype;
  if (split_k_force_fp16_mode) {
    FUSION_PASS_CHECK(node_ptr_cast == nullptr, OP_LOGW(kNameFusionPass, "The node pointer of cast is Null."),
                      return false);
    bool cast_dtype_correct = node_ptr_cast->GetOpDesc()->MutableOutputDesc(0)->GetDataType() == DT_FLOAT;
    FUSION_PASS_CHECK(!cast_dtype_correct,
                      OP_LOGI(node_ptr_cast, "The output dytpe of Cast is incorrect. DT_FLOAT is expected."),
                      return false);
  }
  return input_dtype_correct && output_dtype_correct;
}

bool AAMatMulNzToNdFusionPass::NeedFusion(const Mapping& mapping) {
  // Not support: cube_vector_split
  PlatformInfo platform_info;
  OptionalInfo opti_compilation_info;
  FUSION_PASS_CHECK(
      PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, opti_compilation_info) != SUCCESS,
      OP_LOGW(kNameFusionPass, "Get platform_info failed."), return false);
  bool cube_vector_split_flag = platform_info.ai_core_spec.cube_vector_split;
  FUSION_PASS_CHECK(cube_vector_split_flag,
                    OP_LOGI(kNameFusionPass, "Scenario where cube and vector are separated is not supported."),
                    return false);

  // maybe get null pointer, not print node name
  FUSION_PASS_CHECK(!IsLinkRelationshipCorrect(mapping),
                    OP_LOGI(kNameFusionPass, "The connection relationship does not meet expectations."),
                    return false);

  FUSION_PASS_CHECK_NOLOG(!IsNumOfNodesOutCorrect(), return false);

  FUSION_PASS_CHECK(
      !IsDtypeCorrect(),
      OP_LOGI(node_ptr_matmul, "only support input and output data types as [float16 in; float/float16 out]."),
      return false);

  FUSION_PASS_CHECK(IsStaticShape() && !IsAligned(),
                    OP_LOGI(node_ptr_matmul, "static shape and unaligned scenes are not supported."),
                    return false);

  return true;
}

Status AAMatMulNzToNdFusionPass::RestoreOriginalValues() {
  node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->SetShape(in_shape_matmul_0);
  node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->SetShape(in_shape_matmul_1);
  node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->SetShape(out_shape_matmul_0);

  FUSION_PASS_CHECK(
      node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->SetShapeRange(in_range_matmul_0) == ge::GRAPH_FAILED,
      OP_LOGE(node_ptr_matmul, "Failed to restore first input shape range of matmul."), return fe::FAILED);
  FUSION_PASS_CHECK(
      node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->SetShapeRange(in_range_matmul_1) == ge::GRAPH_FAILED,
      OP_LOGE(node_ptr_matmul, "Failed to restore second input shape range of matmul."), return fe::FAILED);
  FUSION_PASS_CHECK(
      node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->SetShapeRange(out_range_matmul_0) == ge::GRAPH_FAILED,
      OP_LOGE(node_ptr_matmul, "Failed to restore output shape range of matmul."), return fe::FAILED);

  node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_FRACTAL_NZ);
  node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->SetFormat(FORMAT_FRACTAL_NZ);
  node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_FRACTAL_NZ);
  if (split_k_force_fp16_mode) {
    node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  }

  return fe::NOT_CHANGED;
}

bool AAMatMulNzToNdFusionPass::GetOriginalValue() {
  // save original value
  in_shape_matmul_0 = node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->GetShape();
  in_shape_matmul_1 = node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->GetShape();
  out_shape_matmul_0 = node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->GetShape();
  FUSION_PASS_CHECK(
      node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->GetShapeRange(in_range_matmul_0) == ge::GRAPH_FAILED,
      OP_LOGW(node_ptr_matmul, "Failed to get first input shape range of matmul."), return false);
  FUSION_PASS_CHECK(
      node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->GetShapeRange(in_range_matmul_1) == ge::GRAPH_FAILED,
      OP_LOGW(node_ptr_matmul, "Failed to get second input shape range of matmul."), return false);
  FUSION_PASS_CHECK(
      node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->GetShapeRange(out_range_matmul_0) == ge::GRAPH_FAILED,
      OP_LOGW(node_ptr_matmul, "Failed to get output shape range of matmul."), return false);
  return true;
}

Status AAMatMulNzToNdFusionPass::UpdateNodesInfo() {
  // step1: update shape, range, format
  vector<pair<int64_t, int64_t>> range_data_0;
  auto out_desc_data_0 = node_ptr_data_0->GetOpDesc()->MutableOutputDesc(0);
  node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->SetShape(out_desc_data_0->GetShape());
  FUSION_PASS_CHECK(out_desc_data_0->GetShapeRange(range_data_0) == ge::GRAPH_FAILED,
                    OP_LOGW(node_ptr_data_0, "Failed to get shape range of data0."), return RestoreOriginalValues());
  FUSION_PASS_CHECK(node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->SetShapeRange(range_data_0) == ge::GRAPH_FAILED,
                    OP_LOGW(node_ptr_matmul, "Failed to set first shape range of MatMul."),
                    return RestoreOriginalValues());

  vector<pair<int64_t, int64_t>> range_data_1;
  auto out_desc_data_1 = node_ptr_data_1->GetOpDesc()->MutableOutputDesc(0);
  node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->SetShape(out_desc_data_1->GetShape());
  FUSION_PASS_CHECK(out_desc_data_1->GetShapeRange(range_data_1) == ge::GRAPH_FAILED,
                    OP_LOGW(node_ptr_data_1, "Failed to get shape range of data1."), return RestoreOriginalValues());
  FUSION_PASS_CHECK(node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->SetShapeRange(range_data_1) == ge::GRAPH_FAILED,
                    OP_LOGW(node_ptr_matmul, "Failed to set second shape range of MatMul."),
                    return RestoreOriginalValues());

  node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->SetFormat(FORMAT_ND);

  // step2: update output shape, range, format, dtype
  if (!split_k_mode) {
    vector<pair<int64_t, int64_t>> range_netoutput;
    auto out_desc_netoutput = node_ptr_netoutput->GetOpDesc()->MutableInputDesc(0);
    node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->SetShape(out_desc_netoutput->GetShape());
    FUSION_PASS_CHECK(out_desc_netoutput->GetShapeRange(range_netoutput) == ge::GRAPH_FAILED,
                      OP_LOGW(node_ptr_netoutput, "Failed to get shape range of NetOutput."),
                      return RestoreOriginalValues());
    FUSION_PASS_CHECK(
        node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->SetShapeRange(range_netoutput) == ge::GRAPH_FAILED,
        OP_LOGW(node_ptr_matmul, "Failed to set output shape range of MatMul."), return RestoreOriginalValues());
    node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  }
  if (split_k_force_fp16_mode) {
    node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  }
  return fe::SUCCESS;
}

bool AAMatMulNzToNdFusionPass::RelinkNodesBeforeMatMul() {
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_data_0->GetOutDataAnchor(0),
                                               node_ptr_transdata_0->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                    OP_LOGE(node_ptr_data_0, "Failed to remove edge between data0 and transdata0."),
                    (void)RestoreOriginalValues();
                    return false);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_transdata_0->GetOutDataAnchor(0),
                                               node_ptr_matmul->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                    OP_LOGE(node_ptr_transdata_0, "Failed to remove edge between transdata0 and matmul."),
                    (void)RestoreOriginalValues();
                    return false);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(node_ptr_data_0->GetOutDataAnchor(0),
                                            node_ptr_matmul->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                    OP_LOGE(node_ptr_data_0, "Failed to add edge between data0 and matmul."),
                    (void)RestoreOriginalValues();
                    return false);

  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_data_1->GetOutDataAnchor(0),
                                               node_ptr_transdata_1->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                    OP_LOGE(node_ptr_data_1, "Failed to remove edge between data1 and transdata1."),
                    (void)RestoreOriginalValues();
                    return false);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_transdata_1->GetOutDataAnchor(0),
                                               node_ptr_matmul->GetInDataAnchor(1)) == ge::GRAPH_FAILED,
                    OP_LOGE(node_ptr_transdata_1, "Failed to remove edge between transdata1 and matmul."),
                    (void)RestoreOriginalValues();
                    return false);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(node_ptr_data_1->GetOutDataAnchor(0),
                                            node_ptr_matmul->GetInDataAnchor(1)) == ge::GRAPH_FAILED,
                    OP_LOGE(node_ptr_data_1, "Failed to add edge between data1 and matmul."),
                    (void)RestoreOriginalValues();
                    return false);
  return true;
}

bool AAMatMulNzToNdFusionPass::RelinkNodesAfterMatMul() {
  // only need to relink output transdata if it is not in split_k pattern
  if (!split_k_mode) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_matmul->GetOutDataAnchor(0),
                                                 node_ptr_transdata_out->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                      OP_LOGE(node_ptr_matmul, "Failed to remove edge between matmul and transdata_out."),
                      (void)RestoreOriginalValues();
                      return false);
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_transdata_out->GetOutDataAnchor(0),
                                                 node_ptr_netoutput->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                      OP_LOGE(node_ptr_transdata_out, "Failed to remove edge between transdata_out and netoutput."),
                      (void)RestoreOriginalValues();
                      return false);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(node_ptr_matmul->GetOutDataAnchor(0),
                                              node_ptr_netoutput->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                      OP_LOGE(node_ptr_matmul, "Failed to add edge between matmul and netoutput."),
                      (void)RestoreOriginalValues();
                      return false);
  }
  if (split_k_force_fp16_mode) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_matmul->GetOutDataAnchor(0),
                                                 node_ptr_cast->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                      OP_LOGE(node_ptr_matmul, "Failed to remove edge between matmul and cast."),
                      (void)RestoreOriginalValues();
                      return false);
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_cast->GetOutDataAnchor(0),
                                                 node_ptr_netoutput->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                      OP_LOGE(node_ptr_cast, "Failed to remove edge between cast and netoutput."),
                      (void)RestoreOriginalValues();
                      return false);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(node_ptr_matmul->GetOutDataAnchor(0),
                                              node_ptr_netoutput->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                      OP_LOGE(node_ptr_matmul, "Failed to add edge between matmul and netoutput."),
                      (void)RestoreOriginalValues();
                      return false);
  }
  return true;
}

bool AAMatMulNzToNdFusionPass::RemoveRedanduntNodes(ge::ComputeGraph* graph) {
  FUSION_PASS_CHECK(graph->RemoveNode(node_ptr_transdata_0) == ge::GRAPH_FAILED,
                    OP_LOGE(node_ptr_transdata_0, "Failed to remove transdata0."), (void)RestoreOriginalValues();
                    return false);
  FUSION_PASS_CHECK(graph->RemoveNode(node_ptr_transdata_1) == ge::GRAPH_FAILED,
                    OP_LOGE(node_ptr_transdata_1, "Failed to remove transdata1."), (void)RestoreOriginalValues();
                    return false);
  if (!split_k_mode) {
    FUSION_PASS_CHECK(graph->RemoveNode(node_ptr_transdata_out) == ge::GRAPH_FAILED,
                      OP_LOGE(node_ptr_transdata_out, "Failed to remove transdata_out."), (void)RestoreOriginalValues();
                      return false);
  }
  if (split_k_force_fp16_mode) {
    FUSION_PASS_CHECK(graph->RemoveNode(node_ptr_cast) == ge::GRAPH_FAILED,
                      OP_LOGE(node_ptr_cast, "Failed to remove cast."), (void)RestoreOriginalValues();
                      return false);
  }
  return true;
}

Status AAMatMulNzToNdFusionPass::DoFusion(ge::ComputeGraph* graph) {
  // step0: save original value
  if (!GetOriginalValue()) {
    return fe::NOT_CHANGED;
  }
  // step1~2 : Update Nodes' shape, range, format, etc.
  auto update_result = UpdateNodesInfo();
  // if fail to update and succeed to restore, return NOT_CHANGED
  if (update_result != fe::SUCCESS) {
    return update_result;
  }
  // step3: relink
  if (!RelinkNodesBeforeMatMul() || !RelinkNodesAfterMatMul()) {
    CUBE_INNER_ERR_REPORT(node_ptr_matmul, "failed to relink matmul");
    return fe::FAILED;
  }
  // step4: remove transdata or Cast which is redandunt to the graph
  if (!RemoveRedanduntNodes(graph)) {
    CUBE_INNER_ERR_REPORT(node_ptr_matmul, "failed to remove redandunt node for matmul");
    return fe::FAILED;
  }

  OP_LOGI(node_ptr_matmul, "succeed to execute fusion pass");
  return fe::SUCCESS;
}

Status AAMatMulNzToNdFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                        vector<ge::NodePtr>& /* fusionNodes */) {
  if (!NeedFusion(mapping)) {
    return NOT_CHANGED;
  }

  return DoFusion(&graph);
}
REGISTER_PASS("AAMatMulNzToNdFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, AAMatMulNzToNdFusionPass);
}  // namespace fe
