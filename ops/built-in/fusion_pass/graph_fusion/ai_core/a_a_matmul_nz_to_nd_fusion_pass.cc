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
 * Data(ND)         Data(ND)   Data(ND,optional)                 Data(ND)         Data(ND)   Data(ND,optional)
 *     \              /          /                                   \                /        /
 *   TransData   TransData      /                                     \              /        /
 *       \          /          /                                       \            /        /
 * MatMul/MatMulV2/BatchMatMul/BatchMatMulV2(FRACTAL_NZ)  ->   MatMul/MatMulV2/BatchMatMul/BatchMatMulV2(ND)
 *             |                                                             |
 *         TransData                                                         |
 *             |                                                             |
 *         NetOutput                                                     NetOutput
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
static const string kDescMatMul = "MatMul";
static const string kDescTransdata0 = "Transdata0";
static const string kDescTransdata1 = "Transdata1";
static const string kDescTransdataOut = "TransdataOut";

static const string kOpTypeMatMul = "MatMul";
static const string kOpTypeMatMulV2 = "MatMulV2";
static const string kOpTypeBatchMatMul = "BatchMatMul";
static const string kOpTypeBatchMatMulV2 = "BatchMatMulV2";
static const string kOpTypeTransData = "TransData";
static const string kOpTypeNetOutput = "NetOutput";
static const string kOpTypeData = "Data";

static const string kFormatNd = "ND";
static const string kFormatFractalNz = "FRACTAL_NZ";

static const int kNumDataNodes = 2;
static const int kNumAlignHalf = 16;
static const int kRIdxLast = -1;
static const int kRIdxLastSecond = -2;

static const vector<string> whitelist_op_type = {kOpTypeMatMul, kOpTypeMatMulV2, kOpTypeBatchMatMul,
                                                 kOpTypeBatchMatMulV2};

vector<FusionPattern*> AAMatMulNzToNdFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern(kNameFusionPass);
  FUSION_PASS_CHECK(pattern == nullptr,
                    OP_LOGW(kNameFusionPass.c_str(), "pattern is nullptr, Create pattern not success!"),
                    return patterns);

  OP_LOGD(kNameFusionPass.c_str(), "Start to define pattern.");

  pattern->AddOpDesc(kDescMatMul, {kOpTypeMatMul, kOpTypeMatMulV2, kOpTypeBatchMatMul, kOpTypeBatchMatMulV2})
      .AddOpDesc(kDescTransdata0, {kOpTypeTransData})
      .AddOpDesc(kDescTransdata1, {kOpTypeTransData})
      .AddOpDesc(kDescTransdataOut, {kOpTypeTransData})
      .SetInputs(kDescMatMul, {kDescTransdata0, kDescTransdata1})
      .SetInputs(kDescTransdataOut, {kDescMatMul})
      .SetOutput(kDescTransdataOut);
  patterns.push_back(pattern);
  OP_LOGD(kNameFusionPass.c_str(), "End to define pattern.");
  return patterns;
}

bool AAMatMulNzToNdFusionPass::CheckFormatOfTransData(const NodePtr node_ptr_transdata, const string& expect_src_format,
                                                      const string& expect_dst_format) {
  string src_format;
  string dst_format;
  FUSION_PASS_CHECK(!AttrUtils::GetStr(node_ptr_transdata->GetOpDesc(), "src_format", src_format),
                    OP_LOGE(kNameFusionPass.c_str(), "Unable to get the attribute src_format from TransData."),
                    return false);
  FUSION_PASS_CHECK(!AttrUtils::GetStr(node_ptr_transdata->GetOpDesc(), "dst_format", dst_format),
                    OP_LOGE(kNameFusionPass.c_str(), "Unable to get the attribute dst_format from TransData."),
                    return false);
  if (src_format != expect_src_format || dst_format != expect_dst_format) {
    return false;
  }
  return true;
}

bool AAMatMulNzToNdFusionPass::IsNumOfNodesOutCorrect() {
  auto out_nodes_data_0 = node_ptr_data_0->GetOutDataNodes().size();
  FUSION_PASS_CHECK(out_nodes_data_0 != 1, OP_LOGW(kNameFusionPass.c_str(),
                    "Data(index: 0) before matmul node should only have 1 output, actual %zu.",
                    out_nodes_data_0), return false);

  auto out_nodes_data_1 = node_ptr_data_1->GetOutDataNodes().size();
  FUSION_PASS_CHECK(out_nodes_data_1 != 1, OP_LOGW(kNameFusionPass.c_str(),
                    "Data(index: 1) before matmul node should only have 1 output, actual %zu.",
                    out_nodes_data_1), return false);

  auto out_nodes_transdata_0 = node_ptr_transdata_0->GetOutDataNodes().size();
  FUSION_PASS_CHECK(out_nodes_transdata_0 != 1, OP_LOGW(kNameFusionPass.c_str(),
                    "Transdata(index: 0) before matmul node should only have 1 output, actual %zu.",
                    out_nodes_transdata_0), return false);

  auto out_nodes_transdata_1 = node_ptr_transdata_1->GetOutDataNodes().size();
  FUSION_PASS_CHECK(out_nodes_transdata_1 != 1, OP_LOGW(kNameFusionPass.c_str(),
                    "Transdata(index: 1) before matmul node should only have 1 output, actual %zu.",
                    out_nodes_transdata_1), return false);

  auto out_nodes_matmul = node_ptr_matmul->GetOutDataNodes().size();
  FUSION_PASS_CHECK(out_nodes_matmul != 1,
                    OP_LOGW(kNameFusionPass.c_str(), "Matmul node should only have 1 output, actual %zu.",
                    out_nodes_matmul), return false);

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

bool AAMatMulNzToNdFusionPass::IsLinkRelationshipCorrect(const Mapping& mapping) {
  node_ptr_matmul = GetNodeFromMapping(kDescMatMul, mapping);
  FUSION_PASS_CHECK(node_ptr_matmul == nullptr, OP_LOGW(kNameFusionPass.c_str(),
                    "Get node matmul failed"), return false);
  node_ptr_transdata_0 = GetNodeFromMapping(kDescTransdata0, mapping);
  FUSION_PASS_CHECK(node_ptr_transdata_0 == nullptr, OP_LOGW(kNameFusionPass.c_str(),
                    "Get node transdata(idx: 0) before matmul failed"), return false);
  node_ptr_transdata_1 = GetNodeFromMapping(kDescTransdata1, mapping);
  FUSION_PASS_CHECK(node_ptr_transdata_1 == nullptr, OP_LOGW(kNameFusionPass.c_str(),
                    "Get node transdata(idx: 1) before matmul failed"), return false);
  node_ptr_transdata_out = GetNodeFromMapping(kDescTransdataOut, mapping);
  FUSION_PASS_CHECK(node_ptr_transdata_out == nullptr, OP_LOGW(kNameFusionPass.c_str(),
                    "Get node transdata after matmul failed"), return false);

  auto out_nodes_transdata_out = node_ptr_transdata_out->GetOutDataNodes();
  FUSION_PASS_CHECK(out_nodes_transdata_out.size() != 1, OP_LOGW(kNameFusionPass.c_str(),
                    "Transdata after matmul node should only have 1 output, actual %zu.",
                    out_nodes_transdata_out.size()), return false);
  node_ptr_netoutput = out_nodes_transdata_out.at(0);

  auto in_nodes_transdata0_in = node_ptr_transdata_0->GetInDataNodes();
  FUSION_PASS_CHECK(in_nodes_transdata0_in.size() != 1, OP_LOGW(kNameFusionPass.c_str(),
                    "Transdata(idx: 0) before matmul node should only have 1 input, actual %zu.",
                    in_nodes_transdata0_in.size()), return false);
  node_ptr_data_0 = in_nodes_transdata0_in.at(0);

  auto in_nodes_transdata1_in = node_ptr_transdata_1->GetInDataNodes();
  FUSION_PASS_CHECK(in_nodes_transdata1_in.size() != 1, OP_LOGW(kNameFusionPass.c_str(),
                    "Transdata(idx: 1) before matmul node should only have 1 input, actual %zu.",
                    in_nodes_transdata1_in.size()), return false);
  node_ptr_data_1 = in_nodes_transdata1_in.at(0);

  FUSION_PASS_CHECK(node_ptr_netoutput->GetType() != kOpTypeNetOutput,
                    OP_LOGW(kNameFusionPass.c_str(), "The NetOutput operator does not match."), return false);
  FUSION_PASS_CHECK(node_ptr_data_0->GetType() != kOpTypeData,
                    OP_LOGW(kNameFusionPass.c_str(), "Data(idx:0) before Matmul node does not match."), return false);
  FUSION_PASS_CHECK(node_ptr_data_1->GetType() != kOpTypeData,
                    OP_LOGW(kNameFusionPass.c_str(), "Data(idx:1) before Matmul node does not match."), return false);

  FUSION_PASS_CHECK(!CheckFormatOfTransData(node_ptr_transdata_0, kFormatNd, kFormatFractalNz),
                    OP_LOGW(kNameFusionPass.c_str(), "TransData(idx:0) before MatMul node does not match."),
                    return false);

  FUSION_PASS_CHECK(!CheckFormatOfTransData(node_ptr_transdata_1, kFormatNd, kFormatFractalNz),
                    OP_LOGW(kNameFusionPass.c_str(), "TransData(idx:1) before Matmul node does not match."),
                    return false);

  FUSION_PASS_CHECK(!CheckFormatOfTransData(node_ptr_transdata_out, kFormatFractalNz, kFormatNd),
                    OP_LOGW(kNameFusionPass.c_str(), "TransData after Matmul node does not match."),
                    return false);

  return true;
}

bool AAMatMulNzToNdFusionPass::NeedFusion(const Mapping& mapping) {
  // Not support: cube_vector_split
  PlatformInfo platform_info;
  OptionalInfo opti_compilation_info;
  FUSION_PASS_CHECK(
      PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, opti_compilation_info) != SUCCESS,
      OP_LOGE(kNameFusionPass.c_str(), "Get platform_info failed."), return false);
  bool cube_vector_split_flag = platform_info.ai_core_spec.cube_vector_split;
  FUSION_PASS_CHECK(cube_vector_split_flag,
                    OP_LOGW(kNameFusionPass.c_str(), "Scenario where cube and vector are separated is not supported."),
                    return false);

  FUSION_PASS_CHECK(!IsLinkRelationshipCorrect(mapping),
                    OP_LOGW(kNameFusionPass.c_str(), "The connection relationship does not meet expectations."),
                    return false);

  FUSION_PASS_CHECK(!IsNumOfNodesOutCorrect(),
                    OP_LOGW(kNameFusionPass.c_str(), "The number of nodes does not meet expectations."), return false);

  FUSION_PASS_CHECK(node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->GetDataType() != DT_FLOAT16 ||
                        node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->GetDataType() != DT_FLOAT16 ||
                        node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->GetDataType() != DT_FLOAT16,
                    OP_LOGW(kNameFusionPass.c_str(), "Only support input and output data types as float16."),
                    return false);

  FUSION_PASS_CHECK(IsStaticShape() && !IsAligned(),
                    OP_LOGW(kNameFusionPass.c_str(), "Static shape and unaligned scenes are not supported."),
                    return false);

  return true;
}

void AAMatMulNzToNdFusionPass::RestoreOriginalValues() {
  node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->SetShape(in_shape_matmul_0);
  node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->SetShape(in_shape_matmul_1);
  node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->SetShape(out_shape_matmul_0);

  FUSION_PASS_CHECK(
      node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->SetShapeRange(in_range_matmul_0) == ge::GRAPH_FAILED,
      OP_LOGE(kNameFusionPass.c_str(), "Failed to restore first input shape range of matmul."), return );
  FUSION_PASS_CHECK(
      node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->SetShapeRange(in_range_matmul_1) == ge::GRAPH_FAILED,
      OP_LOGE(kNameFusionPass.c_str(), "Failed to restore second input shape range of matmul."), return );
  FUSION_PASS_CHECK(
      node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->SetShapeRange(out_range_matmul_0) == ge::GRAPH_FAILED,
      OP_LOGE(kNameFusionPass.c_str(), "Failed to restore output shape range of matmul."), return );

  node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_FRACTAL_NZ);
  node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->SetFormat(FORMAT_FRACTAL_NZ);
  node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_FRACTAL_NZ);
}

Status AAMatMulNzToNdFusionPass::DoFusion(ge::ComputeGraph& graph) {
  // step0: save original value
  in_shape_matmul_0 = node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->GetShape();
  in_shape_matmul_1 = node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->GetShape();
  out_shape_matmul_0 = node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->GetShape();
  FUSION_PASS_CHECK(
      node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->GetShapeRange(in_range_matmul_0) == ge::GRAPH_FAILED,
      OP_LOGE(kNameFusionPass.c_str(), "Failed to get first input shape range of matmul."), return fe::FAILED);
  FUSION_PASS_CHECK(
      node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->GetShapeRange(in_range_matmul_1) == ge::GRAPH_FAILED,
      OP_LOGE(kNameFusionPass.c_str(), "Failed to get second input shape range of matmul."), return fe::FAILED);
  FUSION_PASS_CHECK(
      node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->GetShapeRange(out_range_matmul_0) == ge::GRAPH_FAILED,
      OP_LOGE(kNameFusionPass.c_str(), "Failed to get output shape range of matmul."), return fe::FAILED);

  // step1: update shape, range, format
  vector<pair<int64_t, int64_t>> range_data_0;
  auto out_desc_data_0 = node_ptr_data_0->GetOpDesc()->MutableOutputDesc(0);
  node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->SetShape(out_desc_data_0->GetShape());
  FUSION_PASS_CHECK(out_desc_data_0->GetShapeRange(range_data_0) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to get shape range of data0."), RestoreOriginalValues();
                    return fe::FAILED);
  FUSION_PASS_CHECK(node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->SetShapeRange(range_data_0) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to set first shape range of MatMul."),
                    RestoreOriginalValues();
                    return fe::FAILED);

  vector<pair<int64_t, int64_t>> range_data_1;
  auto out_desc_data_1 = node_ptr_data_1->GetOpDesc()->MutableOutputDesc(0);
  node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->SetShape(out_desc_data_1->GetShape());
  FUSION_PASS_CHECK(out_desc_data_1->GetShapeRange(range_data_1) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to get shape range of data1."), RestoreOriginalValues();
                    return fe::FAILED);
  FUSION_PASS_CHECK(node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->SetShapeRange(range_data_1) == ge::GRAPH_FAILED,
                    OP_LOGW(kNameFusionPass.c_str(), "Failed to set second shape range of MatMul."),
                    RestoreOriginalValues();
                    return fe::FAILED);

  vector<pair<int64_t, int64_t>> range_netoutput;
  auto out_desc_netoutput = node_ptr_netoutput->GetOpDesc()->MutableInputDesc(0);
  node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->SetShape(out_desc_netoutput->GetShape());
  FUSION_PASS_CHECK(out_desc_netoutput->GetShapeRange(range_netoutput) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to get shape range of NetOutput."),
                    RestoreOriginalValues();
                    return fe::FAILED);
  FUSION_PASS_CHECK(
      node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->SetShapeRange(range_netoutput) == ge::GRAPH_FAILED,
      OP_LOGW(kNameFusionPass.c_str(), "Failed to set output shape range of MatMul."), RestoreOriginalValues();
      return fe::FAILED);

  node_ptr_matmul->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  node_ptr_matmul->GetOpDesc()->MutableInputDesc(1)->SetFormat(FORMAT_ND);
  node_ptr_matmul->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);

  // step2: relink
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_data_0->GetOutDataAnchor(0),
                                               node_ptr_transdata_0->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to remove edge between data0 and transdata0."),
                    RestoreOriginalValues();
                    return fe::FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_transdata_0->GetOutDataAnchor(0),
                                               node_ptr_matmul->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to remove edge between transdata0 and matmul."),
                    RestoreOriginalValues();
                    return fe::FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(node_ptr_data_0->GetOutDataAnchor(0),
                                            node_ptr_matmul->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to add edge between data0 and matmul."),
                    RestoreOriginalValues();
                    return fe::FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_data_1->GetOutDataAnchor(0),
                                               node_ptr_transdata_1->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to remove edge between data1 and transdata1."),
                    RestoreOriginalValues();
                    return fe::FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_transdata_1->GetOutDataAnchor(0),
                                               node_ptr_matmul->GetInDataAnchor(1)) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to remove edge between transdata1 and matmul."),
                    RestoreOriginalValues();
                    return fe::FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(node_ptr_data_1->GetOutDataAnchor(0),
                                            node_ptr_matmul->GetInDataAnchor(1)) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to add edge between data1 and matmul."),
                    RestoreOriginalValues();
                    return fe::FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_matmul->GetOutDataAnchor(0),
                                               node_ptr_transdata_out->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to remove edge between matmul and transdata_out."),
                    RestoreOriginalValues();
                    return fe::FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_transdata_out->GetOutDataAnchor(0),
                                               node_ptr_netoutput->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to remove edge between transdata_out and netoutput."),
                    RestoreOriginalValues();
                    return fe::FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(node_ptr_matmul->GetOutDataAnchor(0),
                                            node_ptr_netoutput->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to add edge between matmul and netoutput."),
                    RestoreOriginalValues();
                    return fe::FAILED);

  // step3: remove transdata
  FUSION_PASS_CHECK(graph.RemoveNode(node_ptr_transdata_0) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to remove transdata0."), RestoreOriginalValues();
                    return fe::FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(node_ptr_transdata_1) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to remove transdata1."), RestoreOriginalValues();
                    return fe::FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(node_ptr_transdata_out) == ge::GRAPH_FAILED,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to remove transdata_out."), RestoreOriginalValues();
                    return fe::FAILED);

  return fe::SUCCESS;
}

Status AAMatMulNzToNdFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                        vector<ge::NodePtr>& /* fusionNodes */) {
  if (!NeedFusion(mapping)) {
    return NOT_CHANGED;
  }

  return DoFusion(graph);
}
REGISTER_PASS("AAMatMulNzToNdFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, AAMatMulNzToNdFusionPass);
}  // namespace fe
