/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 * \file matmul_transdata_ub_fusion.cpp
 * \brief matmul + transdata ub fusion pass
 * support dtype: float16
 * unsupported scene: one of m,k,n is not divisible by 16(float16)
 * Input1                   Input2         Bias(ND,optional)             Input1          Input2    Bias(ND,optional)
 *     \                      /                /                             \             /           /
 *   TransData(optional) TransData(optional)  /                               \           /           /
 *       \                  /                /                                 \         /           /
 * MatMul/MatMulV2/BatchMatMul/BatchMatMulV2(FRACTAL_NZ)       ->     MatMul/MatMulV2/BatchMatMul/BatchMatMulV2
 *             |                                                                       |
 *         TransData(optional)                                                         |
 *             |                                                                       |
 *         NetOutput                                                                NetOutput
 */
#include "matmul_transdata_ub_fusion.h"

#include <string>
#include <vector>

#include "anchor_util.h"
#include "common/lxfusion_json_util.h"
#include "common/util/platform_info.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "lx_fusion_func.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "../graph_fusion/ai_core/tbe_ops_pass_util.h"

namespace {
static const char kPatternMatmul[] = "matmul";

static const char kOpTypeTransdata[] = "TransData";
static const vector<string> kOpTypeMatmulList = {"MatMul", "MatMulV2", "BatchMatMul", "BatchMatMulV2"};

static const char kFormatNd[] = "ND";
static const char kFormatFractalNz[] = "FRACTAL_NZ";

static const char kFusedOpType[] = "matmul_transdata_fused_op";

static constexpr int kNumAlignHalf = 16;
static constexpr int kRIdxLast = -1;
static constexpr int kRIdxLastSecond = -2;
}  // namespace

namespace fe {
vector<BufferFusionPattern*> MatmulTransdataFusionPass::DefinePatterns() {
  /*
  * ===================== pattern =====================
  *
  * --> Matmul --> Transdata
  *
  * ===================================================
  */
  vector<BufferFusionPattern*> patterns;
  string pass_name = "MatmulTransdataFusion";

  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK(pattern == nullptr,
                    OP_LOGE(kFusedOpType, "new an pattern failed."),
                    return patterns);

  pattern->AddOpDesc(kPatternMatmul, {OP_PATTERN_MATMUL, OP_PATTERN_BATCH_MATMUL}, TBE_PATTERN_NUM_DEFAULT,
                     TBE_PATTERN_NUM_DEFAULT)
    .SetHead({kPatternMatmul});
  patterns.push_back(pattern);
  OP_LOGD(kFusedOpType, "Define pattern %s success.", pass_name.c_str());

  return patterns;
}

void MatmulTransdataFusionPass::IsOutTransdataCorrect(const ge::Node::Vistor<ge::NodePtr>& out_node_matmuls) {
  for (auto& out_node_matmul_ptr : out_node_matmuls) {
    FUSION_PASS_CHECK(out_node_matmul_ptr == nullptr, OP_LOGW(kFusedOpType,
                      "Get out node of matmul failed"), return);
    if (out_node_matmul_ptr->GetType() == kOpTypeTransdata) {
      out_transdata_ptr = out_node_matmul_ptr;
    }
  }
  if (out_transdata_ptr != nullptr && out_node_matmuls.size() != 1) {
    OP_LOGW(kFusedOpType, "Get out of matmul should only has one transdata.");
    out_transdata_ptr = nullptr;
  }
  return ;
}

bool MatmulTransdataFusionPass::CheckFormatOfTransData(const ge::NodePtr& node_ptr_transdata,
                                                       const char *expect_src_format,
                                                       const char *expect_dst_format) const {
  string src_format;
  string dst_format;
  FUSION_PASS_CHECK(node_ptr_transdata == nullptr,
                    OP_LOGD(kFusedOpType, "Skip this node."),
                    return true);
  FUSION_PASS_CHECK(!AttrUtils::GetStr(node_ptr_transdata->GetOpDesc(), "src_format", src_format),
                    OP_LOGE(kFusedOpType, "Unable to get the attribute src_format from TransData."),
                    return false);
  FUSION_PASS_CHECK(!AttrUtils::GetStr(node_ptr_transdata->GetOpDesc(), "dst_format", dst_format),
                    OP_LOGE(kFusedOpType, "Unable to get the attribute dst_format from TransData."),
                    return false);
  if (src_format != expect_src_format || dst_format != expect_dst_format) {
    return false;
  }
  return true;
}

bool MatmulTransdataFusionPass::IsLinkRelationshipCorrect() {
  FUSION_PASS_CHECK(matmul_node_ptr == nullptr, OP_LOGW(kFusedOpType,
                    "Get node matmul failed"), return false);

  auto in_node_matmuls = matmul_node_ptr->GetInDataNodes();
  FUSION_PASS_CHECK(in_node_matmuls.size() < 2, OP_LOGW(kFusedOpType,
                    "Matmul at least has 2 inputs"), return false);

  auto in_node_matmul_ptr_0 = in_node_matmuls.at(0);
  FUSION_PASS_CHECK(in_node_matmul_ptr_0 == nullptr, OP_LOGW(kFusedOpType,
                    "Get input node 0 of matmul failed"), return false);
  if (in_node_matmul_ptr_0->GetType() == kOpTypeTransdata && in_node_matmul_ptr_0->GetInDataNodes().size() == 1) {
    transdata_ptr_0 = in_node_matmul_ptr_0;
    auto in_nodes_transdata0_in = transdata_ptr_0->GetInDataNodes();
    in_ptr_0 = in_nodes_transdata0_in.at(0);
  } else {
    in_ptr_0 = in_node_matmul_ptr_0;
  }

  auto in_node_matmul_ptr_1 = in_node_matmuls.at(1);
  FUSION_PASS_CHECK(in_node_matmul_ptr_1 == nullptr, OP_LOGW(kFusedOpType,
                    "Get input node 1 of matmul failed"), return false);
  if (in_node_matmul_ptr_1->GetType() == kOpTypeTransdata && in_node_matmul_ptr_1->GetInDataNodes().size() == 1) {
    transdata_ptr_1 = in_node_matmul_ptr_1;
    auto in_nodes_transdata1_in = transdata_ptr_1->GetInDataNodes();
    in_ptr_1 = in_nodes_transdata1_in.at(0);
  } else {
    in_ptr_1 = in_node_matmul_ptr_1;
  }

  auto out_node_matmuls = matmul_node_ptr->GetOutDataNodes();

  IsOutTransdataCorrect(out_node_matmuls);

  FUSION_PASS_CHECK(out_transdata_ptr == nullptr && transdata_ptr_0 == nullptr && transdata_ptr_1 == nullptr,
                    OP_LOGW(kFusedOpType, "there is no transdata connecting with matmul"), return false);

  FUSION_PASS_CHECK(!CheckFormatOfTransData(transdata_ptr_0, kFormatNd, kFormatFractalNz),
                    OP_LOGW(kFusedOpType, "TransData(idx:0) format before MatMul node does not match."),
                    return false);

  FUSION_PASS_CHECK(!CheckFormatOfTransData(transdata_ptr_1, kFormatNd, kFormatFractalNz),
                    OP_LOGW(kFusedOpType, "TransData(idx:1) format before Matmul node does not match."),
                    return false);

  FUSION_PASS_CHECK(!CheckFormatOfTransData(out_transdata_ptr, kFormatFractalNz, kFormatNd),
                    OP_LOGW(kFusedOpType, "TransData format after Matmul node does not match."),
                    return false);
  return true;
}

bool MatmulTransdataFusionPass::IsOutOfInTransdataCorrect() {
  if (transdata_ptr_0 != nullptr) {
    FUSION_PASS_CHECK(transdata_ptr_0->GetOutDataNodesSize() != 1,
                      OP_LOGW(kFusedOpType, "Transdata(index: 0) before matmul node has %u outputs.",
                              transdata_ptr_0->GetOutDataNodesSize()),
                      transdata_ptr_0 = nullptr);
  }
  if (transdata_ptr_1 != nullptr) {
    FUSION_PASS_CHECK(transdata_ptr_1->GetOutDataNodesSize() != 1,
                      OP_LOGW(kFusedOpType, "Transdata(index: 1) before matmul node has %u outputs.",
                              transdata_ptr_1->GetOutDataNodesSize()),
                      transdata_ptr_1 = nullptr);
  }
  FUSION_PASS_CHECK(out_transdata_ptr == nullptr && transdata_ptr_0 == nullptr && transdata_ptr_1 == nullptr,
                    OP_LOGW(kFusedOpType, "there is no matched transdata connecting with matmul"), return false);
  return true;
}

bool MatmulTransdataFusionPass::IsStaticShape() const {
  return !HasUnKnowShape(in_ptr_0) && !HasUnKnowShape(in_ptr_1);
}

bool MatmulTransdataFusionPass::IsAligned() const {
  auto shape_data_0 = matmul_node_ptr->GetOpDesc()->MutableInputDesc(0)->GetOriginShape();
  if (transdata_ptr_0 != nullptr) {
    shape_data_0 = transdata_ptr_0->GetOpDesc()->MutableInputDesc(0)->GetOriginShape();
  }
  auto shape_data_1 = matmul_node_ptr->GetOpDesc()->MutableInputDesc(1)->GetOriginShape();
  if (transdata_ptr_1 != nullptr) {
    shape_data_1 = transdata_ptr_1->GetOpDesc()->MutableInputDesc(0)->GetOriginShape();
  }

  auto len_data_0 = shape_data_0.GetDimNum();
  auto len_data_1 = shape_data_1.GetDimNum();
  if (shape_data_0.GetDim(len_data_0 + kRIdxLast) % kNumAlignHalf == 0 &&
      shape_data_0.GetDim(len_data_0 + kRIdxLastSecond) % kNumAlignHalf == 0 &&
      shape_data_1.GetDim(len_data_1 + kRIdxLast) % kNumAlignHalf == 0 &&
      shape_data_1.GetDim(len_data_1 + kRIdxLastSecond) % kNumAlignHalf == 0) {
    return true;
  }
  return false;
}

bool MatmulTransdataFusionPass::NeedFusion() {
  FUSION_PASS_CHECK(!IsLinkRelationshipCorrect(),
                    OP_LOGW(kFusedOpType, "The connection relationship does not meet expectations."),
                    return false);

  FUSION_PASS_CHECK(!IsOutOfInTransdataCorrect(),
                    OP_LOGW(kFusedOpType, "The output number of in_transdatas do not meet expectations."),
                    return false);

  FUSION_PASS_CHECK(matmul_node_ptr->GetOpDesc()->MutableInputDesc(0)->GetDataType() != DT_FLOAT16 ||
                    matmul_node_ptr->GetOpDesc()->MutableInputDesc(1)->GetDataType() != DT_FLOAT16 ||
                    matmul_node_ptr->GetOpDesc()->MutableOutputDesc(0)->GetDataType() != DT_FLOAT16,
                    OP_LOGW(kFusedOpType, "Only support input and output data types as float16."),
                    return false);

  FUSION_PASS_CHECK(!IsStaticShape() || !IsAligned(),
                    OP_LOGW(kFusedOpType, "Static shape and unaligned scenes are not supported."),
                    return false);
  return true;
}

bool MatmulTransdataFusionPass::ModifyTransdataInControlEdge(const ge::NodePtr& node_ptr_transdata) const {
  for (auto& control_transdata_node : node_ptr_transdata->GetInControlNodes()) {
    FUSION_PASS_CHECK(control_transdata_node == nullptr,
                      OP_LOGD(kFusedOpType, "In Transdata has no in control node."), return true);
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(control_transdata_node->GetOutControlAnchor(),
                                                 node_ptr_transdata->GetInControlAnchor()) == ge::GRAPH_FAILED,
                      OP_LOGE(kFusedOpType, "remove control edge between input and transdata failed."),
                      return false);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(control_transdata_node->GetOutControlAnchor(),
                                              matmul_node_ptr->GetInControlAnchor()) == ge::GRAPH_FAILED,
                      OP_LOGE(kFusedOpType, "link control edge between input and matmul failed."),
                      return false);
  }
  return true;
}

bool MatmulTransdataFusionPass::ModifyTransdataOutControlEdge(const ge::NodePtr& node_ptr_transdata) const {
  for (auto& transdata_control_node : node_ptr_transdata->GetOutControlNodes()) {
    FUSION_PASS_CHECK(transdata_control_node == nullptr,
                      OP_LOGD(kFusedOpType, "In Transdata has no out control node."), return true);
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_transdata->GetOutControlAnchor(),
                                                 transdata_control_node->GetInControlAnchor()) == ge::GRAPH_FAILED,
                      OP_LOGE(kFusedOpType, "remove control edge between transdata and output failed."),
                      return false);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(matmul_node_ptr->GetOutControlAnchor(),
                                              transdata_control_node->GetInControlAnchor()) == ge::GRAPH_FAILED,
                      OP_LOGE(kFusedOpType, "link control edge between matmul and output failed."),
                      return false);
  }
  return true;
}

bool MatmulTransdataFusionPass::DelInputTransdata(ge::NodePtr& node_ptr_transdata, const uint32_t idx) {
  if (node_ptr_transdata != nullptr) {
    // step0: update shape, range, format
    vector<pair<int64_t, int64_t>> range_data;
    auto in_transdata_desc = node_ptr_transdata->GetOpDesc()->MutableInputDesc(0);
    matmul_node_ptr->GetOpDesc()->MutableInputDesc(idx)->SetShape(in_transdata_desc->GetShape());
    FUSION_PASS_CHECK(in_transdata_desc->GetShapeRange(range_data) == ge::GRAPH_FAILED,
                      OP_LOGE(kFusedOpType, "Failed to get input shape range of in_transdata:%u", idx),
                      return false);
    FUSION_PASS_CHECK(matmul_node_ptr->GetOpDesc()->MutableInputDesc(idx)->SetShapeRange(range_data) ==
                      ge::GRAPH_FAILED,
                      OP_LOGE(kFusedOpType, "Failed to set input shape range of in_transdata:%u", idx),
                      return false);
    matmul_node_ptr->GetOpDesc()->MutableInputDesc(idx)->SetFormat(FORMAT_ND);

    // step1: relink
    auto in_transdata_peer_out = GetPeerOutAnchorWithInDataAnchor(node_ptr_transdata, 0);
    FUSION_PASS_CHECK(in_transdata_peer_out == nullptr,
                      OP_LOGE(kFusedOpType, "Failed to get peer out anchor of in_transdata:%u", idx),
                      return false);
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(in_transdata_peer_out,
                                                 node_ptr_transdata->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                      OP_LOGE(kFusedOpType, "Failed to remove edge between data and transdata."),
                      return false);
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(node_ptr_transdata->GetOutDataAnchor(0),
                                                 matmul_node_ptr->GetInDataAnchor(idx)) == ge::GRAPH_FAILED,
                      OP_LOGE(kFusedOpType, "Failed to remove edge between input and transdata."),
                      return false);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(in_transdata_peer_out,
                                              matmul_node_ptr->GetInDataAnchor(idx)) == ge::GRAPH_FAILED,
                      OP_LOGE(kFusedOpType, "Failed to add edge between input and matmul."),
                      return false);

    // step2: modify control edge
    FUSION_PASS_CHECK(!ModifyTransdataInControlEdge(node_ptr_transdata),
                      OP_LOGE(kFusedOpType,
                              "Failed to modify control edge between input and intransdata:%u.", idx),
                      return false);
    FUSION_PASS_CHECK(!ModifyTransdataOutControlEdge(node_ptr_transdata),
                      OP_LOGE(kFusedOpType,
                              "Failed to modify control edge between intransdata:%u. and out.", idx),
                      return false);

    // step3: remove transdata
    auto graph = matmul_node_ptr->GetOwnerComputeGraph();
    FUSION_PASS_CHECK(graph == nullptr,
                      OP_LOGE(kFusedOpType, "Failed to get compute graph of matmul."),
                      return false);
    FUSION_PASS_CHECK(graph->RemoveNode(node_ptr_transdata) == ge::GRAPH_FAILED,
                      OP_LOGE(kFusedOpType, "Failed to remove transdata:%u.", idx),
                      return false);
  }
  return true;
}

bool MatmulTransdataFusionPass::DelOutputTransdata() const {
  if (out_transdata_ptr != nullptr) {
    // step0: update shape, range, format
    vector<pair<int64_t, int64_t>> range_data;
    auto out_transdata_desc = out_transdata_ptr->GetOpDesc()->MutableOutputDesc(0);
    matmul_node_ptr->GetOpDesc()->MutableOutputDesc(0)->SetShape(out_transdata_desc->GetShape());
    FUSION_PASS_CHECK(out_transdata_desc->GetShapeRange(range_data) == ge::GRAPH_FAILED,
                      OP_LOGE(kFusedOpType, "Failed to get output shape range of out_transdata."),
                      return false);
    FUSION_PASS_CHECK(matmul_node_ptr->GetOpDesc()->MutableOutputDesc(0)->SetShapeRange(range_data) ==
                      ge::GRAPH_FAILED,
                      OP_LOGE(kFusedOpType, "Failed to set out shape range of out_transdata."),
                      return false);
    matmul_node_ptr->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);

    // step1: relink, output num of transdata can be more than 1
    auto out_transdata_out_anchor = out_transdata_ptr->GetOutDataAnchor(0);
    auto in_nodes_size = out_transdata_out_anchor->GetPeerInDataNodesSize();
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(matmul_node_ptr->GetOutDataAnchor(0),
                      out_transdata_ptr->GetInDataAnchor(0)) == ge::GRAPH_FAILED,
                      OP_LOGE(kFusedOpType, "Failed to remove edge between matmul and out transdata."),
                      return false);

    for (uint32_t idx = 0; idx < in_nodes_size; ++idx) {
      auto out_transdata_peer_in = GetPeerInAnchorByOutDataAnchor(out_transdata_out_anchor, 0);
      FUSION_PASS_CHECK(out_transdata_peer_in == nullptr,
                        OP_LOGE(kFusedOpType, "Failed to get peer out anchor of in_transdata."),
                        return false);
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_transdata_out_anchor,
                                                   out_transdata_peer_in) == ge::GRAPH_FAILED,
                        OP_LOGE(kFusedOpType, "Failed to remove edge between out transdata and data."),
                        return false);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(matmul_node_ptr->GetOutDataAnchor(0),
                                                out_transdata_peer_in) == ge::GRAPH_FAILED,
                        OP_LOGE(kFusedOpType, "Failed to add edge between matmul and out data."),
                        return false);
    }

    // step2: modify control edge
    FUSION_PASS_CHECK(!ModifyTransdataInControlEdge(out_transdata_ptr),
                      OP_LOGE(kFusedOpType,
                              "Failed to modify control edge between input and out transdata."),
                      return false);
    FUSION_PASS_CHECK(!ModifyTransdataOutControlEdge(out_transdata_ptr),
                      OP_LOGE(kFusedOpType,
                              "Failed to modify control edge between out transdata and out."),
                      return false);

    // step3: remove transdata
    auto graph = matmul_node_ptr->GetOwnerComputeGraph();
    FUSION_PASS_CHECK(graph == nullptr,
                      OP_LOGE(kFusedOpType, "Failed to get compute graph of matmul."),
                      return false);
    FUSION_PASS_CHECK(graph->RemoveNode(out_transdata_ptr) == ge::GRAPH_FAILED,
                      OP_LOGE(kFusedOpType, "Failed to remove out transdata."),
                      return false);
  }
  return true;
}

bool MatmulTransdataFusionPass::DoFusion() {
  // delete input transdata
  FUSION_PASS_CHECK(!DelInputTransdata(transdata_ptr_0, 0),
                    OP_LOGW(kFusedOpType, "Failed to del transdata:0."),
                    return false);
  FUSION_PASS_CHECK(!DelInputTransdata(transdata_ptr_1, 1),
                    OP_LOGW(kFusedOpType, "Failed to del transdata:1."),
                    return false);

  // delete output transdata
  FUSION_PASS_CHECK(!DelOutputTransdata(),
                    OP_LOGW(kFusedOpType, "Failed to del output transdata"),
                    return false);
  return true;
}

void MatmulTransdataFusionPass::SetSplitInfo(const BufferFusionMapping &mapping,
                                             std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> matmul_nodes = GetMatchedNodesByDescName(kPatternMatmul, mapping);
  FUSION_PASS_CHECK(matmul_nodes.empty(),
                    OP_LOGW(kFusedOpType, "Matmul node not matched in SetSplitInfo"),
                    return);

  vector<AxisSplitMap> split_maps;
  OpL1FusionType fusion_type = L1FUSION_DISABLE;
  int64_t min_tbe_l1space = 0;

  FUSION_PASS_CHECK(!GetSplitMap(split_maps, matmul_nodes[0], kFusedOpType, fusion_type, min_tbe_l1space),
                    OP_LOGW(kFusedOpType, "get split_maps of matmul node fail in SetSplitInfo"),
                    return);

  SetSplitMap(split_maps, fusion_nodes, kFusedOpType, fusion_type, min_tbe_l1space);
}

Status MatmulTransdataFusionPass::GetFusionNodes(const BufferFusionMapping& mapping,
                                                 vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(kFusedOpType, "Begin to do MatmulTransdataFusion!");

  in_ptr_0 = nullptr;
  transdata_ptr_0 = nullptr;
  in_ptr_1 = nullptr;
  transdata_ptr_1 = nullptr;
  matmul_node_ptr = nullptr;
  out_transdata_ptr = nullptr;

  vector<ge::NodePtr> matmul_nodes = GetMatchedNodesByDescName(kPatternMatmul, mapping);

  FUSION_PASS_CHECK(matmul_nodes.empty(),
                    OP_LOGE(kFusedOpType, "MatMul node is not matched."),
                    return SUCCESS);

  matmul_node_ptr = matmul_nodes.at(0);

  FUSION_PASS_CHECK(std::find(kOpTypeMatmulList.begin(), kOpTypeMatmulList.end(), matmul_node_ptr->GetType()) ==
                    kOpTypeMatmulList.end(),
                    OP_LOGW(kFusedOpType, "Failed to match matmul node. It's %s node.",
                            matmul_node_ptr->GetType().c_str()),
                    return SUCCESS);

  FUSION_PASS_CHECK(!NeedFusion(),
                    OP_LOGW(kFusedOpType, "no need do ub fusion"),
                    return SUCCESS);

  FUSION_PASS_CHECK(!DoFusion(),
                    OP_LOGW(kFusedOpType, "del transdata failed"),
                    return FAILED);

  ge::AttrUtils::SetBool(matmul_node_ptr->GetOpDesc(), NEED_RE_PRECOMPILE, true);
  ge::AttrUtils::SetBool(matmul_node_ptr->GetOwnerComputeGraph(), NEED_RE_PRECOMPILE, true);
  fusion_nodes = GetMatchedNodes(mapping);
  SetSplitInfo(mapping, fusion_nodes);
  OP_LOGD(kFusedOpType, "End to do MatmulTransdataFusionPass!");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("MatmulTransdataFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            MatmulTransdataFusionPass);
}  // namespace fe
