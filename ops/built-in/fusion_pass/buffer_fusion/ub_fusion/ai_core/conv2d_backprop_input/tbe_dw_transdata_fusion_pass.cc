/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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

#include "tbe_dw_transdata_fusion_pass.h"
#include <string>
#include <vector>
#include <unordered_set>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/util/platform_info.h"
#include "anchor_util.h"

namespace fe {
static const char kTypeTransData[] = "transdata1";
static const char kPatternDw[] = "conv2d_backprop_filter";
static const char kOpTypeTransData[] = "TransData";

static const int kFusionOpNumMax = 10;
static const string kFusedOpType = "FusedOp";
static const int kInputNum = 3;
static const int kDedyIndex = 2;
static const int kDynamicShapeDim = -1;
static vector<int64_t> kDilationAllOne = {1, 1, 1, 1};
static pair<int64_t, int64_t> kUnlimitedRange = {1, -1};

/*
   * @brief: define transdata_dw fusion pattern
   *
   *  TransData    TransData
   *      \           /
   *       \         /
   *    Conv2DBackporpFilter
   * pattern limit:
   *       1. both  inputs of dw must be with trans_data
   *       2. format is transfered from NCHW to NC1HWC0 by trans_data
   * @return BufferFusionPattern: return all valid patterns
   */
vector<BufferFusionPattern *> TbeDwTransDataFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;
  string pass_name = "TbeDwTransDataFusionPass";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name, kFusionOpNumMax);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  // define pattern rules
  pattern->AddOpDesc(kTypeTransData, {kOpTypeTransData}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                     TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_DYNAMIC, true)
          .AddOpDesc(kPatternDw, {OP_PATTERN_CONV_BACKPROP_FILTER}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                     TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_DYNAMIC)
          .SetHead({kTypeTransData})
          .SetOutputs(kTypeTransData, {kPatternDw}, TBE_OUTPUT_BRANCH_SINGLE, true)
          .SetOutputs(kPatternDw, {}, TBE_OUTPUT_BRANCH_SINGLE, true);

  patterns.push_back(pattern);

  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name.c_str());
  return patterns;
}


bool TbeDwTransDataFusionPass::CheckSupportTrans(const ge::NodePtr &node) const {
  return node->GetOpDesc()->GetInputDesc(0).GetFormat() == ge::FORMAT_NCHW &&
         node->GetOpDesc()->GetOutputDesc(0).GetFormat() == ge::FORMAT_NC1HWC0;
}


bool TbeDwTransDataFusionPass::CheckDwSupport(const vector<ge::NodePtr> &dw_nodes) const {
  ge::NodePtr dw_node = dw_nodes.at(0);
  FUSION_PASS_CHECK(
    dw_node->GetType() != "Conv2DBackpropFilter",
    OP_LOGW(kFusedOpType.c_str(), "only support op_type Conv2DBackpropFilter"),
    return false
  );
  int64_t group;
  FUSION_PASS_CHECK(
    !ge::AttrUtils::GetInt(dw_node->GetOpDesc(), "groups", group),
    OP_LOGW(kFusedOpType.c_str(), "dw get group failed"),
    return false
  );
  FUSION_PASS_CHECK(
    group != 1,
    OP_LOGW(kFusedOpType.c_str(), "only support group is 1"),
    return false
  );
  vector<int64_t> dilations;
  FUSION_PASS_CHECK(
    !ge::AttrUtils::GetListInt(dw_node->GetOpDesc(), "dilations", dilations),
    OP_LOGW(kFusedOpType.c_str(), "dw get dilation failed"),
    return false
  );
  FUSION_PASS_CHECK(
    dilations != kDilationAllOne,
    OP_LOGW(kFusedOpType.c_str(), "only support dilations is 1, 1, 1, 1"),
    return false
  );
  return true;
}


bool TbeDwTransDataFusionPass::CheckTransdataSupport(const vector<ge::NodePtr> &dw_nodes,
                                                     const vector<ge::NodePtr> &transdata_nodes,
                                                     vector <ge::NodePtr> &fusion_nodes) {
  ge::NodePtr dw_node = dw_nodes.at(0);
  ge::NodePtr transdata_node_1 = transdata_nodes.at(0);
  if (!CheckSupportTrans(transdata_node_1)) {
    OP_LOGW(kFusedOpType.c_str(), "transdata only support NCHW to 5HD");
    return false;
  }
  if (dw_node->GetInDataNodes().size() < kInputNum) {
    OP_LOGW(kFusedOpType.c_str(), "dw must have 3 inputs");
    return false;
  }
  ge::NodePtr transdata_node_2 = dw_node->GetInDataNodes().at(kDedyIndex);
  if (transdata_node_2->GetType() != kOpTypeTransData && !CheckSupportTrans(transdata_node_2)) {
    OP_LOGW(kFusedOpType.c_str(), "dw must have 2 transdata at the same time");
    return false;
  }
  fusion_nodes.push_back(transdata_node_2);
  return true;
}


bool TbeDwTransDataFusionPass::CheckUnlimitedRange(const ge::NodePtr &node) const {
  auto input_desc = GetCurrNodeMutableInputDesc(node, 0);
  vector<int64_t> input_dims = input_desc->GetOriginShape().GetDims();
  vector<pair<int64_t, int64_t>> ranges;
  FUSION_PASS_CHECK(
    input_desc->GetShapeRange(ranges) != GRAPH_SUCCESS,
    OP_LOGW(kFusedOpType.c_str(), "get range failed."),
    return false
  );

  for (auto input_dim : input_dims) {
    if (input_dim != kDynamicShapeDim) {
      OP_LOGW(kFusedOpType.c_str(), "shape is not -1");
      return false;
    }
  }
  for (auto range : ranges) {
    if (range != kUnlimitedRange) {
      OP_LOGW(kFusedOpType.c_str(), "range is not 1, -1");
      return false;
    }
  }
  return true;
}


/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeDwTransDataFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(kFusedOpType.c_str(), "Begin to do TbeDwTransDataFusionPass.");

  fusion_nodes = GetMatchedNodes(mapping);
  vector<ge::NodePtr> transdata_nodes = GetMatchedNodesByDescName(kTypeTransData, mapping);
  vector<ge::NodePtr> dw_nodes = GetMatchedNodesByDescName(kPatternDw, mapping);

  if (transdata_nodes.empty() || dw_nodes.empty()) {
    fusion_nodes.clear();
    OP_LOGW(kFusedOpType.c_str(), "transdata or dw node is empty");
    return SUCCESS;
  }
  if (!CheckDwSupport(dw_nodes)) {
    fusion_nodes.clear();
    OP_LOGW(kFusedOpType.c_str(), "dw check not supported");
    return SUCCESS;
  }
  if (!CheckTransdataSupport(dw_nodes, transdata_nodes, fusion_nodes)) {
    fusion_nodes.clear();
    OP_LOGW(kFusedOpType.c_str(), "transdata check not supported");
    return SUCCESS;
  }

  ge::NodePtr transdata_node_1 = transdata_nodes.at(0);
  ge::NodePtr dw_node = dw_nodes.at(0);
  ge::NodePtr transdata_node_2 = dw_node->GetInDataNodes().at(kDedyIndex);
  if (!CheckUnlimitedRange(transdata_node_1) ||!CheckUnlimitedRange(transdata_node_2)) {
    fusion_nodes.clear();
    OP_LOGW(kFusedOpType.c_str(), "fusion only supported in dynamic mode in NCHW with no range");
    return SUCCESS;
  }

  OP_LOGD(kFusedOpType.c_str(), "End to do TbeDwTransDataFusionPass!");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeDwTransDataFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeDwTransDataFusionPass);
}  // namespace fe
