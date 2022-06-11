/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

#include "tbe_conv_bnreduce_fusion_pass.h"
#include <string>
#include <vector>
#include <memory>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "conv2d_slice_info_cal_base.h"

namespace fe {
using std::vector;

namespace {
static const char kPatternConv[] = "convolution";
static const char kPatternBnreduce[] = "bnreduce";
static const char kPatternOutput1[] = "OUTPUT1";
static const char kPatternOutput2[] = "OUTPUT2";
static const char kPatternOutput3[] = "OUTPUT3";
static const string kFusedOpType = "FusedOp";
static const string OP_PATTERN_TUPLEREDUCE = "tuple_reduce";
}

/*
 * @brief:  define convolution and BNReduce input op fusion pattern
 *
 *   Convolution-->BNReduce
 *
 * fusion node: BNReduce, Convolution
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> ConvBnreduceFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;
  string pass_name = "TbeConvBNReduceFusionPass";

  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name.c_str());

  // define pattern rules
  pattern->AddOpDesc(kPatternBnreduce, {OP_PATTERN_BNREDUCE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOutput1, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOutput2, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOutput3, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternBnreduce, kPatternOutput1, kPatternOutput2, kPatternOutput3},
                  TBE_OUTPUT_BRANCH_MULTI);

  patterns.push_back(pattern);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name.c_str());
  // define pattern rules
  string pass_name1 = "TbeConvBNReduce3outputFusionPass";
  BufferFusionPattern *pattern1 = new (std::nothrow) BufferFusionPattern(pass_name1);
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name1.c_str());
  pattern1->AddOpDesc(kPatternBnreduce, {OP_PATTERN_BNREDUCE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOutput1, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOutput2, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternBnreduce, kPatternOutput1, kPatternOutput2},
                  TBE_OUTPUT_BRANCH_MULTI);
  patterns.push_back(pattern1);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name1.c_str());

  string pass_name2 = "TbeConvBNReduceTuplePatternFusionPass";
  BufferFusionPattern *pattern2 = new (std::nothrow) BufferFusionPattern(pass_name2);
  FUSION_PASS_CHECK(pattern2 == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name2.c_str());
  // define pattern rules
  pattern2->AddOpDesc(kPatternBnreduce, {OP_PATTERN_TUPLEREDUCE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOutput1, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOutput2, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOutput3, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternBnreduce, kPatternOutput1, kPatternOutput2, kPatternOutput3},
                  TBE_OUTPUT_BRANCH_MULTI);
  patterns.push_back(pattern2);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name2.c_str());

  // define pattern rules
  string pass_name3 = "TbeConvBNReduce3TuplePatternoutputFusionPass";
  BufferFusionPattern *pattern3 = new (std::nothrow) BufferFusionPattern(pass_name3);
  FUSION_PASS_CHECK(pattern3 == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name3.c_str());
  pattern3->AddOpDesc(kPatternBnreduce, {OP_PATTERN_TUPLEREDUCE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOutput1, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOutput2, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternBnreduce, kPatternOutput1, kPatternOutput2},
                  TBE_OUTPUT_BRANCH_MULTI);
  patterns.push_back(pattern3);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name3.c_str());
  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status ConvBnreduceFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                              vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(kFusedOpType.c_str(), "Begin to do ConvBNReduce!");
  fusion_nodes = GetMatchedNodes(mapping);
  // multi input node can not be fused except head node
  for (auto &item : mapping) {
    auto opdesc = find(item.first->types.begin(), item.first->types.end(), TBE_PATTERN_OUTPUT_NODE);
    if (opdesc != item.first->types.end()) {
      for (auto &node : item.second) {
        auto node_ptr = find(fusion_nodes.begin(), fusion_nodes.end(), node);
        fusion_nodes.erase(node_ptr);
      }
    }
  }
  for (auto &item : mapping) {
    const BufferFusionOpDesc *op_desc = item.first;
    if (op_desc != nullptr && op_desc->types[0] == OP_PATTERN_TUPLEREDUCE) {
      ge::NodePtr node = item.second[0];
      if (node == nullptr) {
        return FAILED;
      }
      if (node->GetType() != "BNTrainingReduce") {
        fusion_nodes.clear();
        OP_LOGD(fused_op_type_.c_str(), "BNReduce is op [%s, %s], skip fusion.", node->GetName().c_str(), node->GetType().c_str());
        break;
      }
    }
  }
  OP_LOGD(kFusedOpType.c_str(), "End to do ConvBNReduce!");
  return SUCCESS;
}

Status ConvBnreduceFusionPass::CalcFusionOpSliceInfo(vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_slice_info)
{
  OP_LOGD(fused_op_type_.c_str(), "start calc slice info.");
  std::unique_ptr<ConvSliceInfoCalBase> pConvSliceInfoCal = nullptr;
  pConvSliceInfoCal.reset(new (std::nothrow) ConvSliceInfoCalBase());
  CONV_RET_IF_SMART_PTR_IS_NULL(pConvSliceInfoCal);
  Status ret = pConvSliceInfoCal->ConvCalcFusionOpSliceInfo(fusion_nodes, op_slice_info, fused_op_type_);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(fused_op_type_.c_str(), "calc fusion op slice info failed."), return FAILED);
  OP_LOGD(fused_op_type_.c_str(), "end calc slice info.");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeConvBnreduceFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            ConvBnreduceFusionPass);
}  // namespace fe
