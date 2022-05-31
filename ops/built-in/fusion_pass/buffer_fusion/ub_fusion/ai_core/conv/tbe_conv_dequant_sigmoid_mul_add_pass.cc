/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

#include "tbe_conv_dequant_sigmoid_mul_add_pass.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"
#include "lx_fusion_func.h"

namespace fe {
using std::vector;
static const string kPatternConv = "convolution";
static const string kPatternDeq = "dequant";
static const string kPatternOtherInput = "otherInput";
static const string kPatternOtherInput2 = "otherInput2";
static const string kPatternSigmoid = "sigmoid";
static const string kPatternMul = "mul";
static const string kPatternAdd = "add";
static const string kOpTypeSigmoid = "Sigmoid";
static const string kOpTypeMul = "Mul";
static const string kOpTypeAdd = "Add";

vector<BufferFusionPattern*> ConvDequantSigmoidMulAddFusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string pattern_name1 = "TbeConvSigmoidMulAddFusionPass";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pattern_name1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern_name1.c_str());
  // define pattern rules
  pattern1
      ->AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternSigmoid, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternMul, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternSigmoid, kPatternMul}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(kPatternSigmoid, {kPatternMul})
      .SetOutputs(kPatternOtherInput, {kPatternAdd})
      .SetOutputs(kPatternMul, {kPatternAdd});
  patterns.push_back(pattern1);

  string pattern_name2 = "ConvDequantSigmoidMulAddFusionPass";
  BufferFusionPattern* pattern2 = new (std::nothrow) BufferFusionPattern(pattern_name2, TBE_FUSION_OP_NUM_MAX + 1);
  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  // define pattern rules
  pattern2
      ->AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternDeq, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternSigmoid, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternMul, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDeq})
      .SetOutputs(kPatternOtherInput, {kPatternDeq})
      .SetOutputs(kPatternDeq, {kPatternSigmoid, kPatternMul}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(kPatternSigmoid, {kPatternMul})
      .SetOutputs(kPatternMul, {kPatternAdd})
      .SetOutputs(kPatternOtherInput2, {kPatternAdd});
  patterns.push_back(pattern2);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern_name2.c_str());
  return patterns;
}

void SetDeqInfo(std::vector<AxisSplitMap>& split_maps, const ge::NodePtr conv_node,
                bool tensor_mode) {
  size_t split_maps_size = split_maps.size();
  for (size_t i = 0; i < split_maps_size; ++i) {
    auto exist_out = split_maps[i].GetOutputSplitInfoVec();
    std::vector<int64_t> c_out = {1};
    bool valid = !exist_out.empty() && exist_out[0].GetAxis() == c_out;
    auto exist_in = split_maps[i].GetInputSplitInfoVec();
    InputSplitInfo input_info;
    if (valid) {
      if (tensor_mode && !exist_in.empty() && input_info.Initialize()) {
        // process dequant deq_scale if exists
        input_info.SetIndex(conv_node->GetInDataNodes().size());
        // deq_scale is 5hd format
        std::vector<int64_t> axis_c = {1};
        input_info.SetAxis(axis_c);
        // the index 0 info is the base op info
        auto head_overlap = exist_in[0].GetHeadOverLap();
        auto tail_overlap = exist_in[0].GetTailOverLap();
        input_info.SetHeadOverLap(head_overlap);
        input_info.SetTailOverLap(tail_overlap);
        split_maps[i].AddInputSplitInfo(input_info);
      }
      break;
    }
  }
}
/*
 * Set split info for patterns
 */
void ConvDequantSigmoidMulAddFusionPass::SetSplitInfo(const BufferFusionMapping& mapping,
                                                      std::vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Begin to SetSplitInfo.");
  vector<ge::NodePtr> conv_nodes = GetMatchedNodesByDescName(kPatternConv, mapping);
  if (conv_nodes.empty()) {
    OP_LOGD(fused_op_type_.c_str(), "conv node not matched");
    return;
  }
  vector<AxisSplitMap> split_maps;
  OpL1FusionType L1_fusion_type = L1FUSION_DISABLE;
  int64_t min_tbe_L1Space = 0;
  if (!GetSplitMap(split_maps, conv_nodes[0], fused_op_type_, L1_fusion_type, min_tbe_L1Space)) {
    return;
  }
  // >>> start: get deq_scale mode
  bool tensor_mode = false;
  vector<ge::NodePtr> dequant_nodes = GetMatchedNodesByDescName(kPatternDeq, mapping);
  if (!dequant_nodes.empty()) {
    auto deq_scale = dequant_nodes[0]->GetOpDesc()->MutableInputDesc("deq_scale");
    vector<int64_t> scalar = {1};
    tensor_mode = deq_scale != nullptr && deq_scale->GetOriginShape().GetDims() != scalar;
  }
  // <<< end: get deq_scale mode

  // >>> start: process quant and deq_scale
  SetDeqInfo(split_maps, conv_nodes[0], tensor_mode);
  // <<< end: process quant and deq_scale

  SetSplitMap(split_maps, fusion_nodes, fused_op_type_, L1_fusion_type, min_tbe_L1Space);
  OP_LOGD(fused_op_type_.c_str(), "End to SetSplitInfo.");
}

Status ConvDequantSigmoidMulAddFusionPass::GetFusionNodes(const BufferFusionMapping& mapping,
                                                          vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Begin to do ConvDequantSigmoidMulAddFusionPass.");

  fusion_nodes = GetMatchedNodes(mapping);
  auto PatternSigmoid_nodes = GetMatchedNodesByDescName(kPatternSigmoid, mapping);
  auto PatternMul_nodes = GetMatchedNodesByDescName(kPatternMul, mapping);
  auto PatternAdd_nodes = GetMatchedNodesByDescName(kPatternAdd, mapping);
  if (!PatternSigmoid_nodes.empty() && PatternSigmoid_nodes[0]->GetType() != kOpTypeSigmoid) {
    fusion_nodes.clear();
    OP_LOGD(fused_op_type_.c_str(),
            "The optype of node[%s] should be Sigmoid, but actually is [%s], no need to do fusion.",
            PatternSigmoid_nodes[0]->GetName().c_str(), PatternSigmoid_nodes[0]->GetType().c_str());
     return SUCCESS;
  }
  if (!PatternMul_nodes.empty() && PatternMul_nodes[0]->GetType() != kOpTypeMul) {
    fusion_nodes.clear();
    OP_LOGD(fused_op_type_.c_str(),
            "The optype of node[%s] should be Mul, but actually is [%s], no need to do fusion.",
            PatternMul_nodes[0]->GetName().c_str(), PatternMul_nodes[0]->GetType().c_str());
    return SUCCESS;
  }
  if (!PatternAdd_nodes.empty() && PatternAdd_nodes[0]->GetType() != kOpTypeAdd) {
    fusion_nodes.clear();
    OP_LOGD(fused_op_type_.c_str(),
            "The optype of node[%s] should be Add, but actually is [%s], no need to do fusion.",
            PatternAdd_nodes[0]->GetName().c_str(), PatternAdd_nodes[0]->GetType().c_str());
    return SUCCESS;
  }
  SetSplitInfo(mapping, fusion_nodes);
  OP_LOGD(fused_op_type_.c_str(), "End to do ConvDequantSigmoidMulAddFusionPass.");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeConvDequantSigmoidMulAddFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            ConvDequantSigmoidMulAddFusionPass);
}  // namespace fe