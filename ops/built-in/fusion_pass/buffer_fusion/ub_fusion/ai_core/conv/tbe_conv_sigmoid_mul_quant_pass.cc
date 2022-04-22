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

#include "tbe_conv_sigmoid_mul_quant_pass.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"
#include "lx_fusion_func.h"

namespace fe {
using std::vector;
static const string PATTERN_CONV = "convolution";
static const string PATTERN_DEQUANT = "dequant";
static const string PATTERN_FIXPIPE = "FixPipe";
static const string PATTERN_QUANT = "quant";
static const string PATTERN_OTHER_INPUT = "otherInput";
static const string PATTERN_SIGMOID = "sigmoid";
static const string PATTERN_MUL = "mul";
static const string OP_TYPE_SIGMOID = "Sigmoid";
static const string OP_TYPE_MUL = "Mul";

vector<BufferFusionPattern*> ConvSigmoidMulQuantFusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;

  string pattern_name = "TbeConvSigmoidMulFusionPass";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern_name.c_str());
  // define pattern rules
  pattern->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_SIGMOID, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_MUL, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .SetHead({PATTERN_CONV})
          .SetOutputs(PATTERN_CONV, {PATTERN_SIGMOID, PATTERN_MUL}, TBE_OUTPUT_BRANCH_MULTI)
          .SetOutputs(PATTERN_SIGMOID, {PATTERN_MUL})
          .SetOutputs(PATTERN_MUL, {PATTERN_QUANT});
  patterns.push_back(pattern);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern_name.c_str());

  string pattern_name1 = "TbeConvDequantSigmoidMulQuantFusionPass";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pattern_name1, TBE_FUSION_OP_NUM_MAX + 1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern_name1.c_str());
  // define pattern rules
  pattern1->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
   TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                     TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                     TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_SIGMOID, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                     TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_MUL, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                     TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT,
                     TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .SetHead({PATTERN_CONV})
          .SetOutputs(PATTERN_CONV, {PATTERN_DEQUANT})
          .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
          .SetOutputs(PATTERN_DEQUANT, {PATTERN_SIGMOID, PATTERN_MUL}, TBE_OUTPUT_BRANCH_MULTI)
          .SetOutputs(PATTERN_SIGMOID, {PATTERN_MUL})
          .SetOutputs(PATTERN_MUL, {PATTERN_QUANT});
  patterns.push_back(pattern1);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern_name1.c_str());

  string pattern_name2 = "TbeConvFixpipeSigmoidMulQuantFusionPass";
  BufferFusionPattern* pattern2 = new (std::nothrow) BufferFusionPattern(pattern_name2, TBE_FUSION_OP_NUM_MAX + 1);
  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern_name2.c_str());
  // define pattern rules
  pattern2->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_FIXPIPE, {OP_PATTERN_FIXPIPE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_SIGMOID, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_MUL, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .SetHead({PATTERN_CONV})
          .SetOutputs(PATTERN_CONV, {PATTERN_FIXPIPE})
          .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_FIXPIPE})
          .SetOutputs(PATTERN_FIXPIPE, {PATTERN_SIGMOID, PATTERN_MUL}, TBE_OUTPUT_BRANCH_MULTI)
          .SetOutputs(PATTERN_SIGMOID, {PATTERN_MUL})
          .SetOutputs(PATTERN_MUL, {PATTERN_QUANT});
  patterns.push_back(pattern2);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern_name2.c_str());

  return patterns;
}

/*
 * Set split info for patterns
 */
void ConvSigmoidMulQuantFusionPass::SetSplitInfo(const BufferFusionMapping &mapping,
                                                 std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> conv_nodes = GetMatchedNodesByDescName(PATTERN_CONV, mapping);
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
  vector<ge::NodePtr> dequant_nodes = GetMatchedNodesByDescName(PATTERN_DEQUANT, mapping);
  if (!dequant_nodes.empty()) {
    auto deq_scale = dequant_nodes[0]->GetOpDesc()->MutableInputDesc("deq_scale");
    vector<int64_t> scalar = {1};
    tensor_mode = deq_scale != nullptr && deq_scale->GetOriginShape().GetDims() != scalar;
  }
  // <<< end: get deq_scale mode

  // >>> start: process quant and deq_scale
  vector<ge::NodePtr> quant_nodes = GetMatchedNodesByDescName(PATTERN_QUANT, mapping);
  bool has_quant = quant_nodes.size() > 0;
  for (size_t i = 0; i < split_maps.size(); ++i) {
    auto exist_out = split_maps[i].GetOutputSplitInfoVec();
    std::vector<int64_t> c_out = {1};
    bool valid = !exist_out.empty() && exist_out[0].GetAxis() == c_out;
    if (valid) {
      if (has_quant) {
        split_maps.erase(split_maps.begin() + i);
      } else if (tensor_mode) {
        // process dequant deq_scale if exists
        auto exist_in = split_maps[i].GetInputSplitInfoVec();
        if (!exist_in.empty()) {
          InputSplitInfo input_info;
          if (!input_info.Initialize()) {
            OP_LOGD(fused_op_type_.c_str(), "init input_info failed");
          } else {
            input_info.SetIndex(conv_nodes[0]->GetInDataNodes().size());
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
        }
      }
      break;
    }
  }
  // <<< end: process quant and deq_scale

  SetSplitMap(split_maps, fusion_nodes, fused_op_type_, L1_fusion_type, min_tbe_L1Space);
}

Status ConvSigmoidMulQuantFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                     vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Begin to do ConvSigmoidMulQuantFusionPass.");
  fusion_nodes = GetMatchedNodes(mapping);
  for (auto &item : mapping) {
    const BufferFusionOpDesc *op_desc = item.first;
    if (op_desc != nullptr && op_desc->desc_name == PATTERN_SIGMOID) {
      ge::NodePtr node = item.second[0];
      if (node == nullptr) {
        return FAILED;
      }
      if (node->GetType() != OP_TYPE_SIGMOID) {
        fusion_nodes.clear();
        OP_LOGD(fused_op_type_.c_str(), "Eltwise is op [%s, %s], skip fusion.",
                node->GetName().c_str(), node->GetType().c_str());
        break;
      }
    }
    if (op_desc != nullptr && op_desc->desc_name == PATTERN_MUL) {
      ge::NodePtr node = item.second[0];
      if (node == nullptr) {
        return FAILED;
      }
      if (node->GetType() != OP_TYPE_MUL) {
        fusion_nodes.clear();
        OP_LOGD(fused_op_type_.c_str(), "Eltwise is op [%s, %s], skip fusion.",
                node->GetName().c_str(), node->GetType().c_str());
        break;
      }
    }
  }
  SetSplitInfo(mapping, fusion_nodes);
  OP_LOGD(fused_op_type_.c_str(), "End to do ConvSigmoidMulQuantFusionPass.");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeConvSigmoidMulQuantFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            ConvSigmoidMulQuantFusionPass);
}  // namespace fe