/**
 * @file tbe_fullyconnection_elemwise_fusion_pass.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief tbe multi-output fusion pattern
 *
 * @version 1.0
 *
 */

#include "tbe_fullyconnection_elemwise_fusion_pass.h"
#include <string>
#include <vector>
#include <cmath>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"
#include "lx_fusion_func.h"
#include "anchor_util.h"

namespace fe {
static const string PATTERN_FC_MATMUL = "FullyConnection/MatMul/BatchMatmul";     // desc name
static const string PATTERN_DEQUANT = "dequant";
static const string PATTERN_QUANT = "quant";
static const string PATTERN_ELTWISE1 = "eltwise1";      // desc name
static const string PATTERN_ELTWISE2 = "eltwise2";      // desc name
static const string PATTERN_OTHER_INPUT = "InputData";  // desc name
static const string PATTERN_OUTPUT = "output";          // desc name
static const int kNumTwo = 2;
static const vector<string> elemWiseWhiteList = {
    "Elu",        "LeakyRelu",    "Gelu",    "Softsign", "Relu6", "Relu",  "Softplus",
    "Sigmoid",    "Tanh",         "Selu",    "GeluGrad", "Add",   "AddN",  "FastGelu",
    "FastGeluV2", "FastGeluGrad", "Eltwise", "PRelu",    "Mul",   "Power", "Relu6D"};
static const vector<string> matmulWhiteList = {"FullyConnection", "MatMul", "MatMulV2", "BatchMatMul", "BatchMatMulV2"};
/*
 * @brief:  define fully connection elemwise fusion pattern
 *
 * pattern configuration limit:
 *
 * FullyConnection/MatMUL/BatchMatmul + (AscendDequant) +（ReLU/LeakyReLU) + (eltwise)
 *
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeFullyconnectionElemwiseFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;

  string passName0 = "TbeBatchMatMulElemwiseDoubleOut";
  BufferFusionPattern *pattern0 = new (std::nothrow) BufferFusionPattern(passName0, TBE_FUSION_OP_NUM_MAX);
  FUSION_PASS_CHECK(pattern0 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName0.c_str());
  // define pattern rules
  pattern0->AddOpDesc(PATTERN_FC_MATMUL, {OP_PATTERN_BATCH_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
           .AddOpDesc(PATTERN_ELTWISE1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
           .AddOpDesc(PATTERN_OUTPUT, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
           .SetHead({PATTERN_FC_MATMUL})
           .SetOutputs(PATTERN_FC_MATMUL, {PATTERN_ELTWISE1, PATTERN_OUTPUT}, TBE_OUTPUT_BRANCH_MULTI);

  patterns.push_back(pattern0);

  string fcAddPassName = "TbeFullyconnectionAddRelu6FusionPass";
  BufferFusionPattern *fcAddPattern = new (std::nothrow) BufferFusionPattern(fcAddPassName, TBE_FUSION_OP_NUM_MAX);
  FUSION_PASS_CHECK(fcAddPattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass fcAddPattern.", fcAddPassName.c_str());
  // define pattern rules
  fcAddPattern
      ->AddOpDesc(PATTERN_FC_MATMUL, {OP_PATTERN_MATMUL, OP_PATTERN_BATCH_MATMUL}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELTWISE1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELTWISE2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_FC_MATMUL})
      .SetOutputs(PATTERN_FC_MATMUL, {PATTERN_ELTWISE1})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_ELTWISE1})
      .SetOutputs(PATTERN_ELTWISE1, {PATTERN_ELTWISE2})
      .SetOutputs(PATTERN_ELTWISE2, {}, TBE_OUTPUT_BRANCH_SINGLE, true);

  patterns.push_back(fcAddPattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass fcAddPattern.", fcAddPassName.c_str());

  string passName1 = "TbeFullyconnectionDequantElemwiseQuantFusionPass";
  BufferFusionPattern *pattern1 = new(std::nothrow) BufferFusionPattern(passName1, TBE_FUSION_OP_NUM_MAX);
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
  return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName1.c_str());
  // define pattern rules
  pattern1->AddOpDesc(PATTERN_FC_MATMUL, {OP_PATTERN_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELTWISE1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_FC_MATMUL})
      .SetOutputs(PATTERN_FC_MATMUL, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_DEQUANT, {PATTERN_ELTWISE1})
      .SetOutputs(PATTERN_ELTWISE1, {PATTERN_QUANT});
  patterns.push_back(pattern1);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName1.c_str());
  
  string passName = "TbeFullyconnectionElemwiseDequantFusionPass";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(passName, TBE_FUSION_OP_NUM_MAX);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  // define pattern rules
  pattern->AddOpDesc(PATTERN_FC_MATMUL, {OP_PATTERN_MATMUL, OP_PATTERN_BATCH_MATMUL},
                     TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_ELTWISE1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_ELTWISE2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .SetHead({PATTERN_FC_MATMUL})
          .SetOutputs(PATTERN_FC_MATMUL, {PATTERN_DEQUANT})
          .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
          .SetOutputs(PATTERN_DEQUANT, {PATTERN_ELTWISE1})
          .SetOutputs(PATTERN_ELTWISE1, {PATTERN_ELTWISE2})
          .SetOutputs(PATTERN_ELTWISE2, {}, TBE_OUTPUT_BRANCH_SINGLE, true);

  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());
  return patterns;
}

void TbeFullyconnectionElemwiseFusionPass::SetSplitInfo(const BufferFusionMapping &mapping,
                                                        std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> fcNodes = GetMatchedNodesByDescName(PATTERN_FC_MATMUL, mapping);
  vector<ge::NodePtr> reluNodes = GetMatchedNodesByDescName(PATTERN_ELTWISE1, mapping);
  vector<ge::NodePtr> elemWiseNodes = GetMatchedNodesByDescName(PATTERN_ELTWISE2, mapping);
  vector<ge::NodePtr> dequantNodes = GetMatchedNodesByDescName(PATTERN_DEQUANT, mapping);

  int n_axis = 0;
  for (const auto& fcNode : fcNodes) {
    if (fcNode->GetType() == "FullyConnection") {
      int axis;
      if (!ge::AttrUtils::GetInt(fcNode->GetOpDesc(), "axis", axis)) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "FullyConnection op[%s] type[%s] node does not have axis attr!",
                fcNode->GetName().c_str(), fcNode->GetType().c_str());
        return;
      }
      auto input0desc = GetCurrNodeInputDesc(fcNode, 0);
      FUSION_PASS_CHECK(input0desc == nullptr,
                  CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc0 is null"),
                  return);
      if (axis == kNumTwo) {
        n_axis = 1;
      } else if (input0desc->GetFormat() == ge::FORMAT_FRACTAL_NZ) {
        n_axis = 0;
      } else {
        n_axis = 1;
      }
    } else {
      n_axis = 0;
    }
  }

  int pre = 0;
  vector<AxisSplitMap> split_maps;
  OpL1FusionType fusion_type = L1FUSION_DISABLE;
  int64_t min_tbe_l1space = 0;
  if (!fcNodes.empty()) {
    pre += fcNodes[0]->GetInDataNodes().size() - 1;
    if (!GetSplitMap(split_maps, fcNodes[0], FUSED_OP_TYPE, fusion_type, min_tbe_l1space)) {
      return;
    }
  }

  bool tensor_mode = false;
  if (!dequantNodes.empty()) {
    pre += 1;
    auto deq_scale = GetCurrNodeMutableInputDesc(dequantNodes[0], "deq_scale");
    vector<int64_t> scalar = {1};
    tensor_mode = deq_scale != nullptr && deq_scale->GetOriginShape().GetDims() != scalar;
  }
  // the dequant is scala mode, can not split c_dim
  if (!tensor_mode) {
    DelSplitInfoByOutputAxis(split_maps, n_axis);
  }

  if (elemWiseNodes.empty()) {
    elemWiseNodes = reluNodes;
  }
  if (!elemWiseNodes.empty()){
    AddElemwiseSplitMap(split_maps, elemWiseNodes[0], pre);
  }
  SetSplitMap(split_maps, fusion_nodes, FUSED_OP_TYPE, fusion_type, min_tbe_l1space);
}

bool CheckPreNodeIsFcNode(const ge::NodePtr& reluNode) {
  for (auto in_anchor : reluNode->GetAllInDataAnchors()) {
    ge::OutDataAnchorPtr pre_out_anchor = in_anchor->GetPeerOutAnchor();
    if (pre_out_anchor == nullptr) {
      continue;
    }
    ge::NodePtr pre_node = pre_out_anchor->GetOwnerNode();
    if (pre_node == nullptr || pre_node->GetType() != "FullyConnection") {
      continue;
    }
    return true;
  }
  return false;
}

bool TbeFullyconnectionElemwiseFusionPass::CheckMatmulDequantGeluQuantFusion(const BufferFusionMapping &mapping){
  // check matmul dequant gelu quant pass
  vector<ge::NodePtr> reluNodes = GetMatchedNodesByDescName(PATTERN_ELTWISE1, mapping);
  vector<ge::NodePtr> elemWiseNodes = GetMatchedNodesByDescName(PATTERN_ELTWISE2, mapping);
  vector<ge::NodePtr> dequantNodes = GetMatchedNodesByDescName(PATTERN_DEQUANT, mapping);
  vector<ge::NodePtr> quantNodes = GetMatchedNodesByDescName(PATTERN_QUANT, mapping);
  if (!dequantNodes.empty() && !quantNodes.empty() && !reluNodes.empty() && elemWiseNodes.empty()) {
    for (const auto &reluNode: reluNodes) {
      bool is_quant_flag = false;
      for (auto out_anchor: reluNode->GetAllOutDataAnchors()) {
        for (auto next_in_anchor: out_anchor->GetPeerInDataAnchors()) {
          ge::NodePtr nextNode = next_in_anchor->GetOwnerNode();
          if (nextNode == nullptr || nextNode->GetType() != "AscendQuant") {
            if (nextNode == nullptr) {
              OP_LOGD(FUSED_OP_TYPE.c_str(),
                      "elemwise node connect to nullptr, skip this node!");
            } else {
              OP_LOGD(FUSED_OP_TYPE.c_str(),
                      "elemwise node connect to type[%s], is not supported for this ub fusion pass, skip this node!",
                      nextNode->GetType().c_str());
            }
            continue;
          }
          is_quant_flag = true;
        }
      }
      if (!is_quant_flag || reluNode->GetType() != "Gelu") {
        return false;
      }
    }
  }
  return true;
}

Status TbeFullyconnectionElemwiseFusionPass::CheckDynamicMode(vector<ge::NodePtr>& matmulNodes,
                                                              vector<ge::NodePtr>& fusionNodes) const {
  for (const auto& matmulNode : matmulNodes){
    auto input0desc = GetCurrNodeInputDesc(matmulNode, 0);
    auto input1desc = GetCurrNodeInputDesc(matmulNode, 1);
    FUSION_PASS_CHECK(input0desc == nullptr,
                  CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc0 is null"),
                  return FAILED);
    FUSION_PASS_CHECK(input1desc == nullptr,
                  CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc1 is null"),
                  return FAILED);
    vector<int64_t> input0Dims = input0desc->GetOriginShape().GetDims();
    vector<int64_t> input1Dims = input1desc->GetOriginShape().GetDims();
    vector<int64_t> allDims;
    allDims.resize(input0Dims.size() + input1Dims.size());
    merge(input0Dims.begin(), input0Dims.end(), input1Dims.begin(), input1Dims.end(), allDims.begin());
    for (auto singleDim : allDims) {
      if (singleDim < 0) {
        fusionNodes.clear();
        OP_LOGW(FUSED_OP_TYPE.c_str(), "ub fusion not support dynamic shape");
        return SUCCESS;
      }
    }
  }
  return NOT_CHANGED;
}

Status FusionReturn(bool &cond, vector<ge::NodePtr> &fusionNodes) {
  if (cond) {
    fusionNodes.clear();
    return SUCCESS;
  }
  return NOT_CHANGED;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeFullyconnectionElemwiseFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                            vector<ge::NodePtr> &fusionNodes) {
  OP_LOGD("Begin to do TbeFullyconnectionElemwiseFusionPass!");
  fusionNodes = GetMatchedNodes(mapping);

  // buffer fusion do not support dynamic shape now
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_FC_MATMUL, mapping);
  Status ret = CheckDynamicMode(matmulNodes, fusionNodes);
  FUSION_PASS_CHECK(ret != NOT_CHANGED,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "check dynamic mode failed"),
                    return ret);

  vector<ge::NodePtr> fcNodes = GetMatchedNodesByDescName(PATTERN_FC_MATMUL, mapping);
  vector<ge::NodePtr> reluNodes = GetMatchedNodesByDescName(PATTERN_ELTWISE1, mapping);
  vector<ge::NodePtr> elemWiseNodes = GetMatchedNodesByDescName(PATTERN_ELTWISE2, mapping);
  // check whether the fc/matmul/batchmatmul op
  for (const auto &fcNode : fcNodes) {
    if (find(matmulWhiteList.begin(), matmulWhiteList.end(), fcNode->GetType()) == matmulWhiteList.end()) {
      fusionNodes.clear();
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "fcNode op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
              fcNode->GetName().c_str(), fcNode->GetType().c_str());
      return SUCCESS;
    }
  }

  // check whether the matmul/dequant/gelu/quant/fusion
  bool isMatmulDequantGeluQuant = !CheckMatmulDequantGeluQuantFusion(mapping);
  FUSION_PASS_CHECK(
      FusionReturn(isMatmulDequantGeluQuant, fusionNodes) == SUCCESS,
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "Eltwise op type is not supported for matmul dequant gelu quant ub fusion pass, skip fusion."),
      return SUCCESS);

  // check whether the relu/leakyrelu op
  for (const auto &reluNode : reluNodes) {
    if (elemWiseNodes.empty()) {
      bool unvalid_elemWise1_type = reluNode->GetType() != "Relu" && reluNode->GetType() != "LeakyRelu" &&
                                    find(elemWiseWhiteList.begin(),
                                         elemWiseWhiteList.end(), reluNode->GetType()) == elemWiseWhiteList.end();
      if (unvalid_elemWise1_type) {
        fusionNodes.clear();
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
                reluNode->GetName().c_str(), reluNode->GetType().c_str());
        return SUCCESS;
      }
    } else {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise1 type is %s, Eltwise2 is not empty.", reluNode->GetType().c_str());
      bool unvalid_elemWise1_type =
          reluNode->GetType() != "Relu" && reluNode->GetType() != "LeakyRelu" && reluNode->GetType() != "Add";
      if (unvalid_elemWise1_type) {
        fusionNodes.clear();
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
                reluNode->GetName().c_str(), reluNode->GetType().c_str());
        return SUCCESS;
      }
      if (reluNode->GetType() == "Add") {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise1 type is add, start this ub fusion.");
        // the input nodes of add must be 2
        FUSION_PASS_CHECK(reluNode->GetAllInDataAnchors().size() != 2,
                          CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input nodes must be two"), return FAILED);
        // previous nodes are all not FC
        bool pre_node_is_fc = !CheckPreNodeIsFcNode(reluNode);
        FUSION_PASS_CHECK(FusionReturn(pre_node_is_fc, fusionNodes) == SUCCESS,
                          OP_LOGD(FUSED_OP_TYPE.c_str(), "both the inputs are not FullyConnection"),
                          return SUCCESS);
        ge::OutDataAnchorPtr out_anchor = reluNode->GetOutDataAnchor(0);
        FUSION_PASS_CHECK(out_anchor->GetPeerInDataAnchors().size() != 1,
                          CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output node must be one"), return FAILED);
        // the elemwise2 is not relu6
        bool is_relu = out_anchor->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetType() != "Relu6";
        FUSION_PASS_CHECK(FusionReturn(is_relu, fusionNodes) == SUCCESS,
                          OP_LOGD(FUSED_OP_TYPE.c_str(),
                                  "Eltwise2 type not relu6, graph not support this ub fusion pass, skip fusion."),
                          return SUCCESS);
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise2 type is relu6, this ub fusion success.");
      }
      float negative_slope = 0;
      if (reluNode->GetType() == "LeakyRelu") {
        FUSION_PASS_CHECK(!ge::AttrUtils::GetFloat(reluNode->GetOpDesc(), "negative_slope", negative_slope),
                          CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                "LeakyRelu op[%s] type[%s] node does not have negative slope attr!",
                                                reluNode->GetName().c_str(), reluNode->GetType().c_str()),
                          return FAILED);
        if (std::fabs(negative_slope) > std::numeric_limits<float>::epsilon()) {
          fusionNodes.clear();
          OP_LOGD(FUSED_OP_TYPE.c_str(), "LeakyRelu op[%s] type[%s] node has negative slope.",
                  reluNode->GetName().c_str(), reluNode->GetType().c_str());
          return SUCCESS;
        }
      }
    }
  }

  // check whether the EltWise op is in the whitelist
  for (const auto &elemWiseNode : elemWiseNodes) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise2 type is %s.", elemWiseNode->GetType().c_str());
    if (find(elemWiseWhiteList.begin(), elemWiseWhiteList.end(), elemWiseNode->GetType()) == elemWiseWhiteList.end()) {
      fusionNodes.clear();
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
              elemWiseNode->GetName().c_str(), elemWiseNode->GetType().c_str());
      return SUCCESS;
    }
  }

  // the outputData can't be fused
  for (auto& item : mapping) {
    auto opdesc = find(item.first->types.begin(), item.first->types.end(), TBE_PATTERN_OUTPUT_NODE);
    if (opdesc == item.first->types.end()) {
      continue;
    }
    for (auto& node : item.second) {
      auto node_ptr = find(fusionNodes.begin(), fusionNodes.end(), node);
      if (node_ptr != fusionNodes.end()) {
        fusionNodes.erase(node_ptr);
      }
    }
  }

  SetSplitInfo(mapping, fusionNodes);
  OP_LOGD("End to do TbeFullyconnectionElemwiseFusionPass!");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeFullyconnectionElemwiseDequantFusionPass",
                            BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeFullyconnectionElemwiseFusionPass);
}  // namespace fe
