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
#include "common/util/platform_info.h"
#include "fusion_pre_trans_func.h"

namespace fe {
static const string PATTERN_FC_MATMUL = "FullyConnection/MatMul/BatchMatmul";     // desc name
static const string PATTERN_DEQUANT = "dequant";
static const string PATTERN_QUANT = "quant";
static const string PATTERN_ELTWISE1 = "eltwise1";      // desc name
static const string PATTERN_ELTWISE2 = "eltwise2";      // desc name
static const string PATTERN_OTHER_INPUT = "InputData";  // desc name
static const string PATTERN_OTHER_INPUT2 = "InputData2";  // desc name
static const string PATTERN_OUTPUT = "output";          // desc name
static const int kNumTwo = 2;
static const vector<string> elemWiseWhiteList = {
    "Elu",        "LeakyRelu",    "Gelu",    "Softsign", "Relu6", "Relu",  "Softplus",
    "Sigmoid",    "Tanh",         "Selu",    "GeluGrad", "Add",   "AddN",  "FastGelu",
    "FastGeluV2", "FastGeluGrad", "Eltwise", "PRelu",    "Mul",   "Muls",   "Power",  "Relu6D", "TanhGrad"};
static const vector<string> matmulWhiteList = {"FullyConnection", "MatMul", "MatMulV2", "BatchMatMul", "BatchMatMulV2"};
static const vector<int64_t> scalar = {1};
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
  pattern0->AddOpDesc(PATTERN_FC_MATMUL, {OP_PATTERN_BATCH_MATMUL, OP_PATTERN_GEMM},
                      TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
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
      ->AddOpDesc(PATTERN_FC_MATMUL, {OP_PATTERN_MATMUL, OP_PATTERN_GEMM, OP_PATTERN_BATCH_MATMUL},
                  TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
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

  string passName2 = "TbeFullyconnectionElemwiseDoubleOutElemwiseFusionPass";
  BufferFusionPattern *pattern2 = new (std::nothrow) BufferFusionPattern(passName2);
  FUSION_PASS_CHECK(pattern2 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern2.", passName2.c_str());
  // define pattern rules
  pattern2->AddOpDesc(PATTERN_FC_MATMUL, {OP_PATTERN_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_ELTWISE1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_ELTWISE2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_OTHER_INPUT2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_OUTPUT, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .SetHead({PATTERN_FC_MATMUL})
          .SetOutputs(PATTERN_FC_MATMUL, {PATTERN_ELTWISE1})
          .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_ELTWISE1})
          .SetOutputs(PATTERN_ELTWISE1, {PATTERN_ELTWISE2, PATTERN_OUTPUT}, TBE_OUTPUT_BRANCH_MULTI, false, true)
          .SetOutputs(PATTERN_OTHER_INPUT2, {PATTERN_ELTWISE2})
          .SetOutputs(PATTERN_ELTWISE2, {}, TBE_OUTPUT_BRANCH_SINGLE, true);

  patterns.push_back(pattern2);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern2.", passName2.c_str());

  string passName = "TbeFullyconnectionElemwiseDequantFusionPass";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(passName, TBE_FUSION_OP_NUM_MAX);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  // define pattern rules
  pattern->AddOpDesc(PATTERN_FC_MATMUL, {OP_PATTERN_MATMUL, OP_PATTERN_GEMM, OP_PATTERN_BATCH_MATMUL},
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

void TbeFullyconnectionElemwiseFusionPass::SetSplitInfoOfMultiOut(const std::vector<ge::NodePtr> &nodes_fc,
                                                                  vector<AxisSplitMap> *split_maps) {
  if (!nodes_fc.empty() && nodes_fc[0]->GetOutDataNodes().size() > 1) {
    auto size = nodes_fc[0]->GetOutDataNodes().size();
    for (auto &split_map : *split_maps) {
      for (size_t i = 1; i < size; ++i) {
        auto out_split_info = CopyOutputSplitInfo(*(split_map.GetOutputSplitInfos()[0]));
        out_split_info.SetIndex(i);

        split_map.AddOutputSplitInfo(out_split_info);
      }
    }
  }
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
    tensor_mode = deq_scale != nullptr && deq_scale->GetOriginShape().GetDims() != scalar;
    if (!tensor_mode) {
      // the dequant is scala mode, can not split c_dim
      DelSplitInfoByOutputAxis(split_maps, n_axis);
    }
  }

  if (elemWiseNodes.empty()) {
    elemWiseNodes = reluNodes;
  }
  if (!elemWiseNodes.empty()) {
    AddElemwiseSplitMap(split_maps, elemWiseNodes[0], pre);
  }

  SetSplitInfoOfMultiOut(fcNodes, &split_maps);

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

bool TbeFullyconnectionElemwiseFusionPass::CheckMatmulDequantGeluQuantFusion(const vector<ge::NodePtr> &reluNodes) {
  // check matmul dequant gelu quant pass
  for (const auto &reluNode : reluNodes) {
    if (reluNode->GetType() != "Gelu") {
      return true;
    }
  }
  PlatformInfo platformInfo;
  OptionalInfo optionalInfo;
  FUSION_PASS_CHECK(
      PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Get platform_info failed."), return true);
  const auto &instrinsicScatterVcmp = platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_abs"];
  bool supportFP32Flag =
      find(instrinsicScatterVcmp.begin(), instrinsicScatterVcmp.end(), "float32") != instrinsicScatterVcmp.end();
  if (!supportFP32Flag) {
    return true;
  }
  return false;
}

Status TbeFullyconnectionElemwiseFusionPass::CheckDynamicMode(vector<ge::NodePtr>& matmulNodes,
                                                              vector<ge::NodePtr>& fusionNodes) const {
  for (const auto& matmulNode : matmulNodes) {
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

Status TbeFullyconnectionElemwiseFusionPass::CheckReluNode(const ge::NodePtr &relu_node,
                                                           const vector<ge::NodePtr> &elemwise_nodes,
                                                           vector<ge::NodePtr> &fusion_nodes) {
  if (elemwise_nodes.empty()) {
    bool unvalid_elemWise1_type = relu_node->GetType() != "Relu" && relu_node->GetType() != "LeakyRelu" &&
                                  find(elemWiseWhiteList.begin(),
                                       elemWiseWhiteList.end(), relu_node->GetType()) == elemWiseWhiteList.end();
    if (unvalid_elemWise1_type) {
      fusion_nodes.clear();
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
              relu_node->GetName().c_str(), relu_node->GetType().c_str());
      return SUCCESS;
    }
  } else {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise1 type is %s, Eltwise2 is not empty.", relu_node->GetType().c_str());
    bool unvalid_elemWise1_type =
      relu_node->GetType() != "Relu" && relu_node->GetType() != "LeakyRelu" && relu_node->GetType() != "Add" &&
      relu_node->GetType() != "Muls" && relu_node->GetType() != "AddN";
    if (unvalid_elemWise1_type) {
      fusion_nodes.clear();
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
              relu_node->GetName().c_str(), relu_node->GetType().c_str());
      return SUCCESS;
    }
    if (relu_node->GetType() == "Add") {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise1 type is add, start this ub fusion.");
      // the input nodes of add must be 2
      FUSION_PASS_CHECK(relu_node->GetAllInDataAnchors().size() != 2,
                        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input nodes must be two"), return FAILED);
      // previous nodes are all not FC
      bool pre_node_is_fc = !CheckPreNodeIsFcNode(relu_node);
      FUSION_PASS_CHECK(FusionReturn(pre_node_is_fc, fusion_nodes) == SUCCESS,
                        OP_LOGD(FUSED_OP_TYPE.c_str(), "both the inputs are not FullyConnection"),
                        return SUCCESS);
      ge::OutDataAnchorPtr out_anchor = relu_node->GetOutDataAnchor(0);
      FUSION_PASS_CHECK(out_anchor->GetPeerInDataAnchors().size() != 1,
                        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output node must be one"), return FAILED);
      // the elemwise2 is not relu6
      bool is_relu = out_anchor->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetType() != "Relu6";
      FUSION_PASS_CHECK(FusionReturn(is_relu, fusion_nodes) == SUCCESS,
                        OP_LOGD(FUSED_OP_TYPE.c_str(),
                                "Eltwise2 type not relu6, graph not support this ub fusion pass, skip fusion."),
                        return SUCCESS);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise2 type is relu6, this ub fusion success.");
    }
    float negative_slope = 0;
    if (relu_node->GetType() == "LeakyRelu") {
      FUSION_PASS_CHECK(!ge::AttrUtils::GetFloat(relu_node->GetOpDesc(), "negative_slope", negative_slope),
                        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                              "LeakyRelu op[%s] type[%s] node does not have negative slope attr!",
                                              relu_node->GetName().c_str(), relu_node->GetType().c_str()),
                        return FAILED);
      if (std::fabs(negative_slope) > std::numeric_limits<float>::epsilon()) {
        fusion_nodes.clear();
        OP_LOGD(FUSED_OP_TYPE.c_str(), "LeakyRelu op[%s] type[%s] node has negative slope.",
                relu_node->GetName().c_str(), relu_node->GetType().c_str());
        return SUCCESS;
      }
    }
  }
  return NOT_CHANGED;
}

Status TbeFullyconnectionElemwiseFusionPass::CheckElemwiseNode(const ge::NodePtr &elemwise_node,
                                                               const vector<ge::NodePtr> &fc_nodes,
                                                               const vector<ge::NodePtr> &relu_nodes,
                                                               const vector<ge::NodePtr> &dequant_nodes,
                                                               vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise2 type is %s.", elemwise_node->GetType().c_str());
  bool invalid_elemwise_type = (relu_nodes.empty() && elemwise_node->GetType() == "TanhGrad") ||
                                find(elemWiseWhiteList.begin(),
                                    elemWiseWhiteList.end(), elemwise_node->GetType()) == elemWiseWhiteList.end();
  if (invalid_elemwise_type) {
    fusion_nodes.clear();
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
            elemwise_node->GetName().c_str(), elemwise_node->GetType().c_str());
    return SUCCESS;
  }
  invalid_elemwise_type = !relu_nodes.empty() &&
                          relu_nodes[0]->GetType() == "AddN" && elemwise_node->GetType() == "Mul";
  FUSION_PASS_CHECK(invalid_elemwise_type,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "MatMul + AddN + Mul is not supported for performance."),
                    fusion_nodes.clear();
                    return SUCCESS);
  // fusion will break bmm + add/relu + elewise in TbeBatchMatmulElementWiseFusionPass
  bool is_batchmatmul = fc_nodes[0]->GetType() == "BatchMatMul" || fc_nodes[0]->GetType() == "BatchMatMulV2";
  bool is_wrong_type = elemwise_node->GetType() == "Add" || elemwise_node->GetType() == "Relu";
  bool clear_fusion = is_batchmatmul && relu_nodes.empty() && dequant_nodes.empty() && is_wrong_type;
  FUSION_PASS_CHECK(clear_fusion,
                    OP_LOGD(FUSED_OP_TYPE.c_str(),
                            "BatchMatmul + type[%s] is not supported for this ub fusion pass, skip fusion.",
                            elemwise_node->GetType().c_str()),
                    fusion_nodes.clear();
                    return SUCCESS);
  return NOT_CHANGED;
}

void TbeFullyconnectionElemwiseFusionPass::CheckOutputFusion(const BufferFusionMapping &mapping,
                                                             vector<ge::NodePtr> &fusionNodes) {
  // the outputData can't be fused
  for (auto &item : mapping) {
    auto opdesc = find(item.first->types.begin(), item.first->types.end(), TBE_PATTERN_OUTPUT_NODE);
    if (opdesc == item.first->types.end()) {
      continue;
    }
    for (auto &node : item.second) {
      auto node_ptr = find(fusionNodes.begin(), fusionNodes.end(), node);
      if (node_ptr != fusionNodes.end()) {
        fusionNodes.erase(node_ptr);
      }
    }
  }
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
  if (fusionNodes.size() == 1) {
    fusionNodes.clear();
    return SUCCESS;
  }
  // buffer fusion do not support dynamic shape now
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_FC_MATMUL, mapping);
  FusePreTransdata(matmulNodes, fusionNodes);
  Status ret = CheckDynamicMode(matmulNodes, fusionNodes);
  FUSION_PASS_CHECK(ret != NOT_CHANGED,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "check dynamic mode failed"),
                    return ret);

  vector<ge::NodePtr> fcNodes = GetMatchedNodesByDescName(PATTERN_FC_MATMUL, mapping);
  vector<ge::NodePtr> reluNodes = GetMatchedNodesByDescName(PATTERN_ELTWISE1, mapping);
  vector<ge::NodePtr> elemWiseNodes = GetMatchedNodesByDescName(PATTERN_ELTWISE2, mapping);
  vector<ge::NodePtr> dequantNodes = GetMatchedNodesByDescName(PATTERN_DEQUANT, mapping);
  vector<ge::NodePtr> quantNodes = GetMatchedNodesByDescName(PATTERN_QUANT, mapping);

  // check whether the fc/matmul/batchmatmul op
  for (const auto &fcNode : fcNodes) {
    if (find(matmulWhiteList.begin(), matmulWhiteList.end(), fcNode->GetType()) == matmulWhiteList.end()) {
      fusionNodes.clear();
      OP_LOGD(FUSED_OP_TYPE.c_str(), "fcNode op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
              fcNode->GetName().c_str(), fcNode->GetType().c_str());
      return SUCCESS;
    }
  }

  // check whether the matmul/dequant/gelu/quant/fusion
  bool isdequantElemwiseQuant =
      !dequantNodes.empty() && !quantNodes.empty() && !reluNodes.empty() && elemWiseNodes.empty();
  if (isdequantElemwiseQuant) {
    bool elemwiseisGelu = CheckMatmulDequantGeluQuantFusion(reluNodes);
    FUSION_PASS_CHECK(
        FusionReturn(elemwiseisGelu, fusionNodes) == SUCCESS,
        OP_LOGD(FUSED_OP_TYPE.c_str(),
                "Eltwise op type is not supported for matmul dequant gelu quant ub fusion pass, skip fusion."),
        return SUCCESS);
  }

  // check whether the relu/leakyrelu op
  for (const auto &reluNode : reluNodes) {
    Status relu_ret = CheckReluNode(reluNode, elemWiseNodes, fusionNodes);
    FUSION_PASS_CHECK(relu_ret != NOT_CHANGED,
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "Check Eltwise1 node failed."),
                      return relu_ret);
  }

  // check whether the EltWise op is in the whitelist
  for (const auto &elemWiseNode : elemWiseNodes) {
    Status elemwise_ret = CheckElemwiseNode(elemWiseNode, fcNodes, reluNodes, dequantNodes, fusionNodes);
    FUSION_PASS_CHECK(elemwise_ret != NOT_CHANGED,
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "Check Eltwise2 node failed."),
                      return elemwise_ret);
  }

  CheckOutputFusion(mapping, fusionNodes);
  SetSplitInfo(mapping, fusionNodes);
  OP_LOGD("End to do TbeFullyconnectionElemwiseFusionPass!");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeFullyconnectionElemwiseDequantFusionPass",
                            BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeFullyconnectionElemwiseFusionPass);
}  // namespace fe
