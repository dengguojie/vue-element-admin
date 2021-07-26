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
static const vector<string> elemWiseWhiteList = {
  "Elu", "LeakyRelu", "Gelu", "Softsign", "Relu6", "Relu", "Softplus", "Sigmoid", "Tanh", "Selu",
  "GeluGrad", "Add", "AddN", "FastGelu", "FastGeluGrad", "Eltwise", "PRelu", "Mul", "Power", "Relu6D"};
static const vector<string> matmulWhiteList = {
  "FullyConnection", "MatMul", "MatMulV2", "BatchMatMul", "BatchMatMulV2"
};
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
  string passName = "TbeFullyconnectionElemwiseDequantFusionPass";

  BufferFusionPattern *pattern =
          new (std::nothrow) BufferFusionPattern(passName, TBE_FUSION_OP_NUM_MAX);
  if (pattern == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed.");
    return patterns;
  }

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

void TbeFullyconnectionElemwiseFusionPass::SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes) {
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
      if (axis == 2) {
        n_axis = 1;
      } else if(input0desc->GetFormat() == ge::FORMAT_FRACTAL_NZ) {
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
  if (!fcNodes.empty()) {
    pre += fcNodes[0]->GetInDataNodes().size() - 1;
    if (!GetSplitMap(split_maps, fcNodes[0], FUSED_OP_TYPE)) {
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
  SetSplitMap(split_maps, fusion_nodes, FUSED_OP_TYPE);
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

  // check whether the relu/leakyrelu op
  for (const auto& reluNode : reluNodes) {
    if (elemWiseNodes.empty()) {
      if (reluNode->GetType() != "Relu" && reluNode->GetType() != "LeakyRelu" &&
          find(elemWiseWhiteList.begin(), elemWiseWhiteList.end(), reluNode->GetType()) == elemWiseWhiteList.end()) {
        fusionNodes.clear();
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
                reluNode->GetName().c_str(), reluNode->GetType().c_str());
        return SUCCESS;
      }
    } else {
      if (reluNode->GetType() != "Relu" && reluNode->GetType() != "LeakyRelu") {
        fusionNodes.clear();
        OP_LOGD(FUSED_OP_TYPE.c_str(), "Eltwise op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
                reluNode->GetName().c_str(), reluNode->GetType().c_str());
        return SUCCESS;
      }
      float negative_slope = 0;
      if (reluNode->GetType() == "LeakyRelu") {
        if (!ge::AttrUtils::GetFloat(reluNode->GetOpDesc(), "negative_slope", negative_slope)) {
          OP_LOGE(FUSED_OP_TYPE.c_str(), "LeakyRelu op[%s] type[%s] node does not have negative slope attr!",
                  reluNode->GetName().c_str(), reluNode->GetType().c_str());
          return FAILED;
        }
        if (negative_slope != 0) {
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
    if (find(elemWiseWhiteList.begin(), elemWiseWhiteList.end(), elemWiseNode->GetType()) ==
        elemWiseWhiteList.end()) {
      fusionNodes.clear();
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "Eltwise op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
              elemWiseNode->GetName().c_str(), elemWiseNode->GetType().c_str());
      return SUCCESS;
    }
  }
  SetSplitInfo(mapping, fusionNodes);
  OP_LOGD("End to do TbeFullyconnectionElemwiseFusionPass!");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeFullyconnectionElemwiseDequantFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeFullyconnectionElemwiseFusionPass);
}  // namespace fe
