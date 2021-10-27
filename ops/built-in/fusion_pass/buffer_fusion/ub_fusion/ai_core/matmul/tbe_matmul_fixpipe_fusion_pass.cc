/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "tbe_matmul_fixpipe_fusion_pass.h"
#include <string>
#include <vector>
#include <unordered_set>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/util/platform_info.h"

namespace fe {

static const char kTypeTransData1[] = "transdata1";
static const char kPatternCube[] = "cube";
static const char kPatternGEMM[] = "GEMM";
static const char kPatternQuant[] = "quant";
static const char kPatternElemwise[] = "elemwise";
static const char kTypeTransData2[] = "transdata2";
static const string kOpTypeTransData = "TransData";
static const int kFusionOpNumMax = 10;
static const string kFusedOpType = "FusedOp";

// white list of OP_PATTERN_ELEMWISE
static const std::unordered_set<string> kWhiteListOfElemwiseNode = { "LeakyRelu", "Relu", "PRelu" };

/*
 * @brief: define transdata_cube fusion pattern
 *
 *  (Transdata)? + Cube + (Dequent/Quent/Requent)? + (Elemwise)? + (Transdata)?
 *  pattern limit:
 *          1.Transdata,Dequent/Quent/Requent,Elemwise are optional,Cube is required.
 *          2.Elemwise supports LeakyRelu,Relu,PRelu
 *          3.Cube supports Matmul,Conv2d,Conv_dx,Conv_dw.
 *          4.Matmul only support Matmul
 *
 * @return BufferFusionPattern: return all valid patterns
 */
vector<BufferFusionPattern *> TbeMatmulFixpipeFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;
  string pass_name = "TbeMatmulFixpipeFusionPass";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name, kFusionOpNumMax);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  // define pattern rules
  pattern->AddOpDesc(kTypeTransData1, {kOpTypeTransData},
                   TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE, true)
        .AddOpDesc(kPatternCube, {kPatternGEMM},
                   TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
        .AddOpDesc(kPatternQuant, {OP_PATTERN_DEQUANT, OP_PATTERN_QUANT, OP_PATTERN_REQUANT},
                   TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
        .AddOpDesc(kPatternElemwise, {OP_PATTERN_ELEMWISE},
                   TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
        .AddOpDesc(kTypeTransData2, {kOpTypeTransData},
                   TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE, true)
        .SetHead({kTypeTransData1, kPatternCube})
        .SetOutputs(kTypeTransData1, {kPatternCube})
        .SetOutputs(kPatternCube, {kPatternQuant}, TBE_OUTPUT_BRANCH_SINGLE, true)
        .SetOutputs(kPatternQuant, {kPatternElemwise}, TBE_OUTPUT_BRANCH_SINGLE, true)
        .SetOutputs(kPatternElemwise, {kTypeTransData2});

  patterns.push_back(pattern);

  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name.c_str());
  return patterns;
}

bool TbeMatmulFixpipeFusionPass::IsInWhiteListOfElemwiseOp(const vector<ge::NodePtr> &elemwise_nodes) {
  for (auto &elemwise_node : elemwise_nodes) {
    string op_type = elemwise_node->GetType();
    auto count = kWhiteListOfElemwiseNode.count(op_type);
    if (count == 0) {
      OP_LOGD(kFusedOpType.c_str(), "node:%s[type:%s] not in elemwise white_list.",
              elemwise_node->GetName().c_str(), op_type.c_str());
      return false;
    } else if (op_type == "LeakyRelu" or op_type == "PRelu") {
      if (elemwise_node->GetOpDesc()->GetInputDesc(0).GetDataType() != ge::DT_FLOAT16) {
        OP_LOGD(kFusedOpType.c_str(), "node:%s[type:%s] only support fp16.",
                elemwise_node->GetName().c_str(), op_type.c_str());
        return false;
      }
    }
  }
  return true;
}

bool TbeMatmulFixpipeFusionPass::MatmulSupportTrans(const ge::NodePtr &node, const bool &is_input) {
  if(is_input) {
    return node->GetOpDesc()->GetInputDesc(0).GetFormat() == ge::FORMAT_ND &&
           node->GetOpDesc()->GetOutputDesc(0).GetFormat() == ge::FORMAT_FRACTAL_NZ;
  } else {
    return node->GetOpDesc()->GetInputDesc(0).GetFormat() == ge::FORMAT_FRACTAL_NZ &&
           node->GetOpDesc()->GetOutputDesc(0).GetFormat() == ge::FORMAT_ND;
  }
}

void TbeMatmulFixpipeFusionPass::CheckCubeSupportTransNodes(const vector<ge::NodePtr> &cube_nodes,
                                                            const vector<ge::NodePtr> &transdata1_nodes,
                                                            const vector<ge::NodePtr> &transdata2_nodes,
                                                            vector<ge::NodePtr> &fusion_nodes) {
  if (cube_nodes.empty()) {
    return;
  }
  ge::NodePtr cube_node = cube_nodes.at(0);
  bool cube_is_matmul = cube_node->GetType() == "MatMulV2" || cube_node->GetType() == "MatMul";

  if (cube_is_matmul) {
    if (!transdata1_nodes.empty()) {
      bool weight_trans = true;
      if (!MatmulSupportTrans(transdata1_nodes[0], true)) {
        auto iter = find(fusion_nodes.begin(), fusion_nodes.end(), transdata1_nodes[0]);
        if (iter != fusion_nodes.end()) {
          fusion_nodes.erase(iter);
          weight_trans = false;
        }
      }
      if (weight_trans && cube_node->GetInDataNodes().size() >= 2) {
        ge::NodePtr weight_trans_node = cube_node->GetInDataNodes().at(1);
        if (weight_trans_node->GetType() == kOpTypeTransData && MatmulSupportTrans(weight_trans_node, true)) {
          fusion_nodes.push_back(weight_trans_node);
        }
      }
    }
    if (!transdata2_nodes.empty()) {
      if (!MatmulSupportTrans(transdata2_nodes[0], false)) {
        auto iter = find(fusion_nodes.begin(), fusion_nodes.end(), transdata2_nodes[0]);
        if (iter != fusion_nodes.end()) {
          fusion_nodes.erase(iter);
        }
      }
    }
  } else {
    fusion_nodes.clear();
  }
  return;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeMatmulFixpipeFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                  vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(kFusedOpType.c_str(), "Begin to do TbeMatmulFixpipeFusionPass.");
  PlatformInfo platformInfo;
  OptionalInfo optionalInfo;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
    OP_LOGW(kFusedOpType.c_str(), "Get platform info failed, not fusion.");
    return SUCCESS;
  }
  if (platformInfo.ai_core_spec.cube_vector_split != 1) {
    OP_LOGD(kFusedOpType.c_str(), "Fusion pass not support cube vector split.");
    return SUCCESS;
  }
  fusion_nodes = GetMatchedNodes(mapping);

  vector<ge::NodePtr> transdata1_nodes = GetMatchedNodesByDescName(kTypeTransData1, mapping);
  vector<ge::NodePtr> cube_nodes = GetMatchedNodesByDescName(kPatternCube, mapping);
  vector<ge::NodePtr> elemwise_nodes = GetMatchedNodesByDescName(kPatternElemwise, mapping);
  vector<ge::NodePtr> transdata2_nodes = GetMatchedNodesByDescName(kTypeTransData2, mapping);
  if (!elemwise_nodes.empty()) {
    if (!IsInWhiteListOfElemwiseOp(elemwise_nodes)) {
      fusion_nodes.clear();
      return SUCCESS;
    }
  }
  CheckCubeSupportTransNodes(cube_nodes, transdata1_nodes, transdata2_nodes, fusion_nodes);

  if (fusion_nodes.size() == 1) {
    fusion_nodes.clear();
  }
  OP_LOGD(kFusedOpType.c_str(), "End to do TbeMatmulFixpipeFusionPass!");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeMatmulFixpipeFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeMatmulFixpipeFusionPass);
}  // namespace fe
