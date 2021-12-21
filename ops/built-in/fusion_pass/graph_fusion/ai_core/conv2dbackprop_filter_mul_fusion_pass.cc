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

/*!
 * \file conv2dbackprop_filter_mul_fusion_pass.cc
 * \brief conv2dbackprop_filter_mul_fusion pass
 */

#include "conv2dbackprop_filter_mul_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "quant_host_cpu_op_common.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "securec.h"
#include "common/util/error_manager/error_manager.h"
#include "../../../op_proto/util/error_util.h"
#include "anchor_util.h"

using namespace std;
using namespace ge;

namespace fe {
static const float FLOAT_NUM_ONE = 1;
static const std::string PATTERN_CONV2DBPFILTER = "Conv2DBackpropFilterD";
static const std::string CONSTANTOP = "Const";
static const std::string CONV2DBPFILTER = "Conv2DBackpropFilterD";
static const int kDimSize = 4;
static const int kIndex2 = 2;
static const int kIndex3 = 3;

/*!
 * @brief: parse nodes matched in mapping and call graph DoFusion
 * @param [in] graph: original graph
 * @param [in] dwOutNode: the pointer of dw node
 * @param [in] inputOriginFormat: the origin format of dw node's input
 * @return NodePtr: the pointer of mul node
 */
NodePtr Conv2DbpFilterMulFusionPass::AddMul(ge::ComputeGraph& graph,
                                            ge::NodePtr& dwOutNode,
                                            ge::Format& inputOriginFormat) {
  ge::NodePtr mulNode = nullptr;

  // create a new node desc
  ge::OpDescPtr mulDesc;
  FUSION_PASS_MAKE_SHARED(mulDesc = std::make_shared<ge::OpDesc>(dwOutNode->GetName() + "_mul_layer", "Mul"),
                          return nullptr);
  // get and set mulDesc's inputDesc
  auto dwOutNodePtr = dwOutNode->GetOpDesc();
  FUSION_PASS_CHECK(dwOutNodePtr == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dwOutNodePtr is null"),
                    return nullptr);
  ge::GeTensorDesc inputDesc = dwOutNodePtr->GetOutputDesc(0);
  ge::GeShape mulShape = inputDesc.GetShape();
  inputDesc.SetShape(mulShape);
  inputDesc.SetOriginShape(mulShape);
  inputDesc.SetOriginFormat(inputOriginFormat);
  inputDesc.SetDataType(ge::DT_FLOAT);
  inputDesc.SetOriginDataType(ge::DT_FLOAT);
  // create and set mulDesc's outputDesc
  ge::GeTensorDesc outputDesc;
  outputDesc.SetShape(mulShape);
  outputDesc.SetOriginShape(mulShape);
  outputDesc.SetOriginFormat(inputOriginFormat);
  outputDesc.SetDataType(ge::DT_FLOAT);
  outputDesc.SetOriginDataType(ge::DT_FLOAT);
  // mulDesc setInput inputDesc & outputDesc
  FUSION_PASS_CHECK(mulDesc->AddInputDesc(inputDesc) != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mulDesc input failed"),
                    return nullptr);
  FUSION_PASS_CHECK(mulDesc->AddOutputDesc(outputDesc) != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mulDesc output failed"),
                    return nullptr);
  // graph add mulNode by mulDesc
  mulNode = graph.AddNode(mulDesc);
  FUSION_PASS_CHECK(mulNode == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mul node failed"),
                    return nullptr);
  // modify edge info
  ge::OutDataAnchorPtr dwAnchorPtr1 = dwOutNode->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(dwAnchorPtr1 == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get data anchor 0 of output failed"),
                    return nullptr);
  for (auto postAnchorPtr0 : dwAnchorPtr1->GetPeerInDataAnchors()) {
    /*
     * dwAnchorPtr1 : the edge to dw node
     * postAnchorPtr0 : the edges from dw node
     *       |    ---> outdata anchor (dwAnchorPtr1)
     *       v
     *       dw
     *     |  |  |  ---> indata anchors (postAnchorPtr0)
     *     v  v  v
     */

    // remove edge between dw and next node
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(postAnchorPtr0, dwAnchorPtr1) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(),
                              "remove edge between dw and dw's next node failed"),
                      return nullptr);

    // add edge between mul and next node
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mulNode->GetOutDataAnchor(0), postAnchorPtr0) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(),
                              "add edge between mul node and dw's next node failed"),
                      return nullptr);
  }
  // add edge between dw and mul
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(dwAnchorPtr1, mulNode->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add edge between dw node and mul node failed"),
                    return nullptr);
  return mulNode;
}

/*!
 * @brief: add a const node to mul node
 * @param [in] mulNode: the pointer of mul node
 * @param [in] matrixSize: the size of const node
 * @return success for add a const node to mul node
 */
Status Conv2DbpFilterMulFusionPass::AddAssit(ge::NodePtr& mulNode,
                                             const int64_t matrixSize) const {
  // get OriginDesc info
  ge::ConstGeTensorDescPtr inputDesc0 = GetCurrNodeInputDesc(mulNode, 0);
  FUSION_PASS_CHECK(inputDesc0 == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc0 is null"),
                    return FAILED);
  ge::Format inputDesc0OriginFormat = inputDesc0->GetOriginFormat();
  ge::GeShape inputDesc0Shape = inputDesc0->GetOriginShape();
  vector<int64_t> inDimInfo  = inputDesc0Shape.GetDims();

  // create inputAssit & fill data by NnSet
  FUSION_PASS_CHECK(matrixSize <= 0,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "matrixSize id Invalid"), return PARAM_INVALID);
  unique_ptr<float[]> inputAssit(new (std::nothrow) float[matrixSize]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                    return PARAM_INVALID);
  Status ret = NnSet(matrixSize, FLOAT_NUM_ONE, *reinterpret_cast<float*>(inputAssit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed"), return ret);

  // create and set assitDesc
  ge::GeTensorDesc assitDesc;
  ge::GeShape assitShapeOrigin(inDimInfo);
  assitDesc.SetFormat(inputDesc0OriginFormat);
  assitDesc.SetOriginFormat(inputDesc0OriginFormat);
  assitDesc.SetShape(assitShapeOrigin);
  assitDesc.SetOriginShape(assitShapeOrigin);
  assitDesc.SetDataType(ge::DT_FLOAT);
  assitDesc.SetOriginDataType(ge::DT_FLOAT);

  // create assitTensorPtr by assitDesc & add assit node by SetWeights
  ge::GeTensorPtr assitPtr = nullptr;
  FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(
                                        assitDesc,
                                        reinterpret_cast<uint8_t*>(inputAssit.get()),
                                        matrixSize * sizeof(float))),
                          assitPtr = nullptr;
                          return PARAM_INVALID);
  vector<ge::GeTensorPtr> weights = {assitPtr};
  ge::OpDescUtils::SetWeights(mulNode, weights);

  // set constInput Type to 'Const'
  auto constInputNodes = OpDescUtils::GetConstInputs(mulNode);
  NodePtr constInput = nullptr;
  if (constInputNodes.size() != 0) {
    constInput = constInputNodes[0];
  } else {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constInputNodes is null, fusion failed");
    return PARAM_INVALID;
  }
  auto const_op_desc = constInput->GetOpDesc();
  FUSION_PASS_CHECK(const_op_desc == nullptr,
                  CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dwOutNodePtr is null"),
                  return FAILED);
  constInput->GetOpDesc()->SetType(CONSTANTOP);

  return SUCCESS;
}

/*!
 * @brief: Define dw+mul pattern.
 * The graph struct need to adapt is shown as follows:
 *
 *            dw   const
 *             |  /
 *            mul
 *             |
 *           output
 *
 *  Notice: the struct can be captured by
 *      dw + mul pattern
 *  @return vector<FusionPattern*> All valid patterns.
 */

vector<FusionPattern*> Conv2DbpFilterMulFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("Conv2DbpFilterMulFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new pattern obj failed"),
                    return patterns);
  pattern->AddOpDesc(PATTERN_CONV2DBPFILTER, {CONV2DBPFILTER}).SetOutput(PATTERN_CONV2DBPFILTER);
  patterns.push_back(pattern);
  return patterns;
}

/*!
 * @brief: parse nodes matched in mapping and call graph DoFusion
 * @param [in] graph: original graph
 * @param [in] mapping: matched pattern
 * @param [out] newNodes: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status Conv2DbpFilterMulFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                           vector<ge::NodePtr>& /* new_nodes */) {
  // dwNode info
  ge::NodePtr dwNode = GetNodeFromMapping(PATTERN_CONV2DBPFILTER, mapping);
  FUSION_PASS_CHECK(dwNode == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dw Node is null, fusion failed"),
                    return PARAM_INVALID);
  ge::OpDescPtr dwDesc = dwNode->GetOpDesc();
  FUSION_PASS_CHECK(dwDesc == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dw Node's Desc is null, fusion failed"),
                    return PARAM_INVALID);
  ge::GeTensorDesc dwOutputDesc = dwDesc->GetOutputDesc(0);
  ge::GeShape dwOutputShape = dwOutputDesc.GetShape();
  ge::Format dwOutputOriginFormat = dwOutputDesc.GetOriginFormat();

  // get dw's out node shape info
  vector<int64_t> outputDimInfo = dwOutputShape.GetDims();
  int64_t filterN = 0;
  int64_t filterC = 0;
  int64_t filterH = 0;
  int64_t filterW = 0;
  int64_t groups = 0;
  ge::AttrUtils::GetInt(dwNode->GetOpDesc(), "groups", groups);
  if (groups <= 1) {
    return NOT_CHANGED;
  }
  if (outputDimInfo.size() == kDimSize) {
    if (dwOutputOriginFormat == FORMAT_NHWC){
      filterN = outputDimInfo[0];
      filterH = outputDimInfo[1];
      filterW = outputDimInfo[kIndex2];
      filterC = outputDimInfo[kIndex3];
    } else if (dwOutputOriginFormat == FORMAT_NCHW){
      filterN = outputDimInfo[0];
      filterC = outputDimInfo[1];
      filterH = outputDimInfo[kIndex2];
      filterW = outputDimInfo[kIndex3];
    } else if (dwOutputOriginFormat == FORMAT_HWCN){
      filterH = outputDimInfo[0];
      filterW = outputDimInfo[1];
      filterC = outputDimInfo[kIndex2];
      filterN = outputDimInfo[kIndex3];
    } else {
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outputOriginFormat only support NHWC and NCHW and HWCN");
      return NOT_CHANGED;
    }
  } else {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "dimInfo is not right");
    return NOT_CHANGED;
  }
  int64_t matrixSize = filterN *  filterC * filterH * filterW;
  FUSION_PASS_CHECK(matrixSize <= 0,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "matrixSize Invalid"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(filterN % groups != 0,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "filterN is not a multiple of groups"),
                    return PARAM_INVALID);

  // add nodes
  ge::NodePtr mulNode = AddMul(graph, dwNode, dwOutputOriginFormat);
  FUSION_PASS_CHECK(mulNode == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mul Node failed"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(AddAssit(mulNode, matrixSize) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add assit failed"),
                    return PARAM_INVALID);

  return SUCCESS;
}

REGISTER_PASS("Conv2DbpFilterMulFusionPass", BUILT_IN_GRAPH_PASS, Conv2DbpFilterMulFusionPass);
} // namespace fe
