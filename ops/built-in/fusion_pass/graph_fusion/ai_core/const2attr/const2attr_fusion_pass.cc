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

/*!
 * \file const2attr_fusion_pass.cpp
 * \brief all const2attr pass.
 */
#include "const2attr_fusion_pass.h"
#include <string>
#include <vector>
#include "op_log.h"
#include "error_util.h"
#include "fusion_const2attr_registry.h"
#include "fusion_precheck_func.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "graph_optimizer/fusion_common/graph_pass_util.h"
#include "graph_optimizer/fusion_common/fusion_statistic_recorder.h"
#include "pattern_fusion_util.h"

namespace fe {
REGISTER_PASS("ConstToAttrPass", BUILT_IN_GRAPH_PASS, Const2AttrFusionPass);

vector<FusionPattern*> Const2AttrFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  return patterns;
}

Status Const2AttrFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  return SUCCESS;
}
Status Const2AttrFusionPass::Run(ge::ComputeGraph& graph, OpsKernelInfoStorePtr opsKernelInfoStorePtr) {
  FUSION_PASS_CHECK(opsKernelInfoStorePtr == nullptr, 
                    VECTOR_FUSION_INNER_ERR_REPORT("Const2AttrFusion", "opsKernelInfoStorePtr is nullptr"),
                    return FAILED);

  int32_t matchTimes = 0;
  int32_t effectTimes = 0;
  for (ge::NodePtr& node : graph.GetDirectNode()) {
    string oriOpType = node->GetOpDesc()->GetType();
    FusionConst2AttrOpRegister reg("");
    if (FusionConst2AttrOpRegistry::Instance()->GetRegisterByOriType(oriOpType, reg) != SUCCESS) {
      continue;
    }
    matchTimes++;
    std::string opType;
    bool needCheck = false;
    std::vector<PassAttrInfo> attrVec;
    function<Status(ge::NodePtr)> preCheckFunc;
    reg.GetAttrInfo(opType, needCheck, attrVec, preCheckFunc);

    if (preCheckFunc != nullptr) {
      Status preCheckFuncResult = preCheckFunc(node);
      if (preCheckFuncResult != SUCCESS && preCheckFuncResult != NOT_CHANGED) {
        OP_LOGD(opType.c_str(), "node:%s type:%s const to attr preCheck failed, return failed.",
                node->GetName().c_str(), opType.c_str());
        return FAILED;
      }

      if (preCheckFuncResult == NOT_CHANGED) {
        OP_LOGD(opType.c_str(), "node:%s type:%s const to attr preCheck failed, not change.", node->GetName().c_str(),
                opType.c_str());
        continue;
      }
    }

    if (needCheck) {
      bool unknownShape = false;
      bool is_dynamic = ((ge::NodeUtils::GetNodeUnknownShapeStatus(*(node.get()), unknownShape) == ge::GRAPH_SUCCESS &&
                         unknownShape));
      if (is_dynamic) {
        std::string oriUnSupportedReason;
        bool isOriSupported = opsKernelInfoStorePtr->CheckSupported(node, oriUnSupportedReason);
        if (isOriSupported) {
          OP_LOGD(opType.c_str(), "Op[name:%s,type:%s] is supported", node->GetOpDesc()->GetName().c_str(),
                  opType.c_str());
          continue;
        } else {
          OP_LOGD(opType.c_str(), "Op[name:%s,type:%s] is not supported, reason is %s.",
                  node->GetOpDesc()->GetName().c_str(), node->GetOpDesc()->GetType().c_str(),
                  oriUnSupportedReason.c_str());
        }
      }

      ge::OpDescPtr opDescPtr = PatternFusionUtil::GetFusionOpDesc(node, opType, attrVec);
      if (opDescPtr == nullptr) {
        OP_LOGW(opType.c_str(), "OpDesc %s is nullptr", opType.c_str());
        continue;
      }

      std::string unSupportedReason;
      bool isSupported = opsKernelInfoStorePtr->CheckSupported(opDescPtr, unSupportedReason);
      if (!isSupported) {
        OP_LOGI(opType.c_str(), "%s not supported, %s", opType.c_str(), unSupportedReason.c_str());
        continue;
      }
    }

    // for op data dump
    std::vector<ge::NodePtr> originalNodes;
    Status dumpret = PatternFusionUtil::RecordOriginalNamesForConstToAttr(node, attrVec, originalNodes);
    if (dumpret != SUCCESS) {
      OP_LOGW(opType.c_str(), "node:%s RecordOriginalNames for DataDump not success.", node->GetName().c_str());
    }

    ge::NodePtr fusionNode = nullptr;
    Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, node, opType, attrVec, fusionNode);
    if (ret != SUCCESS) {
      OP_LOGI(opType.c_str(), "%s has input which is not a constant, graph not changed.", oriOpType.c_str());
      continue;
    }
    effectTimes++;

    // for op data dump
    if (PatternFusionUtil::SetOutputDescAttrForDataDump(node, fusionNode) != SUCCESS) {
      OP_LOGW(opType.c_str(), "node:%s SetAttr for DataDump not success.", fusionNode->GetName().c_str());
    }
    GraphPassUtil::RecordOriginalNames(originalNodes, fusionNode);
    OP_LOGD(opType.c_str(), "%s const2attr success", oriOpType.c_str());
  }
  // save matchTimes and effectTimes
  FusionInfo fusionInfo(graph.GetSessionID(), to_string(graph.GetGraphID()), GetName(), matchTimes, effectTimes);
  FusionStatisticRecorder::Instance().UpdateGraphFusionMatchTimes(fusionInfo);
  FusionStatisticRecorder::Instance().UpdateGraphFusionEffectTimes(fusionInfo);
  OP_LOGD("ConstToAttrPass",
          "SessionId[%d], GraphId[%d], GraphFusionPass[%s]: pattern=undefined, matchedTimes=%d, effectedTimes=%d.",
          graph.GetSessionID(), graph.GetGraphID(), GetName().c_str(), matchTimes, effectTimes);

  return SUCCESS;
}

REGISTER_CONST2ATTR("ApplyRMSPropD")
    .OriginOpType("ApplyRMSProp")
    .SetPreCheckFunc(ApplyRmsPropPreCheck)
    .SetConstToAttr(4, "rho", "SetFloat")
    .SetConstToAttr(5, "momentum", "SetFloat")
    .SetConstToAttr(6, "epsilon", "SetFloat");

REGISTER_CONST2ATTR("ApplyAdaMaxD").OriginOpType("ApplyAdaMax");

REGISTER_CONST2ATTR("ApplyAdadeltaD").OriginOpType("ApplyAdadelta");

REGISTER_CONST2ATTR("ApplyAdagradD").OriginOpType("ApplyAdagrad");

REGISTER_CONST2ATTR("ApplyAdagradDAD").OriginOpType("ApplyAdagradDA");

REGISTER_CONST2ATTR("ApplyFtrlD").OriginOpType("ApplyFtrl");

REGISTER_CONST2ATTR("ApplyCenteredRMSPropD").OriginOpType("ApplyCenteredRMSProp");

REGISTER_CONST2ATTR("ApplyAdamD").NeedCheckSupported(true).OriginOpType("ApplyAdam");

REGISTER_CONST2ATTR("SparseApplyProximalAdagradD").OriginOpType("SparseApplyProximalAdagrad");

REGISTER_CONST2ATTR("ApplyAddSignD").OriginOpType("ApplyAddSign");

REGISTER_CONST2ATTR("ApplyPowerSignD").OriginOpType("ApplyPowerSign");

REGISTER_CONST2ATTR("ApplyProximalAdagradD").OriginOpType("ApplyProximalAdagrad");

REGISTER_CONST2ATTR("SparseApplyRMSPropD")
    .OriginOpType("SparseApplyRMSProp")
    .SetPreCheckFunc(SparseApplyRmsPropPreCheck)
    .SetConstToAttr(4, "rho", "SetFloat")
    .SetConstToAttr(5, "momentum", "SetFloat")
    .SetConstToAttr(6, "epsilon", "SetFloat");

REGISTER_CONST2ATTR("ApplyAdagradV2D")
    .OriginOpType("ApplyAdagradV2")
    .SetPreCheckFunc(ApplyAdagradV2PreCheck)
    .SetConstToAttr(3, "epsilon", "SetFloat");

REGISTER_CONST2ATTR("ArgMaxD")
    .OriginOpType("ArgMaxV2")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "dimension", "SetInt");

REGISTER_CONST2ATTR("ArgMinD").OriginOpType("ArgMin").NeedCheckSupported(true).SetConstToAttr(1, "dimension", "SetInt");

REGISTER_CONST2ATTR("CropAndResizeD")
    .OriginOpType("CropAndResize")
    .NeedCheckSupported(true)
    .SetConstToAttr(3, "crop_size", "SetListInt");

REGISTER_CONST2ATTR("BroadcastToD")
    .OriginOpType("BroadcastTo")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "shape", "SetListInt");

REGISTER_CONST2ATTR("BatchToSpaceNDD")
    .OriginOpType("BatchToSpaceND")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "block_shape", "SetListInt")
    .SetConstToAttr(2, "crops", "SetListInt");

REGISTER_CONST2ATTR("SpaceToBatchNDD")
    .OriginOpType("SpaceToBatchND")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "block_shape", "SetListInt")
    .SetConstToAttr(2, "paddings", "SetListInt");

REGISTER_CONST2ATTR("ScatterNdD")
    .OriginOpType("ScatterNd")
    .NeedCheckSupported(true)
    .SetConstToAttr(2, "shape", "SetListInt");

REGISTER_CONST2ATTR("Conv2DBackpropFilterD")
    .OriginOpType("Conv2DBackpropFilter")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "filter_size", "SetListInt");

REGISTER_CONST2ATTR("Conv2DBackpropInputD")
    .OriginOpType("Conv2DBackpropInput")
    .NeedCheckSupported(true)
    .SetConstToAttr(0, "input_size", "SetListInt");

REGISTER_CONST2ATTR("Conv3DBackpropFilterD")
    .OriginOpType("Conv3DBackpropFilter")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "filter_size", "SetListInt");

REGISTER_CONST2ATTR("Conv3DBackpropInputD")
    .OriginOpType("Conv3DBackpropInput")
    .SetConstToAttr(0, "input_size", "SetListInt");

REGISTER_CONST2ATTR("Conv2DTransposeD")
    .OriginOpType("Conv2DTranspose")
    .NeedCheckSupported(true)
    .SetConstToAttr(0, "input_size", "SetListInt");

REGISTER_CONST2ATTR("Conv3DTransposeD").OriginOpType("Conv3DTranspose").SetConstToAttr(0, "input_size", "SetListInt");

REGISTER_CONST2ATTR("DepthwiseConv2DBackpropFilterD")
    .OriginOpType("DepthwiseConv2DBackpropFilter")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "filter_size", "SetListInt");

REGISTER_CONST2ATTR("DepthwiseConv2DBackpropInputD")
    .OriginOpType("DepthwiseConv2DBackpropInput")
    .SetConstToAttr(0, "input_size", "SetListInt");
REGISTER_CONST2ATTR("FillD").OriginOpType("Fill").NeedCheckSupported(true).SetConstToAttr(0, "dims", "SetListInt");

REGISTER_CONST2ATTR("HistogramFixedWidthD")
    .OriginOpType("HistogramFixedWidth")
    .NeedCheckSupported(true)
    .SetConstToAttr(2, "nbins", "SetInt");

REGISTER_CONST2ATTR("InTopKD").OriginOpType("InTopK").NeedCheckSupported(true).SetConstToAttr(2, "k", "SetInt");

REGISTER_CONST2ATTR("InplaceAddD")
    .OriginOpType("InplaceAdd")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "indices", "SetListInt");

REGISTER_CONST2ATTR("InplaceSubD")
    .OriginOpType("InplaceSub")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "indices", "SetListInt");

REGISTER_CONST2ATTR("InplaceUpdateD")
    .OriginOpType("InplaceUpdate")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "indices", "SetListInt");

REGISTER_CONST2ATTR("MaxPoolExt2")
    .OriginOpType("MaxPoolV2")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "ksize", "SetListInt")
    .SetConstToAttr(2, "strides", "SetListInt");

REGISTER_CONST2ATTR("ReduceMeanD")
    .OriginOpType("ReduceMean")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "axes", "SetListInt");

REGISTER_CONST2ATTR("ReduceAllD")
    .OriginOpType("ReduceAll")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "axes", "SetListInt");

REGISTER_CONST2ATTR("ReduceAnyD")
    .OriginOpType("ReduceAny")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "axes", "SetListInt");

REGISTER_CONST2ATTR("ReduceMaxD")
    .OriginOpType("ReduceMax")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "axes", "SetListInt");

REGISTER_CONST2ATTR("ReduceMinD")
    .OriginOpType("ReduceMin")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "axes", "SetListInt");

REGISTER_CONST2ATTR("ReduceProdD")
    .OriginOpType("ReduceProd")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "axes", "SetListInt");

REGISTER_CONST2ATTR("CumprodD").OriginOpType("Cumprod").NeedCheckSupported(true).SetConstToAttr(1, "axis", "SetInt");

REGISTER_CONST2ATTR("CumsumD").OriginOpType("Cumsum").NeedCheckSupported(true).SetConstToAttr(1, "axis", "SetInt");

REGISTER_CONST2ATTR("ResizeBilinearV2D")
    .OriginOpType("ResizeBilinearV2")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "size", "SetListInt");

REGISTER_CONST2ATTR("ResizeNearestNeighborV2D")
    .OriginOpType("ResizeNearestNeighborV2")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "size", "SetListInt");

REGISTER_CONST2ATTR("ReverseV2D")
    .OriginOpType("ReverseV2")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "axis", "SetListInt");

REGISTER_CONST2ATTR("SliceD")
    .OriginOpType("Slice")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "offsets", "SetListInt")
    .SetConstToAttr(2, "size", "SetListInt");

REGISTER_CONST2ATTR("SparseApplyAdagradD").OriginOpType("SparseApplyAdagrad").SetConstToAttr(2, "lr", "SetFloat");

REGISTER_CONST2ATTR("SparseApplyAdagradV2D")
    .OriginOpType("SparseApplyAdagradV2")
    .SetConstToAttr(2, "lr", "SetFloat")
    .SetConstToAttr(3, "epsilon", "SetFloat");

REGISTER_CONST2ATTR("SparseApplyFtrlD")
    .OriginOpType("SparseApplyFtrl")
    .SetConstToAttr(5, "lr", "SetFloat")
    .SetConstToAttr(6, "l1", "SetFloat")
    .SetConstToAttr(7, "l2", "SetFloat")
    .SetConstToAttr(8, "lr_power", "SetFloat");

REGISTER_CONST2ATTR("SparseApplyFtrlV2D")
    .OriginOpType("SparseApplyFtrlV2")
    .SetConstToAttr(5, "lr", "SetFloat")
    .SetConstToAttr(6, "l1", "SetFloat")
    .SetConstToAttr(7, "l2", "SetFloat")
    .SetConstToAttr(8, "l2_shrinkage", "SetFloat")
    .SetConstToAttr(9, "lr_power", "SetFloat");

REGISTER_CONST2ATTR("StridedSliceAssignD")
    .OriginOpType("StridedSliceAssign")
    .SetConstToAttr(1, "begin", "SetListInt")
    .SetConstToAttr(2, "end", "SetListInt")
    .SetConstToAttr(3, "strides", "SetListInt");

REGISTER_CONST2ATTR("TransposeD")
    .OriginOpType("Transpose")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "perm", "SetListInt");

REGISTER_CONST2ATTR("UnsortedSegmentMinD")
    .OriginOpType("UnsortedSegmentMin")
    .NeedCheckSupported(true)
    .SetConstToAttr(2, "num_segments", "SetInt");

REGISTER_CONST2ATTR("UnsortedSegmentMaxD")
    .OriginOpType("UnsortedSegmentMax")
    .NeedCheckSupported(true)
    .SetConstToAttr(2, "num_segments", "SetInt");

REGISTER_CONST2ATTR("UnsortedSegmentProdD")
    .OriginOpType("UnsortedSegmentProd")
    .NeedCheckSupported(true)
    .SetConstToAttr(2, "num_segments", "SetInt");

REGISTER_CONST2ATTR("UnsortedSegmentSumD")
    .OriginOpType("UnsortedSegmentSum")
    .NeedCheckSupported(true)
    .SetConstToAttr(2, "num_segments", "SetInt");

REGISTER_CONST2ATTR("ClipBoxesD")
    .OriginOpType("ClipBoxes")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "img_size", "SetListInt");

REGISTER_CONST2ATTR("RpnProposalsD")
    .OriginOpType("RpnProposals")
    .NeedCheckSupported(true)
    .SetConstToAttr(2, "img_size", "SetListInt");

REGISTER_CONST2ATTR("ApplyFtrlV2D").OriginOpType("ApplyFtrlV2");

REGISTER_CONST2ATTR("ApplyMomentumD").OriginOpType("ApplyMomentum");

REGISTER_CONST2ATTR("ApplyKerasMomentumD").OriginOpType("ApplyKerasMomentum");

REGISTER_CONST2ATTR("SparseApplyAdadeltaD")
    .OriginOpType("SparseApplyAdadelta")
    .SetConstToAttr(5, "epsilon", "SetFloat");

REGISTER_CONST2ATTR("ApplyAdamWithAmsgradD")
    .OriginOpType("ApplyAdamWithAmsgrad")
    .SetConstToAttr(7, "beta1", "SetFloat")
    .SetConstToAttr(8, "beta2", "SetFloat")
    .SetConstToAttr(9, "epsilon", "SetFloat");

REGISTER_CONST2ATTR("EuclideanNormD")
    .OriginOpType("EuclideanNorm")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "axes", "SetListInt");

REGISTER_CONST2ATTR("DropOutDoMaskV3D")
    .OriginOpType("DropOutDoMaskV3")
    .NeedCheckSupported(true)
    .SetConstToAttr(2, "keep_prob", "SetFloat");

REGISTER_CONST2ATTR("CumulativeLogsumexpD")
    .OriginOpType("CumulativeLogsumexp")
    .NeedCheckSupported(true)
    .SetConstToAttr(1, "axis", "SetInt");

REGISTER_CONST2ATTR("FillV2D").OriginOpType("FillV2").NeedCheckSupported(true).SetConstToAttr(0, "dims", "SetListInt");

REGISTER_CONST2ATTR("ExpandD").OriginOpType("Expand").NeedCheckSupported(true).SetConstToAttr(1, "shape", "SetListInt");
}  // namespace fe
