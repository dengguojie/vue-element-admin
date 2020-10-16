/**
 * @file fe_graph_optimizer.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief all addoutput pass.
 *
 * @version 1.0
 *
 */

#include "apply_addoutput_fusion_pass.h"
#include "op_log.h"
#include "fusion_addoutput_registry.h"
#include "fusion_precheck_func.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include <string>
#include <vector>

namespace fe {
REGISTER_PASS("ApplyAddOutputPass", BUILT_IN_GRAPH_PASS, AddOutputFusionPass);

vector<FusionPattern*> AddOutputFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  return patterns;
}

Status AddOutputFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                   vector<ge::NodePtr>& newNodes) {
  return SUCCESS;
}

Status AddOutputFusionPass::Run(ge::ComputeGraph& graph,
                                OpsKernelInfoStorePtr opsKernelInfoStorePtr) {
  FUSION_PASS_CHECK(opsKernelInfoStorePtr == nullptr,
           OP_LOGE("ApplyAddOutput", "opsKernelInfoStorePtr is nullptr"), return FAILED);

  for (ge::NodePtr& node : graph.GetDirectNode()) {
    string opType = node->GetOpDesc()->GetType();
    FusionAddOutputOpRegister reg("");
    if (FusionAddOutputOpRegistry::Instance()->GetRegisterByOpType(
            opType, reg) != SUCCESS) {
      continue;
    }

    std::vector<PassInputInfo> inputInfoVec;
    function<bool(ge::NodePtr)> preCheckFunc;
    reg.GetInputInfo(inputInfoVec, preCheckFunc);

    if (preCheckFunc != nullptr) {
      if (!preCheckFunc(node)) {
        OP_LOGD(opType.c_str(), "node:%s type:%s add input to output preCheck failed.",
                node->GetName().c_str(), opType.c_str());
        continue;
      }
    }

    ge::NodePtr fusionNode = nullptr;
    Status ret = PatternFusionUtil::AddInputToOutput(node, inputInfoVec);
    if (ret != SUCCESS) {
      OP_LOGI(opType.c_str(), "node:%s type:%s add input to output failed.",
              node->GetName().c_str(), opType.c_str());
      return FAILED;
    }

    OP_LOGD(opType.c_str(), "node:%s type:%s addoutput success", node->GetName().c_str(),
            opType.c_str());
  }
  return SUCCESS;
}

REGISTER_ADDOUTPUT("ApplyRMSProp")
    .SetAddOutput(1, "ms")
    .SetAddOutput(2, "mom")
    .SetPreCheckFunc(ApplyRmsPropPreCheck);
REGISTER_ADDOUTPUT("FusedMulApplyMomentum")
    .SetAddOutput(1, "accum")
    .SetPreCheckFunc(FusedMulApplyMomentumPreCheck);
REGISTER_ADDOUTPUT("FusedMulApplyMomentumExtern")
    .SetAddOutput(1, "accum")
    .SetPreCheckFunc(FusedMulApplyMomentumExternPreCheck);
REGISTER_ADDOUTPUT("ApplyAdagrad")
    .SetAddOutput(1, "accum");
REGISTER_ADDOUTPUT("ApplyAdagradDA")
    .SetAddOutput(1, "gradient_accumulator")
    .SetAddOutput(2, "gradient_squared_accumulator");
REGISTER_ADDOUTPUT("ApplyAdadelta")
    .SetAddOutput(1, "accum")
    .SetAddOutput(2, "accum_update");
REGISTER_ADDOUTPUT("ApplyPowerSign")
    .SetAddOutput(1, "m");
REGISTER_ADDOUTPUT("ApplyProximalAdagrad")
    .SetAddOutput(1, "accum");
REGISTER_ADDOUTPUT("ApplyAdaMax")
    .SetAddOutput(1, "m")
    .SetAddOutput(2, "v");
REGISTER_ADDOUTPUT("ApplyAdagradV2")
    .SetAddOutput(1, "accum")
    .SetPreCheckFunc(ApplyAdagradV2PreCheck);
REGISTER_ADDOUTPUT("ApplyKerasMomentum")
    .SetAddOutput(1, "accum")
    .SetPreCheckFunc(ApplyKerasMomentumPreCheck);
REGISTER_ADDOUTPUT("SparseApplyFtrl")
    .SetAddOutput(1, "accum")
    .SetAddOutput(2, "linear")
    .SetPreCheckFunc(SparseApplyFtrlPreCheck);
REGISTER_ADDOUTPUT("SparseApplyFtrlV2")
    .SetAddOutput(1, "accum")
    .SetAddOutput(2, "linear")
    .SetPreCheckFunc(SparseApplyFtrlV2PreCheck);
REGISTER_ADDOUTPUT("SparseApplyAdagradV2")
    .SetAddOutput(1, "accum")
    .SetPreCheckFunc(SparseApplyAdagradV2PreCheck);
REGISTER_ADDOUTPUT("SparseApplyProximalAdagrad")
    .SetAddOutput(1, "accum");
REGISTER_ADDOUTPUT("SparseApplyAdagrad")
    .SetAddOutput(1, "accum");
REGISTER_ADDOUTPUT("ApplyFtrlV2")
    .SetAddOutput(1, "accum")
    .SetAddOutput(2, "linear");
REGISTER_ADDOUTPUT("ApplyMomentum")
    .SetAddOutput(1, "accum");
REGISTER_ADDOUTPUT("ApplyFtrl")
    .SetAddOutput(1, "accum")
    .SetAddOutput(2, "linear");
REGISTER_ADDOUTPUT("ApplyAdam")
    .SetAddOutput(1, "m")
    .SetAddOutput(2, "v");
REGISTER_ADDOUTPUT("ApplyCenteredRMSProp")
    .SetAddOutput(1, "mg")
    .SetAddOutput(2, "ms")
    .SetAddOutput(3, "mom");
REGISTER_ADDOUTPUT("ApplyAddSign")
    .SetAddOutput(1, "m");
REGISTER_ADDOUTPUT("SparseApplyRMSProp")
    .SetAddOutput(1, "ms")
    .SetAddOutput(2, "mom")
    .SetPreCheckFunc(SparseApplyRmsPropPreCheck);
REGISTER_ADDOUTPUT("SparseApplyAdadelta")
    .SetAddOutput(1, "accum")
    .SetAddOutput(2, "accum_update")
    .SetPreCheckFunc(SparseApplyAdadeltaPreCheck);
REGISTER_ADDOUTPUT("ApplyAdamWithAmsgrad")
    .SetAddOutput(1, "m")
    .SetAddOutput(2, "v")
    .SetAddOutput(3, "vhat");
}  // namespace fe
